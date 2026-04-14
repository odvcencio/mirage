package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	compressAIBalle2018Source           = "https://interdigitalinc.github.io/CompressAI/_modules/compressai/zoo/image.html"
	defaultCompressAIBalleArchitectures = "bmshj2018-factorized,bmshj2018-hyperprior"
)

var compressAIBalle2018MSEURLs = map[string]map[int]string{
	"bmshj2018-factorized": {
		1: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-1-446d5c7f.pth.tar",
		2: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-2-87279a02.pth.tar",
		3: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-3-5c6f152b.pth.tar",
		4: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-4-1ed4405a.pth.tar",
		5: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-5-866ba797.pth.tar",
		6: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-6-9b02ea3a.pth.tar",
		7: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-7-6dfd6734.pth.tar",
		8: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-factorized-prior-8-5232faa3.pth.tar",
	},
	"bmshj2018-hyperprior": {
		1: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-1-7eb97409.pth.tar",
		2: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-2-93677231.pth.tar",
		3: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-3-6d87be32.pth.tar",
		4: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-4-de1b779c.pth.tar",
		5: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-5-f8b614e1.pth.tar",
		6: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-6-1ab9c41e.pth.tar",
		7: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-7-3804dcbd.pth.tar",
		8: "https://compressai.s3.amazonaws.com/models/v1/bmshj2018-hyperprior-8-a583f0cf.pth.tar",
	},
}

func runFetchCompressAIBaseline(args []string) error {
	fs := flag.NewFlagSet("fetch-compressai-baseline", flag.ExitOnError)
	outDir := fs.String("out-dir", "baselines/compressai", "directory for CompressAI baseline checkpoints and manifest")
	architecturesFlag := defaultCompressAIBalleArchitectures
	fs.StringVar(&architecturesFlag, "architecture", defaultCompressAIBalleArchitectures, "comma-separated CompressAI architecture names")
	fs.StringVar(&architecturesFlag, "architectures", defaultCompressAIBalleArchitectures, "comma-separated CompressAI architecture names")
	metric := fs.String("metric", "mse", "CompressAI metric")
	qualitiesFlag := fs.String("qualities", "1,2,3,4,5,6,7,8", "comma-separated quality levels")
	force := fs.Bool("force", false, "redownload files that already exist")
	manifestOnly := fs.Bool("manifest-only", false, "write manifest without downloading checkpoints")
	timeout := fs.Duration("timeout", 30*time.Minute, "per-command download timeout")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *metric != "mse" {
		return fmt.Errorf("only -metric mse is wired for the v1 baseline")
	}
	architectures, err := parseStringList(architecturesFlag)
	if err != nil {
		return err
	}
	qualities, err := parseIntList(*qualitiesFlag)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	client := &http.Client{}
	manifest := compressAIBaselineManifest{
		Architectures: append([]string(nil), architectures...),
		Metric:        *metric,
		Source:        compressAIBalle2018Source,
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339),
		Checkpoints:   make([]compressAIBaselineCheckpoint, 0, len(architectures)*len(qualities)),
	}
	for _, architecture := range architectures {
		urls := compressAIBalle2018MSEURLs[architecture]
		if urls == nil {
			return fmt.Errorf("unsupported CompressAI architecture %q", architecture)
		}
		for _, quality := range qualities {
			url := urls[quality]
			if url == "" {
				return fmt.Errorf("unsupported CompressAI quality %d for architecture %q", quality, architecture)
			}
			path := filepath.Join(*outDir, filepath.Base(url))
			item := compressAIBaselineCheckpoint{
				Architecture: architecture,
				Quality:      quality,
				URL:          url,
				Path:         path,
			}
			if !*manifestOnly {
				bytes, sha, downloaded, err := fetchCompressAICheckpoint(ctx, client, url, path, *force)
				if err != nil {
					return err
				}
				item.Bytes = bytes
				item.SHA256 = sha
				item.Downloaded = downloaded
			}
			manifest.Checkpoints = append(manifest.Checkpoints, item)
			fmt.Printf("architecture=%s quality=%d path=%s downloaded=%t bytes=%d\n", item.Architecture, item.Quality, item.Path, item.Downloaded, item.Bytes)
		}
	}
	data, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	manifestPath := filepath.Join(*outDir, "manifest.json")
	if err := os.WriteFile(manifestPath, append(data, '\n'), 0o644); err != nil {
		return err
	}
	fmt.Printf("manifest=%s\n", manifestPath)
	return nil
}

func fetchCompressAICheckpoint(ctx context.Context, client *http.Client, url, path string, force bool) (int64, string, bool, error) {
	if !force {
		if info, err := os.Stat(path); err == nil && info.Size() > 0 {
			sha, err := sha256File(path)
			return info.Size(), sha, false, err
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return 0, "", false, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return 0, "", false, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, "", false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, "", false, fmt.Errorf("download %s: %s", url, resp.Status)
	}
	tmp := path + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return 0, "", false, err
	}
	hasher := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(out, hasher), resp.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return 0, "", false, copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return 0, "", false, closeErr
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return 0, "", false, err
	}
	return n, hex.EncodeToString(hasher.Sum(nil)), true, nil
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hasher := sha256.New()
	if _, err := io.Copy(hasher, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func parseIntList(value string) ([]int, error) {
	var out []int
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		parsed, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q", part)
		}
		out = append(out, parsed)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one value is required")
	}
	return out, nil
}

func parseStringList(value string) ([]string, error) {
	var out []string
	seen := map[string]bool{}
	for _, part := range strings.Split(value, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if seen[part] {
			continue
		}
		seen[part] = true
		out = append(out, part)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("at least one value is required")
	}
	return out, nil
}

type compressAIBaselineManifest struct {
	Architectures []string                       `json:"architectures"`
	Metric        string                         `json:"metric"`
	Source        string                         `json:"source"`
	GeneratedAt   string                         `json:"generated_at"`
	Checkpoints   []compressAIBaselineCheckpoint `json:"checkpoints"`
}

type compressAIBaselineCheckpoint struct {
	Architecture string `json:"architecture"`
	Quality      int    `json:"quality"`
	URL          string `json:"url"`
	Path         string `json:"path"`
	Bytes        int64  `json:"bytes,omitempty"`
	SHA256       string `json:"sha256,omitempty"`
	Downloaded   bool   `json:"downloaded"`
}
