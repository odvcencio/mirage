package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/mirage"
)

func runEvalMantaKodak(args []string) error {
	fs := flag.NewFlagSet("eval-manta-kodak", flag.ExitOnError)
	var inputs stringListFlag
	var dirs stringListFlag
	fs.Var(&inputs, "in", "input PNG, JPEG, or PPM; repeat or comma-separate")
	fs.Var(&dirs, "dir", "directory of PNG, JPEG, or PPM images; repeat or comma-separate")
	maxImages := fs.Int("max-images", 0, "maximum decoded images to evaluate; <=0 means all")
	runDir := fs.String("run-dir", "", "directory containing mirage_v1_lambda_*.mll and .weights.mll checkpoints")
	modulePath := fs.String("manta-module", "", "single Manta Mirage v1 .mll module")
	weightPath := fs.String("manta-weights", "", "single Manta Mirage v1 .weights.mll file")
	fs.StringVar(modulePath, "module", "", "alias for -manta-module")
	fs.StringVar(weightPath, "weights", "", "alias for -manta-weights")
	evalBackend := fs.String("eval-backend", "webgpu", "Manta eval backend: webgpu or reference")
	outDir := fs.String("out-dir", "", "optional directory for .mrg artifacts and eval_summary.json")
	jsonOut := fs.Bool("json", false, "emit machine-readable JSON summary to stdout")
	allowMismatch := fs.Bool("allow-model-mismatch", false, "allow Manta module fingerprint mismatch")
	if err := fs.Parse(args); err != nil {
		return err
	}
	paths, err := collectTrainingImagePaths(inputs, dirs)
	if err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("eval-manta-kodak requires at least one -in image or -dir")
	}
	if *maxImages > 0 && len(paths) > *maxImages {
		paths = paths[:*maxImages]
	}
	models, err := collectMantaEvalModels(*runDir, *modulePath, *weightPath)
	if err != nil {
		return err
	}
	if *outDir != "" {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			return err
		}
	}
	summary := evalMantaKodakSummary{
		Images: append([]string(nil), paths...),
		Models: make([]evalMantaKodakModelSummary, 0, len(models)),
	}
	ctx := context.Background()
	for _, model := range models {
		codec, err := mirage.LoadMantaCodec(ctx, mirage.MantaCodecOptions{
			ModulePath:         model.ModulePath,
			WeightPath:         model.WeightPath,
			Backend:            *evalBackend,
			AllowModelMismatch: *allowMismatch,
		})
		if err != nil {
			return err
		}
		run := evalMantaKodakModelSummary{
			Label:      model.Label,
			Lambda:     model.Lambda,
			ModulePath: model.ModulePath,
			WeightPath: model.WeightPath,
			Images:     make([]evalMantaKodakImageSummary, 0, len(paths)),
		}
		for _, path := range paths {
			metric, err := evalMantaKodakImage(ctx, codec, path, *outDir, model.Label)
			if err != nil {
				return err
			}
			run.Images = append(run.Images, metric)
			run.AvgMSE += metric.MSE
			run.AvgPSNR += metric.PSNR
			run.AvgBPP += metric.BPP
			run.AvgBytes += float64(metric.Bytes)
		}
		scale := 1 / float64(len(run.Images))
		run.AvgMSE *= scale
		run.AvgPSNR *= scale
		run.AvgBPP *= scale
		run.AvgBytes *= scale
		summary.Models = append(summary.Models, run)
		if !*jsonOut {
			fmt.Printf("model=%s images=%d avg_mse=%.8f avg_psnr=%.4f avg_bpp=%.4f avg_bytes=%.1f\n",
				run.Label,
				len(run.Images),
				run.AvgMSE,
				run.AvgPSNR,
				run.AvgBPP,
				run.AvgBytes,
			)
		}
	}
	if *outDir != "" {
		data, err := json.MarshalIndent(summary, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(*outDir, "eval_summary.json"), append(data, '\n'), 0o644); err != nil {
			return err
		}
	}
	if *jsonOut {
		data, err := json.MarshalIndent(summary, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(data))
	}
	return nil
}

func evalMantaKodakImage(ctx context.Context, codec *mirage.MantaCodec, path, outDir, label string) (evalMantaKodakImageSummary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return evalMantaKodakImageSummary{}, err
	}
	source, format, err := mirage.DecodeImage(bytes.NewReader(data))
	if err != nil {
		return evalMantaKodakImageSummary{}, fmt.Errorf("%s: %w", path, err)
	}
	file, err := codec.Encode(ctx, source)
	if err != nil {
		return evalMantaKodakImageSummary{}, fmt.Errorf("%s encode: %w", path, err)
	}
	encoded, err := file.MarshalBinary()
	if err != nil {
		return evalMantaKodakImageSummary{}, err
	}
	mrgPath := ""
	if outDir != "" {
		modelDir := filepath.Join(outDir, label)
		if err := os.MkdirAll(modelDir, 0o755); err != nil {
			return evalMantaKodakImageSummary{}, err
		}
		mrgPath = filepath.Join(modelDir, strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))+".mrg")
		if err := os.WriteFile(mrgPath, encoded, 0o644); err != nil {
			return evalMantaKodakImageSummary{}, err
		}
	}
	decoded, err := codec.Decode(ctx, file)
	if err != nil {
		return evalMantaKodakImageSummary{}, fmt.Errorf("%s decode: %w", path, err)
	}
	if source.Width != decoded.Width || source.Height != decoded.Height {
		source, err = mirage.CenterCropRGB(source, decoded.Width, decoded.Height)
		if err != nil {
			return evalMantaKodakImageSummary{}, err
		}
	}
	mse, err := mirage.MSE(source, decoded)
	if err != nil {
		return evalMantaKodakImageSummary{}, err
	}
	psnr, err := mirage.PSNR(source, decoded)
	if err != nil {
		return evalMantaKodakImageSummary{}, err
	}
	return evalMantaKodakImageSummary{
		Path:   path,
		Format: format,
		MRG:    mrgPath,
		Width:  decoded.Width,
		Height: decoded.Height,
		MSE:    mse,
		PSNR:   psnr,
		BPP:    mirage.BitsPerPixel(file),
		Bytes:  file.Header.TotalBytes(),
	}, nil
}

func collectMantaEvalModels(runDir, modulePath, weightPath string) ([]evalMantaModel, error) {
	if runDir != "" {
		if modulePath != "" || weightPath != "" {
			return nil, fmt.Errorf("use either -run-dir or -manta-module/-manta-weights, not both")
		}
		matches, err := filepath.Glob(filepath.Join(runDir, "mirage_v1_lambda_*.mll"))
		if err != nil {
			return nil, err
		}
		sortStrings(matches)
		models := make([]evalMantaModel, 0, len(matches))
		for _, mod := range matches {
			if strings.HasSuffix(mod, ".weights.mll") {
				continue
			}
			weights := strings.TrimSuffix(mod, ".mll") + ".weights.mll"
			if _, err := os.Stat(weights); err != nil {
				return nil, fmt.Errorf("missing weights for %s: %w", mod, err)
			}
			model, err := evalMantaModelFromPaths(mod, weights)
			if err != nil {
				return nil, err
			}
			models = append(models, model)
		}
		if len(models) == 0 {
			return nil, fmt.Errorf("no mirage_v1_lambda_*.mll modules found in %s", runDir)
		}
		return models, nil
	}
	if modulePath == "" || weightPath == "" {
		return nil, fmt.Errorf("eval-manta-kodak requires -run-dir or both -manta-module and -manta-weights")
	}
	model, err := evalMantaModelFromPaths(modulePath, weightPath)
	if err != nil {
		return nil, err
	}
	return []evalMantaModel{model}, nil
}

func evalMantaModelFromPaths(modulePath, weightPath string) (evalMantaModel, error) {
	mod, err := mantaartifact.ReadFile(modulePath)
	if err != nil {
		return evalMantaModel{}, err
	}
	label := strings.TrimSuffix(filepath.Base(modulePath), ".mll")
	return evalMantaModel{
		Label:      label,
		Lambda:     metadataFloat64(mod.Metadata["lambda"]),
		ModulePath: modulePath,
		WeightPath: weightPath,
	}, nil
}

func metadataFloat64(value any) *float64 {
	var parsed float64
	switch v := value.(type) {
	case float64:
		parsed = v
	case float32:
		parsed = float64(v)
	case int:
		parsed = float64(v)
	case int64:
		parsed = float64(v)
	case json.Number:
		f, err := v.Float64()
		if err != nil {
			return nil
		}
		parsed = f
	case string:
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil
		}
		parsed = f
	default:
		return nil
	}
	return &parsed
}

type evalMantaModel struct {
	Label      string
	Lambda     *float64
	ModulePath string
	WeightPath string
}

type evalMantaKodakSummary struct {
	Images []string                     `json:"images"`
	Models []evalMantaKodakModelSummary `json:"models"`
}

type evalMantaKodakModelSummary struct {
	Label      string                       `json:"label"`
	Lambda     *float64                     `json:"lambda,omitempty"`
	ModulePath string                       `json:"module_path"`
	WeightPath string                       `json:"weight_path"`
	AvgMSE     float64                      `json:"avg_mse"`
	AvgPSNR    float64                      `json:"avg_psnr"`
	AvgBPP     float64                      `json:"avg_bpp"`
	AvgBytes   float64                      `json:"avg_bytes"`
	Images     []evalMantaKodakImageSummary `json:"images"`
}

type evalMantaKodakImageSummary struct {
	Path   string  `json:"path"`
	Format string  `json:"format"`
	MRG    string  `json:"mrg,omitempty"`
	Width  int     `json:"width"`
	Height int     `json:"height"`
	MSE    float64 `json:"mse"`
	PSNR   float64 `json:"psnr"`
	BPP    float64 `json:"bpp"`
	Bytes  uint64  `json:"bytes"`
}
