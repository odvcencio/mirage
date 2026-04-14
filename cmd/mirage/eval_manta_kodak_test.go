package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	mantamodels "github.com/odvcencio/manta/models"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/mirage"
)

func TestRunEvalMantaKodakSingleCheckpoint(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "kodim01.png")
	img := testTrainingImage(t, 16, 16)
	var encoded bytes.Buffer
	if err := mirage.EncodePNG(&encoded, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(sourcePath, encoded.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg := mirage.MantaConfig{
		ImageWidth:     16,
		ImageHeight:    16,
		LatentChannels: 4,
		HyperChannels:  4,
		BitWidth:       2,
		Lambda:         0,
		LambdaSet:      true,
	}
	mod, err := mirage.MantaModule(cfg)
	if err != nil {
		t.Fatal(err)
	}
	modulePath := filepath.Join(dir, "mirage_v1_lambda_0.mll")
	weightPath := filepath.Join(dir, "mirage_v1_lambda_0.weights.mll")
	if err := mirage.WriteMantaMLL(modulePath, cfg); err != nil {
		t.Fatal(err)
	}
	weights, err := mantamodels.InitMirageV1ReferenceWeights(mod, 7)
	if err != nil {
		t.Fatal(err)
	}
	if err := mantaruntime.NewWeightFile(weights).WriteFile(weightPath); err != nil {
		t.Fatal(err)
	}

	outDir := filepath.Join(dir, "eval")
	if err := runEvalMantaKodak([]string{
		"-in", sourcePath,
		"-manta-module", modulePath,
		"-manta-weights", weightPath,
		"-out-dir", outDir,
	}); err != nil {
		t.Fatal(err)
	}
	summaryPath := filepath.Join(outDir, "eval_summary.json")
	data, err := os.ReadFile(summaryPath)
	if err != nil {
		t.Fatal(err)
	}
	var summary evalMantaKodakSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatal(err)
	}
	if len(summary.Models) != 1 || len(summary.Models[0].Images) != 1 {
		t.Fatalf("unexpected summary shape: %+v", summary)
	}
	if summary.Models[0].AvgBPP <= 0 || summary.Models[0].AvgPSNR <= 0 {
		t.Fatalf("unexpected metrics: %+v", summary.Models[0])
	}
	if summary.Models[0].Images[0].MRG == "" {
		t.Fatalf("expected mrg artifact path")
	}
	if _, err := os.Stat(summary.Models[0].Images[0].MRG); err != nil {
		t.Fatalf("expected mrg artifact: %v", err)
	}
}

func TestRunFetchCompressAIBaselineManifestOnly(t *testing.T) {
	outDir := filepath.Join(t.TempDir(), "compressai")
	if err := runFetchCompressAIBaseline([]string{
		"-out-dir", outDir,
		"-qualities", "1,3",
		"-manifest-only",
	}); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(filepath.Join(outDir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var manifest compressAIBaselineManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Metric != "mse" {
		t.Fatalf("unexpected manifest metadata: %+v", manifest)
	}
	if len(manifest.Architectures) != 2 ||
		manifest.Architectures[0] != "bmshj2018-factorized" ||
		manifest.Architectures[1] != "bmshj2018-hyperprior" {
		t.Fatalf("unexpected architectures: %+v", manifest.Architectures)
	}
	if len(manifest.Checkpoints) != 4 ||
		manifest.Checkpoints[0].Architecture != "bmshj2018-factorized" ||
		manifest.Checkpoints[0].Quality != 1 ||
		manifest.Checkpoints[1].Architecture != "bmshj2018-factorized" ||
		manifest.Checkpoints[1].Quality != 3 ||
		manifest.Checkpoints[2].Architecture != "bmshj2018-hyperprior" ||
		manifest.Checkpoints[2].Quality != 1 ||
		manifest.Checkpoints[3].Architecture != "bmshj2018-hyperprior" ||
		manifest.Checkpoints[3].Quality != 3 {
		t.Fatalf("unexpected checkpoints: %+v", manifest.Checkpoints)
	}
	if manifest.Checkpoints[0].Downloaded {
		t.Fatalf("manifest-only should not mark checkpoints downloaded")
	}
}
