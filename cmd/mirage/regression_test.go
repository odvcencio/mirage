package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestMirageShortRecipeCPURegression(t *testing.T) {
	if os.Getenv("MIRAGE_RUN_CPU_RECIPE_REGRESSION") != "1" {
		t.Skip("set MIRAGE_RUN_CPU_RECIPE_REGRESSION=1 to run the long CPU training regression")
	}
	kodakDir := os.Getenv("MIRAGE_KODAK_DIR")
	if kodakDir == "" {
		t.Fatal("MIRAGE_KODAK_DIR must point at the Kodak image directory")
	}
	if info, err := os.Stat(kodakDir); err != nil || !info.IsDir() {
		t.Fatalf("MIRAGE_KODAK_DIR=%q is not a readable directory: %v", kodakDir, err)
	}

	root := t.TempDir()
	runDir := filepath.Join(root, "run")
	evalDir := filepath.Join(root, "eval")
	if err := runTrainMantaKodak([]string{
		"-dir", kodakDir,
		"-max-images", "10",
		"-steps", "2000",
		"-crop", "256",
		"-lambdas", "0.01",
		"-bits", "4",
		"-latent-channels", "16",
		"-hyper-channels", "8",
		"-optimizer", "adam",
		"-lr", "0.001",
		"-lr-schedule", "cosine",
		"-lr-final", "0.000001",
		"-clip", "1",
		"-out-dir", runDir,
	}); err != nil {
		t.Fatal(err)
	}
	if err := runEvalMantaKodak([]string{
		"-dir", kodakDir,
		"-max-images", "10",
		"-run-dir", runDir,
		"-out-dir", evalDir,
	}); err != nil {
		t.Fatal(err)
	}

	var trainSummary trainMantaKodakSummary
	readJSONFile(t, filepath.Join(runDir, "summary.json"), &trainSummary)
	if len(trainSummary.Runs) != 1 {
		t.Fatalf("training runs = %d want 1", len(trainSummary.Runs))
	}
	trainRun := trainSummary.Runs[0]
	if trainRun.Steps != 2000 || trainRun.LatentChannels != 16 || trainRun.HyperChannels != 8 || trainRun.BitWidth != 4 {
		t.Fatalf("unexpected training recipe: %+v", trainRun)
	}

	var evalSummary evalMantaKodakSummary
	readJSONFile(t, filepath.Join(evalDir, "eval_summary.json"), &evalSummary)
	if len(evalSummary.Models) != 1 {
		t.Fatalf("eval models = %d want 1", len(evalSummary.Models))
	}
	model := evalSummary.Models[0]
	assertClose(t, "avg_psnr", model.AvgPSNR, 22.2543, 0.5)
	assertClose(t, "avg_bpp", model.AvgBPP, 0.3355, 0.01)
}

func readJSONFile(t *testing.T, path string, out any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, out); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}

func assertClose(t *testing.T, name string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %.6f want %.6f +/- %.6f", name, got, want, tolerance)
	}
}
