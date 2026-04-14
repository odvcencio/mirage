package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/mirage"
)

func TestRunInitMantaWritesRunnableArtifact(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "mirage_v1.mll")
	if err := runInitManta([]string{
		"-out", path,
		"-bits", "2",
		"-height", "16",
		"-width", "16",
		"-latent-channels", "8",
	}); err != nil {
		t.Fatal(err)
	}
	mod, err := mantaartifact.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if mod.Name != "mirage_image_v1" {
		t.Fatalf("module name = %q", mod.Name)
	}
	if len(mod.EntryPoints) != 4 || len(mod.Steps) == 0 {
		t.Fatalf("unexpected artifact shape: entrypoints=%d steps=%d", len(mod.EntryPoints), len(mod.Steps))
	}
	if err := runCheckManta([]string{"-in", path, "-entry", "train_step"}); err != nil {
		t.Fatal(err)
	}
}

func TestRunTrainMantaSmokeUsesDecodedImageFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "source.png")
	img, err := mirage.NewRGBImage(16, 16)
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			img.Pix[y*img.Width+x] = float32(x) / float32(img.Width-1)
			img.Pix[img.Width*img.Height+y*img.Width+x] = float32(y) / float32(img.Height-1)
			img.Pix[2*img.Width*img.Height+y*img.Width+x] = 0.25 + 0.5*float32((x+y)%7)/6
		}
	}
	var encoded bytes.Buffer
	if err := mirage.EncodePNG(&encoded, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, encoded.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runTrainMantaSmoke([]string{"-in", path, "-steps", "4", "-bits", "2", "-latent-channels", "4", "-hyper-channels", "4"}); err != nil {
		t.Fatal(err)
	}
}

func TestRunTrainMantaKodakUsesDirectoryAndLambdaSweep(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"kodim02.png", "kodim01.png"} {
		path := filepath.Join(dir, name)
		var encoded bytes.Buffer
		if err := mirage.EncodePNG(&encoded, testTrainingImage(t, 48, 48)); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, encoded.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	outDir := filepath.Join(dir, "runs")
	if err := runTrainMantaKodak([]string{
		"-dir", dir,
		"-max-images", "1",
		"-steps", "1",
		"-crop", "32",
		"-crop-mode", "random",
		"-random-crops-per-image", "2",
		"-crop-seed", "99",
		"-lambdas", "0.001,0.01",
		"-bits", "2",
		"-latent-channels", "4",
		"-hyper-channels", "4",
		"-optimizer", "adam",
		"-lr", "0.001",
		"-lr-schedule", "cosine",
		"-lr-final", "0.0001",
		"-lambda-schedule", "linear",
		"-lambda-start", "0",
		"-lambda-ramp-steps", "2",
		"-freeze-analysis-steps", "1",
		"-checkpoint-every", "1",
		"-out-dir", outDir,
	}); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{
		filepath.Join(outDir, "summary.json"),
		filepath.Join(outDir, "mirage_v1_lambda_0p001.mll"),
		filepath.Join(outDir, "mirage_v1_lambda_0p001.weights.mll"),
		filepath.Join(outDir, "mirage_v1_lambda_0p001_step_000001.mll"),
		filepath.Join(outDir, "mirage_v1_lambda_0p001_step_000001.weights.mll"),
		filepath.Join(outDir, "mirage_v1_lambda_0p01.mll"),
		filepath.Join(outDir, "mirage_v1_lambda_0p01.weights.mll"),
		filepath.Join(outDir, "mirage_v1_lambda_0p01_step_000001.mll"),
		filepath.Join(outDir, "mirage_v1_lambda_0p01_step_000001.weights.mll"),
	} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("expected output %s: %v", path, err)
		}
	}
	mod, err := mantaartifact.ReadFile(filepath.Join(outDir, "mirage_v1_lambda_0p001.mll"))
	if err != nil {
		t.Fatal(err)
	}
	entry, err := mantaEntryByName(mod, "analyze")
	if err != nil {
		t.Fatal(err)
	}
	if len(entry.Inputs) != 1 || entry.Inputs[0].Type.Tensor == nil {
		t.Fatalf("unexpected analyze inputs: %+v", entry.Inputs)
	}
	if got, want := entry.Inputs[0].Type.Tensor.Shape, []string{"1", "3", "32", "32"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("analyze input shape = %v want %v", got, want)
	}
	data, err := os.ReadFile(filepath.Join(outDir, "summary.json"))
	if err != nil {
		t.Fatal(err)
	}
	var summary trainMantaKodakSummary
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatal(err)
	}
	if len(summary.Runs) != 2 {
		t.Fatalf("summary runs = %d want 2", len(summary.Runs))
	}
	if got := summary.Runs[0].CropMode; got != "random" {
		t.Fatalf("crop mode = %q want random", got)
	}
	if got := summary.Runs[0].TrainingCrops; got != 2 {
		t.Fatalf("training crops = %d want 2", got)
	}
	if got := summary.Runs[0].CropSeed; got != 99 {
		t.Fatalf("crop seed = %d want 99", got)
	}
	if got := summary.Runs[0].LRSchedule; got != "cosine" {
		t.Fatalf("lr schedule = %q want cosine", got)
	}
	if got := summary.Runs[0].LambdaSchedule; got != "linear" {
		t.Fatalf("lambda schedule = %q want linear", got)
	}
	if got := summary.Runs[0].LambdaRamp; got != 2 {
		t.Fatalf("lambda ramp = %d want 2", got)
	}
	if got := summary.Runs[0].FreezeAnalysis; got != 1 {
		t.Fatalf("freeze analysis steps = %d want 1", got)
	}
	if len(summary.Runs[0].Checkpoints) != 1 || summary.Runs[0].Checkpoints[0].Step != 1 {
		t.Fatalf("unexpected checkpoints: %+v", summary.Runs[0].Checkpoints)
	}
}

func TestParseFactorization(t *testing.T) {
	tests := map[string]struct {
		want    string
		wantErr bool
	}{
		"categorical": {want: "categorical"},
		"cat":         {want: "categorical"},
		"bit-plane":   {want: "bit-plane"},
		"bitplane":    {want: "bit-plane"},
		"bad":         {wantErr: true},
	}
	for input, tc := range tests {
		got, err := parseFactorization(input)
		if tc.wantErr {
			if err == nil {
				t.Fatalf("parseFactorization(%q) succeeded", input)
			}
			continue
		}
		if err != nil {
			t.Fatalf("parseFactorization(%q): %v", input, err)
		}
		if got.String() != tc.want {
			t.Fatalf("parseFactorization(%q) = %s want %s", input, got, tc.want)
		}
	}
}

func TestParseFloat64List(t *testing.T) {
	got, err := parseFloat64List("0.001, 0.01,0.1")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0] != 0.001 || got[1] != 0.01 || got[2] != 0.1 {
		t.Fatalf("parseFloat64List returned %v", got)
	}
	if _, err := parseFloat64List(""); err == nil {
		t.Fatal("expected empty list error")
	}
}

func testTrainingImage(t *testing.T, width, height int) mirage.RGBImage {
	t.Helper()
	img, err := mirage.NewRGBImage(width, height)
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			img.Pix[y*img.Width+x] = float32(x) / float32(img.Width-1)
			img.Pix[img.Width*img.Height+y*img.Width+x] = float32(y) / float32(img.Height-1)
			img.Pix[2*img.Width*img.Height+y*img.Width+x] = 0.25 + 0.5*float32((x+y)%7)/6
		}
	}
	return img
}
