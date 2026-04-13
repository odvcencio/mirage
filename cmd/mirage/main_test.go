package main

import (
	"path/filepath"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
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
	if len(mod.EntryPoints) != 2 || len(mod.Steps) == 0 {
		t.Fatalf("unexpected artifact shape: entrypoints=%d steps=%d", len(mod.EntryPoints), len(mod.Steps))
	}
	if err := runCheckManta([]string{"-in", path, "-entry", "train_step"}); err != nil {
		t.Fatal(err)
	}
}
