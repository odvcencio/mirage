package mirage

import (
	"encoding/hex"
	"path/filepath"
	"testing"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
)

func TestMantaModuleExport(t *testing.T) {
	mod, err := MantaModule(DefaultMantaConfig())
	if err != nil {
		t.Fatal(err)
	}
	if mod.Name == "" {
		t.Fatal("module name is empty")
	}
	if len(mod.EntryPoints) == 0 {
		t.Fatal("expected entrypoints")
	}
	data, err := EncodeMantaMLL(DefaultMantaConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(data) == 0 {
		t.Fatal("expected MLL bytes")
	}
	fp, err := MantaModelFingerprint(DefaultMantaConfig())
	if err != nil {
		t.Fatal(err)
	}
	if hex.EncodeToString(fp[:]) == "" {
		t.Fatal("expected fingerprint")
	}
}

func TestWriteMantaMLLCreatesParentDirectory(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "mirage_v1.mll")
	if err := WriteMantaMLL(path, MantaConfig{BitWidth: 2, ImageHeight: 16, ImageWidth: 16, LatentChannels: 8}); err != nil {
		t.Fatal(err)
	}
	mod, err := mantaartifact.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if mod.Name == "" || len(mod.Steps) == 0 {
		t.Fatalf("unexpected module: %+v", mod)
	}
}
