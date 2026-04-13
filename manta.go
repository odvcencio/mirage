//go:build !js

package mirage

import (
	"fmt"
	"os"
	"path/filepath"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	"github.com/odvcencio/manta/models"
)

// MantaModule returns the Manta Mirage Image v1 module. The artifact contains
// the first-class conv/GDN/TurboQuant/rate-distortion op surface used by the
// learned codec path.
func MantaModule(cfg MantaConfig) (*mantaartifact.Module, error) {
	return models.DefaultMirageV1Module(mantaModelsConfig(cfg))
}

// EncodeMantaMLL returns the serialized MLL container for the Manta Mirage v1
// module.
func EncodeMantaMLL(cfg MantaConfig) ([]byte, error) {
	mod, err := MantaModule(cfg)
	if err != nil {
		return nil, err
	}
	return mantaartifact.EncodeMLL(mod)
}

// WriteMantaMLL writes the Mirage v1 Manta module to path.
func WriteMantaMLL(path string, cfg MantaConfig) error {
	if path == "" {
		return fmt.Errorf("mirage: output path is required")
	}
	mod, err := MantaModule(cfg)
	if err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return mantaartifact.WriteFile(path, mod)
}

// MantaModelFingerprint returns the model fingerprint for the exported Manta
// artifact bytes. Manta-backed .mrg encoders store this in the file header.
func MantaModelFingerprint(cfg MantaConfig) ([32]byte, error) {
	data, err := EncodeMantaMLL(cfg)
	if err != nil {
		return [32]byte{}, err
	}
	return FingerprintModel(data), nil
}

func mantaModelsConfig(cfg MantaConfig) models.MirageV1Config {
	return models.MirageV1Config{
		Name:           cfg.Name,
		ImageChannels:  cfg.ImageChannels,
		ImageHeight:    cfg.ImageHeight,
		ImageWidth:     cfg.ImageWidth,
		LatentChannels: cfg.LatentChannels,
		HyperChannels:  cfg.HyperChannels,
		BitWidth:       cfg.BitWidth,
		Seed:           cfg.Seed,
		Factorization:  mantaFactorizationString(cfg.Factorization),
		Lambda:         cfg.Lambda,
	}
}
