package mirage

// MantaConfig controls the Mirage-owned Manta artifact export.
type MantaConfig struct {
	Name           string
	ImageChannels  int
	ImageHeight    int
	ImageWidth     int
	LatentChannels int
	HyperChannels  int
	BitWidth       int
	Seed           int64
	Factorization  Factorization
	Lambda         float64
}

// DefaultMantaConfig returns the default Manta Mirage Image v1 shape.
func DefaultMantaConfig() MantaConfig {
	return MantaConfig{}
}

func mantaFactorizationString(f Factorization) string {
	switch f {
	case FactorizationBitPlane:
		return "bit-plane"
	default:
		return "categorical"
	}
}
