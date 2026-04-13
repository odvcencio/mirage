package mirage

import (
	"bytes"
	"fmt"

	turboquant "github.com/odvcencio/turboquant"
)

const (
	// PatchSize is the executable Go v1 analysis/synthesis tile size. The Manta
	// Balle pipeline will replace this while preserving the bitstream substrate.
	PatchSize = 4
	// PatchLatentChannels is RGB channels times one PatchSize x PatchSize tile.
	PatchLatentChannels = 3 * PatchSize * PatchSize
	// DefaultSeed is the deterministic TurboQuant rotation seed for the default
	// Go v1 model artifact.
	DefaultSeed int64 = 0x4d697261676531
)

// EncodeOptions configures the executable Go v1 codec path.
type EncodeOptions struct {
	BitWidth         int
	Seed             int64
	DistortionMetric DistortionMetric
	Factorization    Factorization
}

// DecodeOptions configures the executable Go v1 decoder path.
type DecodeOptions struct {
	Seed               int64
	AllowModelMismatch bool
}

// DefaultEncodeOptions returns the default 4-bit categorical v1 codec config.
func DefaultEncodeOptions() EncodeOptions {
	return EncodeOptions{
		BitWidth:         4,
		Seed:             DefaultSeed,
		DistortionMetric: DistortionMSE,
		Factorization:    FactorizationCategorical,
	}
}

// DefaultDecodeOptions returns options for decoding the default v1 model.
func DefaultDecodeOptions() DecodeOptions {
	return DecodeOptions{Seed: DefaultSeed}
}

// ModelFingerprintForSeed returns the fingerprint used by the executable Go v1
// model path. A full Manta .mll artifact will supply this in the Manta-backed
// implementation.
func ModelFingerprintForSeed(seed int64) [32]byte {
	artifact := fmt.Appendf(nil, "mirage-image-v1-go-patch;patch=%d;channels=%d;seed=%d", PatchSize, PatchLatentChannels, seed)
	return FingerprintModel(artifact)
}

// DefaultModelFingerprint returns the fingerprint for the default Go v1 model.
func DefaultModelFingerprint() [32]byte {
	return ModelFingerprintForSeed(DefaultSeed)
}

// Encode converts an RGB image into a Mirage v1 .mrg file. This host-side path
// is intentionally simple: 4x4 RGB patches are the analysis latent and
// TurboQuant supplies the codec's structural quantizer.
func Encode(img RGBImage, opts EncodeOptions) (File, error) {
	if err := img.validate(); err != nil {
		return File{}, err
	}
	if opts.BitWidth == 0 {
		opts.BitWidth = 4
	}
	if opts.Seed == 0 {
		opts.Seed = DefaultSeed
	}
	if opts.DistortionMetric != DistortionMSE && opts.DistortionMetric != DistortionMSSSIM {
		return File{}, fmt.Errorf("mirage: unsupported distortion metric %d", opts.DistortionMetric)
	}
	if opts.Factorization != FactorizationCategorical && opts.Factorization != FactorizationBitPlane {
		return File{}, fmt.Errorf("mirage: unsupported factorization %d", opts.Factorization)
	}

	latents, shape, err := AnalyzePatches(img)
	if err != nil {
		return File{}, err
	}
	code, err := EncodeLatents(latents, shape, opts.BitWidth, opts.Seed)
	if err != nil {
		return File{}, err
	}
	coordPayload, err := EncodeBytes(code.Coordinates)
	if err != nil {
		return File{}, err
	}
	normPayload, err := EncodeBytes(code.Norms)
	if err != nil {
		return File{}, err
	}
	return BuildFile(HeaderOptions{
		DistortionMetric: opts.DistortionMetric,
		Factorization:    opts.Factorization,
		BitWidth:         opts.BitWidth,
		ImageWidth:       uint32(img.Width),
		ImageHeight:      uint32(img.Height),
		LatentChannels:   uint32(shape.Channels),
		LatentHeight:     uint32(shape.Height),
		LatentWidth:      uint32(shape.Width),
		ModelFingerprint: ModelFingerprintForSeed(opts.Seed),
	}, Payloads{
		CCoords: coordPayload,
		CNorms:  normPayload,
	})
}

// Decode reconstructs an RGB image from a Mirage v1 .mrg file produced by
// Encode. Manta-backed decoders should use the same File/Header substrate and
// replace the patch synthesis path with synthesize_hyperprior+synthesize_image.
func Decode(file File, opts DecodeOptions) (RGBImage, error) {
	if err := file.Header.Validate(); err != nil {
		return RGBImage{}, err
	}
	if opts.Seed == 0 {
		opts.Seed = DefaultSeed
	}
	wantFingerprint := ModelFingerprintForSeed(opts.Seed)
	if !opts.AllowModelMismatch && file.Header.ModelFingerprint != wantFingerprint {
		return RGBImage{}, fmt.Errorf("mirage: model fingerprint mismatch")
	}
	shape := file.Header.LatentShape()
	if shape.Channels != PatchLatentChannels {
		return RGBImage{}, fmt.Errorf("mirage: unsupported Go v1 latent channel count %d", shape.Channels)
	}
	if shape.Height != ceilDiv(int(file.Header.ImageHeight), PatchSize) || shape.Width != ceilDiv(int(file.Header.ImageWidth), PatchSize) {
		return RGBImage{}, fmt.Errorf("mirage: latent shape does not match patch model")
	}
	coordLen := shape.Positions() * turboquant.PackedSize(shape.Channels, int(file.Header.BitWidth))
	normLen := shape.Positions()
	coords, err := DecodeBytes(file.Payloads.CCoords, coordLen)
	if err != nil {
		return RGBImage{}, err
	}
	norms, err := DecodeBytes(file.Payloads.CNorms, normLen)
	if err != nil {
		return RGBImage{}, err
	}
	code := LatentCode{
		Shape:       shape,
		BitWidth:    int(file.Header.BitWidth),
		Coordinates: coords,
		Norms:       norms,
	}
	latents, err := DecodeLatents(code, opts.Seed)
	if err != nil {
		return RGBImage{}, err
	}
	return SynthesizePatches(latents, shape, int(file.Header.ImageWidth), int(file.Header.ImageHeight))
}

// EncodeBytesMRG serializes an RGB image directly to .mrg bytes.
func EncodeBytesMRG(img RGBImage, opts EncodeOptions) ([]byte, error) {
	file, err := Encode(img, opts)
	if err != nil {
		return nil, err
	}
	return file.MarshalBinary()
}

// DecodeBytesMRG parses and decodes a complete .mrg byte slice.
func DecodeBytesMRG(data []byte, opts DecodeOptions) (RGBImage, error) {
	file, err := ParseFile(data)
	if err != nil {
		return RGBImage{}, err
	}
	return Decode(file, opts)
}

// EncodeImageReader reads PNG/JPEG/PPM image data and emits a complete .mrg
// file. The returned format is the detected source image format.
func EncodeImageReader(data []byte, opts EncodeOptions) ([]byte, string, error) {
	img, format, err := DecodeImage(bytes.NewReader(data))
	if err != nil {
		return nil, "", err
	}
	encoded, err := EncodeBytesMRG(img, opts)
	return encoded, format, err
}

// BitsPerPixel returns the .mrg container rate including header and CRC.
func BitsPerPixel(file File) float64 {
	pixels := file.Header.ImagePixels()
	if pixels == 0 {
		return 0
	}
	return float64(file.Header.TotalBytes()*8) / float64(pixels)
}

func ceilDiv(n, d int) int {
	return (n + d - 1) / d
}
