//go:build !js

package mirage

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
)

const mantaCodecProbabilityFloor = 1e-9

// MantaCodecOptions identifies a trained Manta Mirage v1 module plus its
// matching runtime weight file.
type MantaCodecOptions struct {
	ModulePath         string
	WeightPath         string
	AllowModelMismatch bool
	CDFTotal           uint32
}

// MantaCodec runs the learned Mirage v1 Manta deployment graph and wraps its
// tensor outputs in the .mrg container and arithmetic coder.
type MantaCodec struct {
	mod                *mantaartifact.Module
	prog               *mantaruntime.Program
	weights            map[string]*backend.Tensor
	fingerprint        [32]byte
	spec               mantaCodecSpec
	allowModelMismatch bool
	cdfTotal           uint32
}

type mantaCodecSpec struct {
	ImageChannels  int
	ImageHeight    int
	ImageWidth     int
	LatentChannels int
	LatentHeight   int
	LatentWidth    int
	HyperChannels  int
	HyperHeight    int
	HyperWidth     int
	BitWidth       int
	Factorization  Factorization
}

// LoadMantaCodec loads a trained Mirage v1 Manta module and weight file for
// host-side encode/decode round trips.
func LoadMantaCodec(ctx context.Context, opts MantaCodecOptions) (*MantaCodec, error) {
	if opts.ModulePath == "" {
		return nil, fmt.Errorf("mirage: Manta module path is required")
	}
	if opts.WeightPath == "" {
		return nil, fmt.Errorf("mirage: Manta weight path is required")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	moduleBytes, err := os.ReadFile(opts.ModulePath)
	if err != nil {
		return nil, err
	}
	weightBytes, err := os.ReadFile(opts.WeightPath)
	if err != nil {
		return nil, err
	}
	mod, err := mantaartifact.ReadFile(opts.ModulePath)
	if err != nil {
		return nil, err
	}
	spec, err := inferMantaCodecSpec(mod)
	if err != nil {
		return nil, err
	}
	weights, err := mantaruntime.ReadWeightFile(opts.WeightPath)
	if err != nil {
		return nil, err
	}
	prog, err := mantaruntime.New(webgpu.New()).Load(ctx, mod, weights.LoadOptions()...)
	if err != nil {
		return nil, err
	}
	return &MantaCodec{
		mod:                mod,
		prog:               prog,
		weights:            weights.Weights,
		fingerprint:        FingerprintModel(mantaCodecFingerprintBytes(moduleBytes, weightBytes)),
		spec:               spec,
		allowModelMismatch: opts.AllowModelMismatch,
		cdfTotal:           opts.CDFTotal,
	}, nil
}

func mantaCodecFingerprintBytes(moduleBytes, weightBytes []byte) []byte {
	out := make([]byte, 0, len("mirage-manta-codec-v1")+2+len(moduleBytes)+len(weightBytes))
	out = append(out, "mirage-manta-codec-v1"...)
	out = append(out, 0)
	out = append(out, moduleBytes...)
	out = append(out, 0)
	out = append(out, weightBytes...)
	return out
}

// Fingerprint returns the module+weight fingerprint stored in .mrg headers.
func (c *MantaCodec) Fingerprint() [32]byte {
	if c == nil {
		return [32]byte{}
	}
	return c.fingerprint
}

// Encode converts an RGB image into a learned Manta-backed Mirage v1 .mrg file.
// Inputs larger than the module shape are center-cropped to the fixed module
// dimensions.
func (c *MantaCodec) Encode(ctx context.Context, img RGBImage) (File, error) {
	if c == nil || c.prog == nil {
		return File{}, fmt.Errorf("mirage: Manta codec is not loaded")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	x, err := rgbImageToMantaTrainTensor(img, c.spec.ImageWidth, c.spec.ImageHeight)
	if err != nil {
		return File{}, err
	}
	result, err := c.prog.Run(ctx, backend.Request{
		Entry:  "analyze",
		Inputs: map[string]any{"x": x},
	})
	if err != nil {
		return File{}, err
	}
	cz, err := mantaResultTensor(result, "c_z")
	if err != nil {
		return File{}, err
	}
	czNorms, err := mantaResultTensor(result, "c_z_norms")
	if err != nil {
		return File{}, err
	}
	coords, err := mantaResultTensor(result, "c_coords")
	if err != nil {
		return File{}, err
	}
	norms, err := mantaResultTensor(result, "c_norms")
	if err != nil {
		return File{}, err
	}
	piLogits, err := mantaResultTensor(result, "pi_logits")
	if err != nil {
		return File{}, err
	}
	normParams, err := mantaResultTensor(result, "norm_params")
	if err != nil {
		return File{}, err
	}

	zCode, err := latentCodeFromMantaTensors(cz, czNorms, c.spec.BitWidth)
	if err != nil {
		return File{}, fmt.Errorf("c_z: %w", err)
	}
	mainCode, err := latentCodeFromMantaTensors(coords, norms, c.spec.BitWidth)
	if err != nil {
		return File{}, fmt.Errorf("c_coords: %w", err)
	}
	zModels, err := c.hyperpriorModels()
	if err != nil {
		return File{}, err
	}
	zPayloads, err := EncodeLatentPayloadsWithModels(zCode, FactorizationCategorical, zModels)
	if err != nil {
		return File{}, fmt.Errorf("encode c_z: %w", err)
	}
	mainModels, err := mantaMainPayloadModels(mainCode.Shape, c.spec.BitWidth, c.spec.Factorization, piLogits, normParams, c.cdfTotal)
	if err != nil {
		return File{}, err
	}
	mainPayloads, err := EncodeLatentPayloadsWithModels(mainCode, c.spec.Factorization, mainModels)
	if err != nil {
		return File{}, fmt.Errorf("encode main latents: %w", err)
	}
	hyperPayload, err := packMantaHyperPayloads(zPayloads)
	if err != nil {
		return File{}, err
	}
	return BuildFile(HeaderOptions{
		DistortionMetric: DistortionMSE,
		Factorization:    c.spec.Factorization,
		BitWidth:         c.spec.BitWidth,
		ImageWidth:       uint32(c.spec.ImageWidth),
		ImageHeight:      uint32(c.spec.ImageHeight),
		LatentChannels:   uint32(c.spec.LatentChannels),
		LatentHeight:     uint32(c.spec.LatentHeight),
		LatentWidth:      uint32(c.spec.LatentWidth),
		ModelFingerprint: c.fingerprint,
	}, Payloads{
		CZ:      hyperPayload,
		CCoords: mainPayloads.CCoords,
		CNorms:  mainPayloads.CNorms,
	})
}

// Decode reconstructs an RGB image from a learned Manta-backed Mirage v1 .mrg
// file using the loaded module and weights.
func (c *MantaCodec) Decode(ctx context.Context, file File) (RGBImage, error) {
	if c == nil || c.prog == nil {
		return RGBImage{}, fmt.Errorf("mirage: Manta codec is not loaded")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := file.Header.Validate(); err != nil {
		return RGBImage{}, err
	}
	if !c.allowModelMismatch && file.Header.ModelFingerprint != c.fingerprint {
		return RGBImage{}, fmt.Errorf("mirage: model fingerprint mismatch")
	}
	if err := c.validateHeader(file.Header); err != nil {
		return RGBImage{}, err
	}
	zPayloads, err := unpackMantaHyperPayloads(file.Payloads.CZ)
	if err != nil {
		return RGBImage{}, err
	}
	zModels, err := c.hyperpriorModels()
	if err != nil {
		return RGBImage{}, err
	}
	zCode, err := DecodeLatentPayloadsWithModels(zPayloads, LatentShape{
		Channels: c.spec.HyperChannels,
		Height:   c.spec.HyperHeight,
		Width:    c.spec.HyperWidth,
	}, c.spec.BitWidth, FactorizationCategorical, zModels)
	if err != nil {
		return RGBImage{}, fmt.Errorf("decode c_z: %w", err)
	}
	cz, czNorms, err := mantaTensorsFromLatentCode(zCode)
	if err != nil {
		return RGBImage{}, err
	}
	hyper, err := c.prog.Run(ctx, backend.Request{
		Entry: "synthesize_hyperprior",
		Inputs: map[string]any{
			"c_z":       cz,
			"c_z_norms": czNorms,
		},
	})
	if err != nil {
		return RGBImage{}, err
	}
	piLogits, err := mantaResultTensor(hyper, "pi_logits")
	if err != nil {
		return RGBImage{}, err
	}
	normParams, err := mantaResultTensor(hyper, "norm_params")
	if err != nil {
		return RGBImage{}, err
	}
	mainShape := file.Header.LatentShape()
	mainModels, err := mantaMainPayloadModels(mainShape, int(file.Header.BitWidth), file.Header.Factorization(), piLogits, normParams, c.cdfTotal)
	if err != nil {
		return RGBImage{}, err
	}
	mainCode, err := DecodeLatentPayloadsWithModels(file.Payloads, mainShape, int(file.Header.BitWidth), file.Header.Factorization(), mainModels)
	if err != nil {
		return RGBImage{}, fmt.Errorf("decode main latents: %w", err)
	}
	coords, norms, err := mantaTensorsFromLatentCode(mainCode)
	if err != nil {
		return RGBImage{}, err
	}
	synth, err := c.prog.Run(ctx, backend.Request{
		Entry: "synthesize_image",
		Inputs: map[string]any{
			"c_coords": coords,
			"c_norms":  norms,
		},
	})
	if err != nil {
		return RGBImage{}, err
	}
	xHat, err := mantaResultTensor(synth, "x_hat")
	if err != nil {
		return RGBImage{}, err
	}
	return rgbImageFromMantaTensor(xHat)
}

func (c *MantaCodec) validateHeader(h Header) error {
	if int(h.ImageWidth) != c.spec.ImageWidth || int(h.ImageHeight) != c.spec.ImageHeight {
		return fmt.Errorf("mirage: Manta header image %dx%d does not match module %dx%d", h.ImageWidth, h.ImageHeight, c.spec.ImageWidth, c.spec.ImageHeight)
	}
	if int(h.BitWidth) != c.spec.BitWidth {
		return fmt.Errorf("mirage: Manta header bit width %d does not match module %d", h.BitWidth, c.spec.BitWidth)
	}
	if h.Factorization() != c.spec.Factorization {
		return fmt.Errorf("mirage: Manta header factorization %s does not match module %s", h.Factorization(), c.spec.Factorization)
	}
	shape := h.LatentShape()
	if shape.Channels != c.spec.LatentChannels || shape.Height != c.spec.LatentHeight || shape.Width != c.spec.LatentWidth {
		return fmt.Errorf("mirage: Manta header latent shape %+v does not match module %+v", shape, c.spec)
	}
	return nil
}

func (c *MantaCodec) hyperpriorModels() (LatentPayloadModels, error) {
	prior := c.weightTensor("prior_z_logits")
	if prior == nil {
		return LatentPayloadModels{}, fmt.Errorf("mirage: Manta weights missing prior_z_logits")
	}
	coord, err := cdfFromLogitsFloored(prior.F32, c.cdfTotal)
	if err != nil {
		return LatentPayloadModels{}, err
	}
	defaults, err := DefaultLatentPayloadModels(c.spec.BitWidth)
	if err != nil {
		return LatentPayloadModels{}, err
	}
	return LatentPayloadModels{Coordinate: coord, Norm: defaults.Norm}, nil
}

func (c *MantaCodec) weightTensor(name string) *backend.Tensor {
	if c == nil || c.weights == nil {
		return nil
	}
	return c.weights[name]
}

func inferMantaCodecSpec(mod *mantaartifact.Module) (mantaCodecSpec, error) {
	analyze, err := mantaEntryPoint(mod, "analyze")
	if err != nil {
		return mantaCodecSpec{}, err
	}
	x, err := mantaBinding(analyze.Inputs, "x")
	if err != nil {
		return mantaCodecSpec{}, err
	}
	coords, err := mantaBinding(analyze.Outputs, "c_coords")
	if err != nil {
		return mantaCodecSpec{}, err
	}
	cz, err := mantaBinding(analyze.Outputs, "c_z")
	if err != nil {
		return mantaCodecSpec{}, err
	}
	imageShape, err := concreteMantaShape(x.Type.Tensor.Shape)
	if err != nil {
		return mantaCodecSpec{}, fmt.Errorf("image shape: %w", err)
	}
	latentShape, err := concreteMantaShape(coords.Type.Tensor.Shape)
	if err != nil {
		return mantaCodecSpec{}, fmt.Errorf("latent shape: %w", err)
	}
	hyperShape, err := concreteMantaShape(cz.Type.Tensor.Shape)
	if err != nil {
		return mantaCodecSpec{}, fmt.Errorf("hyper shape: %w", err)
	}
	if len(imageShape) != 4 || len(latentShape) != 4 || len(hyperShape) != 4 {
		return mantaCodecSpec{}, fmt.Errorf("mirage: Manta codec expects NCHW image/latent/hyper tensors")
	}
	bitWidth, err := bitWidthFromMantaDType(coords.Type.Tensor.DType)
	if err != nil {
		return mantaCodecSpec{}, err
	}
	factorization, err := factorizationFromMantaMetadata(mod.Metadata["factorization"])
	if err != nil {
		return mantaCodecSpec{}, err
	}
	return mantaCodecSpec{
		ImageChannels:  imageShape[1],
		ImageHeight:    imageShape[2],
		ImageWidth:     imageShape[3],
		LatentChannels: latentShape[1],
		LatentHeight:   latentShape[2],
		LatentWidth:    latentShape[3],
		HyperChannels:  hyperShape[1],
		HyperHeight:    hyperShape[2],
		HyperWidth:     hyperShape[3],
		BitWidth:       bitWidth,
		Factorization:  factorization,
	}, nil
}

func mantaEntryPoint(mod *mantaartifact.Module, name string) (mantaartifact.EntryPoint, error) {
	for _, entry := range mod.EntryPoints {
		if entry.Name == name {
			return entry, nil
		}
	}
	return mantaartifact.EntryPoint{}, fmt.Errorf("mirage: Manta module missing entrypoint %q", name)
}

func mantaBinding(bindings []mantaartifact.ValueBinding, name string) (mantaartifact.ValueBinding, error) {
	for _, binding := range bindings {
		if binding.Name == name {
			if binding.Type.Tensor == nil {
				return mantaartifact.ValueBinding{}, fmt.Errorf("mirage: Manta binding %q is not a tensor", name)
			}
			return binding, nil
		}
	}
	return mantaartifact.ValueBinding{}, fmt.Errorf("mirage: Manta binding %q not found", name)
}

func concreteMantaShape(shape []string) ([]int, error) {
	out := make([]int, len(shape))
	for i, dim := range shape {
		n, err := strconv.Atoi(dim)
		if err != nil {
			return nil, fmt.Errorf("non-concrete dim %q", dim)
		}
		if n <= 0 {
			return nil, fmt.Errorf("non-positive dim %d", n)
		}
		out[i] = n
	}
	return out, nil
}

func bitWidthFromMantaDType(dtype string) (int, error) {
	if !strings.HasPrefix(dtype, "q") {
		return 0, fmt.Errorf("mirage: Manta coordinate dtype %q is not quantized", dtype)
	}
	bits, err := strconv.Atoi(strings.TrimPrefix(dtype, "q"))
	if err != nil {
		return 0, fmt.Errorf("mirage: invalid Manta coordinate dtype %q", dtype)
	}
	if err := validateMirageBitWidth(bits); err != nil {
		return 0, err
	}
	return bits, nil
}

func factorizationFromMantaMetadata(value any) (Factorization, error) {
	switch v := value.(type) {
	case string:
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "", "categorical", "cat":
			return FactorizationCategorical, nil
		case "bit-plane", "bitplane", "bit_plane", "bits":
			return FactorizationBitPlane, nil
		default:
			return 0, fmt.Errorf("mirage: unknown Manta factorization %q", v)
		}
	case nil:
		return FactorizationCategorical, nil
	default:
		return 0, fmt.Errorf("mirage: unsupported Manta factorization metadata %T", value)
	}
}

func mantaResultTensor(result backend.Result, name string) (*backend.Tensor, error) {
	value, ok := result.Outputs[name]
	if !ok {
		return nil, fmt.Errorf("mirage: Manta result missing %q", name)
	}
	tensor, ok := value.Data.(*backend.Tensor)
	if !ok || tensor == nil {
		return nil, fmt.Errorf("mirage: Manta result %q is %T, want tensor", name, value.Data)
	}
	return tensor, nil
}

func latentCodeFromMantaTensors(coords, norms *backend.Tensor, bitWidth int) (LatentCode, error) {
	if coords == nil || norms == nil {
		return LatentCode{}, fmt.Errorf("mirage: missing Manta latent tensors")
	}
	if len(coords.Shape) != 4 || coords.Shape[0] != 1 {
		return LatentCode{}, fmt.Errorf("mirage: coordinates must be NCHW with N=1, got %v", coords.Shape)
	}
	if len(norms.Shape) != 3 || norms.Shape[0] != 1 || norms.Shape[1] != coords.Shape[2] || norms.Shape[2] != coords.Shape[3] {
		return LatentCode{}, fmt.Errorf("mirage: norm shape %v does not match coordinates %v", norms.Shape, coords.Shape)
	}
	shape := LatentShape{Channels: coords.Shape[1], Height: coords.Shape[2], Width: coords.Shape[3]}
	levels := 1 << bitWidth
	symbols := make([]uint16, 0, shape.Elements())
	for y := 0; y < shape.Height; y++ {
		for x := 0; x < shape.Width; x++ {
			for c := 0; c < shape.Channels; c++ {
				raw := coords.F32[offset4(coords.Shape, 0, c, y, x)]
				symbols = append(symbols, uint16(clampInt(int(math.Round(float64(raw))), 0, levels-1)))
			}
		}
	}
	packed, err := PackCoordinateSymbols(symbols, shape, bitWidth)
	if err != nil {
		return LatentCode{}, err
	}
	normBytes := make([]byte, shape.Positions())
	for y := 0; y < shape.Height; y++ {
		for x := 0; x < shape.Width; x++ {
			raw := norms.F32[offset3(norms.Shape, 0, y, x)]
			normBytes[y*shape.Width+x] = byte(clampInt(int(math.Round(float64(raw))), 0, 255))
		}
	}
	code := LatentCode{Shape: shape, BitWidth: bitWidth, Coordinates: packed, Norms: normBytes}
	return code, code.Validate()
}

func mantaTensorsFromLatentCode(code LatentCode) (*backend.Tensor, *backend.Tensor, error) {
	if err := code.Validate(); err != nil {
		return nil, nil, err
	}
	symbols, err := UnpackCoordinateSymbols(code)
	if err != nil {
		return nil, nil, err
	}
	coords := make([]float32, code.Shape.Elements())
	for y := 0; y < code.Shape.Height; y++ {
		for x := 0; x < code.Shape.Width; x++ {
			pos := y*code.Shape.Width + x
			for c := 0; c < code.Shape.Channels; c++ {
				coords[offset3([]int{code.Shape.Channels, code.Shape.Height, code.Shape.Width}, c, y, x)] = float32(symbols[pos*code.Shape.Channels+c])
			}
		}
	}
	norms := make([]float32, code.Shape.Positions())
	for i, value := range code.Norms {
		norms[i] = float32(value)
	}
	coordShape := []int{1, code.Shape.Channels, code.Shape.Height, code.Shape.Width}
	normShape := []int{1, code.Shape.Height, code.Shape.Width}
	return qTensorForBitWidth(code.BitWidth, coordShape, coords), backend.NewTensorQNorm(normShape, norms), nil
}

func mantaMainPayloadModels(shape LatentShape, bitWidth int, factorization Factorization, logits, normParams *backend.Tensor, total uint32) (LatentPayloadModels, error) {
	defaults, err := DefaultLatentPayloadModels(bitWidth)
	if err != nil {
		return LatentPayloadModels{}, err
	}
	out := defaults
	switch factorization {
	case FactorizationCategorical:
		models, err := categoricalCDFsFromNCHWLogits(logits, shape, bitWidth, total)
		if err != nil {
			return LatentPayloadModels{}, err
		}
		out.CoordinateModels = models
	case FactorizationBitPlane:
		models, err := bitPlaneCDFsFromNCHWLogits(logits, shape, bitWidth, total)
		if err != nil {
			return LatentPayloadModels{}, err
		}
		out.CoordinateBitModels = models
	default:
		return LatentPayloadModels{}, fmt.Errorf("mirage: unsupported factorization %d", factorization)
	}
	normModels, err := normCDFsFromNCHWParams(normParams, shape, total)
	if err != nil {
		return LatentPayloadModels{}, err
	}
	out.NormModels = normModels
	return out, nil
}

func categoricalCDFsFromNCHWLogits(logits *backend.Tensor, shape LatentShape, bitWidth int, total uint32) ([]CDF, error) {
	if logits == nil || len(logits.Shape) != 4 {
		return nil, fmt.Errorf("mirage: categorical logits must be NCHW")
	}
	levels := 1 << bitWidth
	if logits.Shape[0] != 1 || logits.Shape[1] < shape.Channels*levels || logits.Shape[2] != shape.Height || logits.Shape[3] != shape.Width {
		return nil, fmt.Errorf("mirage: categorical logits shape %v does not match latent shape %+v", logits.Shape, shape)
	}
	models := make([]CDF, 0, shape.Elements())
	vec := make([]float32, levels)
	for y := 0; y < shape.Height; y++ {
		for x := 0; x < shape.Width; x++ {
			for c := 0; c < shape.Channels; c++ {
				baseChannel := c * levels
				for sym := 0; sym < levels; sym++ {
					vec[sym] = logits.F32[offset4(logits.Shape, 0, baseChannel+sym, y, x)]
				}
				model, err := cdfFromLogitsFloored(vec, total)
				if err != nil {
					return nil, fmt.Errorf("coordinate model y=%d x=%d c=%d: %w", y, x, c, err)
				}
				models = append(models, model)
			}
		}
	}
	return models, nil
}

func bitPlaneCDFsFromNCHWLogits(logits *backend.Tensor, shape LatentShape, bitWidth int, total uint32) ([]CDF, error) {
	if logits == nil || len(logits.Shape) != 4 {
		return nil, fmt.Errorf("mirage: bit-plane logits must be NCHW")
	}
	if logits.Shape[0] != 1 || logits.Shape[1] < shape.Channels*bitWidth*2 || logits.Shape[2] != shape.Height || logits.Shape[3] != shape.Width {
		return nil, fmt.Errorf("mirage: bit-plane logits shape %v does not match latent shape %+v", logits.Shape, shape)
	}
	models := make([]CDF, 0, shape.Elements()*bitWidth)
	vec := make([]float32, 2)
	for y := 0; y < shape.Height; y++ {
		for x := 0; x < shape.Width; x++ {
			for c := 0; c < shape.Channels; c++ {
				for bit := 0; bit < bitWidth; bit++ {
					ch := c*bitWidth*2 + bit*2
					vec[0] = logits.F32[offset4(logits.Shape, 0, ch, y, x)]
					vec[1] = logits.F32[offset4(logits.Shape, 0, ch+1, y, x)]
					model, err := cdfFromLogitsFloored(vec, total)
					if err != nil {
						return nil, fmt.Errorf("bit model y=%d x=%d c=%d bit=%d: %w", y, x, c, bit, err)
					}
					models = append(models, model)
				}
			}
		}
	}
	return models, nil
}

func normCDFsFromNCHWParams(params *backend.Tensor, shape LatentShape, total uint32) ([]CDF, error) {
	if params == nil || len(params.Shape) != 4 {
		return nil, fmt.Errorf("mirage: norm params must be NCHW")
	}
	if params.Shape[0] != 1 || params.Shape[1] < 2 || params.Shape[2] != shape.Height || params.Shape[3] != shape.Width {
		return nil, fmt.Errorf("mirage: norm params shape %v does not match latent shape %+v", params.Shape, shape)
	}
	models := make([]CDF, 0, shape.Positions())
	probs := make([]float32, 256)
	for y := 0; y < shape.Height; y++ {
		for x := 0; x < shape.Width; x++ {
			mu := float64(params.F32[offset4(params.Shape, 0, 0, y, x)])
			sigmaRaw := float64(params.F32[offset4(params.Shape, 0, 1, y, x)])
			sigma := softplus(sigmaRaw) + 1e-6
			if math.IsNaN(mu) || math.IsInf(mu, 0) || math.IsNaN(sigma) || math.IsInf(sigma, 0) || sigma <= 0 {
				return nil, fmt.Errorf("mirage: invalid norm params at y=%d x=%d", y, x)
			}
			for sym := range probs {
				lower, upper := DefaultNorms.normSymbolLogBounds(sym)
				mass := normalCDF((upper-mu)/sigma) - normalCDF((lower-mu)/sigma)
				if mass < 0 && mass > -1e-12 {
					mass = 0
				}
				probs[sym] = float32(math.Max(mass, mantaCodecProbabilityFloor))
			}
			model, err := CDFFromProbabilities(probs, total)
			if err != nil {
				return nil, fmt.Errorf("norm model y=%d x=%d: %w", y, x, err)
			}
			models = append(models, model)
		}
	}
	return models, nil
}

func cdfFromLogitsFloored(logits []float32, total uint32) (CDF, error) {
	if len(logits) == 0 {
		return CDF{}, fmt.Errorf("mirage: empty logits vector")
	}
	maxLogit := math.Inf(-1)
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return CDF{}, fmt.Errorf("mirage: invalid logit at %d", i)
		}
		if float64(v) > maxLogit {
			maxLogit = float64(v)
		}
	}
	probs := make([]float32, len(logits))
	for i, v := range logits {
		probs[i] = float32(math.Max(math.Exp(float64(v)-maxLogit), mantaCodecProbabilityFloor))
	}
	return CDFFromProbabilities(probs, total)
}

func packMantaHyperPayloads(payloads Payloads) ([]byte, error) {
	coordLen, err := lenUint32("c_z coordinates", len(payloads.CCoords))
	if err != nil {
		return nil, err
	}
	normLen, err := lenUint32("c_z norms", len(payloads.CNorms))
	if err != nil {
		return nil, err
	}
	out := make([]byte, 8, 8+len(payloads.CCoords)+len(payloads.CNorms))
	putUint32LE(out[0:4], coordLen)
	putUint32LE(out[4:8], normLen)
	out = append(out, payloads.CCoords...)
	out = append(out, payloads.CNorms...)
	return out, nil
}

func unpackMantaHyperPayloads(data []byte) (Payloads, error) {
	if len(data) < 8 {
		return Payloads{}, fmt.Errorf("mirage: c_z payload too short")
	}
	coordLen := int(uint32LE(data[0:4]))
	normLen := int(uint32LE(data[4:8]))
	if coordLen < 0 || normLen < 0 || len(data) != 8+coordLen+normLen {
		return Payloads{}, fmt.Errorf("mirage: c_z payload length mismatch")
	}
	return Payloads{
		CCoords: append([]byte(nil), data[8:8+coordLen]...),
		CNorms:  append([]byte(nil), data[8+coordLen:]...),
	}, nil
}

func rgbImageFromMantaTensor(t *backend.Tensor) (RGBImage, error) {
	if t == nil || len(t.Shape) != 4 || t.Shape[0] != 1 || t.Shape[1] != 3 {
		return RGBImage{}, fmt.Errorf("mirage: Manta image tensor shape %v is not [1,3,H,W]", shapeOrNil(t))
	}
	img, err := NewRGBImage(t.Shape[3], t.Shape[2])
	if err != nil {
		return RGBImage{}, err
	}
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			for c := 0; c < 3; c++ {
				img.Pix[img.offset(c, y, x)] = clampFloat32(t.F32[offset4(t.Shape, 0, c, y, x)], 0, 1)
			}
		}
	}
	return img, nil
}

func offset4(shape []int, n, c, y, x int) int {
	return ((n*shape[1]+c)*shape[2]+y)*shape[3] + x
}

func offset3(shape []int, a, b, c int) int {
	return (a*shape[1]+b)*shape[2] + c
}

func softplus(v float64) float64 {
	if v > 32 {
		return v
	}
	return math.Log1p(math.Exp(v))
}

func putUint32LE(dst []byte, v uint32) {
	dst[0] = byte(v)
	dst[1] = byte(v >> 8)
	dst[2] = byte(v >> 16)
	dst[3] = byte(v >> 24)
}

func uint32LE(src []byte) uint32 {
	return uint32(src[0]) | uint32(src[1])<<8 | uint32(src[2])<<16 | uint32(src[3])<<24
}

func shapeOrNil(t *backend.Tensor) []int {
	if t == nil {
		return nil
	}
	return t.Shape
}
