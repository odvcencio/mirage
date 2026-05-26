package mirage

import (
	"context"
	"fmt"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
	"github.com/odvcencio/manta/runtime/backends/webgpu"
)

// MantaDecodeResult contains the decoded image plus the execution mode reported
// by the Manta WebGPU TurboQuant decode step.
type MantaDecodeResult struct {
	Image         RGBImage
	ExecutionMode string
}

// DecodeBytesMRGManta decodes the current executable v1 .mrg stream while
// routing the TurboQuant latent reconstruction through Manta's WebGPU backend.
// The arithmetic decoder remains CPU-side by design; in browsers with WebGPU,
// the latent decode runs through a real GPU compute dispatch.
func DecodeBytesMRGManta(ctx context.Context, data []byte, opts DecodeOptions) (MantaDecodeResult, error) {
	file, err := ParseFile(data)
	if err != nil {
		return MantaDecodeResult{}, err
	}
	return DecodeManta(ctx, file, opts)
}

// DecodeManta mirrors Decode but replaces DecodeLatents with a Manta
// turboquant_decode pipeline. Non-WebGPU hosts still use the backend-owned
// reference fallback, which is useful for deterministic tests.
func DecodeManta(ctx context.Context, file File, opts DecodeOptions) (MantaDecodeResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := file.Header.Validate(); err != nil {
		return MantaDecodeResult{}, err
	}
	if opts.Seed == 0 {
		opts.Seed = DefaultSeed
	}
	wantFingerprint := ModelFingerprintForSeed(opts.Seed)
	if !opts.AllowModelMismatch && file.Header.ModelFingerprint != wantFingerprint {
		return MantaDecodeResult{}, fmt.Errorf("mirage: model fingerprint mismatch")
	}
	shape := file.Header.LatentShape()
	if shape.Channels != PatchLatentChannels {
		return MantaDecodeResult{}, fmt.Errorf("mirage: unsupported Go v1 latent channel count %d", shape.Channels)
	}
	if shape.Height != ceilDiv(int(file.Header.ImageHeight), PatchSize) || shape.Width != ceilDiv(int(file.Header.ImageWidth), PatchSize) {
		return MantaDecodeResult{}, fmt.Errorf("mirage: latent shape does not match patch model")
	}
	code, err := DecodeLatentPayloads(file.Payloads, shape, int(file.Header.BitWidth), file.Header.Factorization())
	if err != nil {
		return MantaDecodeResult{}, err
	}
	latents, mode, err := decodeLatentsManta(ctx, code, opts.Seed)
	if err != nil {
		return MantaDecodeResult{}, err
	}
	img, err := SynthesizePatches(latents, shape, int(file.Header.ImageWidth), int(file.Header.ImageHeight))
	if err != nil {
		return MantaDecodeResult{}, err
	}
	return MantaDecodeResult{Image: img, ExecutionMode: mode}, nil
}

func decodeLatentsManta(ctx context.Context, code LatentCode, seed int64) ([]float32, string, error) {
	if err := code.Validate(); err != nil {
		return nil, "", err
	}
	symbols, err := UnpackCoordinateSymbols(code)
	if err != nil {
		return nil, "", err
	}
	coords := make([]float32, code.Shape.Elements())
	for y := 0; y < code.Shape.Height; y++ {
		for x := 0; x < code.Shape.Width; x++ {
			pos := y*code.Shape.Width + x
			for c := 0; c < code.Shape.Channels; c++ {
				coords[code.Shape.offset(c, y, x)] = float32(symbols[pos*code.Shape.Channels+c])
			}
		}
	}
	norms := make([]float32, len(code.Norms))
	for i, value := range code.Norms {
		norms[i] = float32(value)
	}
	mod := turboQuantDecodeModule(code.Shape, code.BitWidth, seed)
	prog, err := mantaruntime.New(webgpu.New()).Load(ctx, mod)
	if err != nil {
		return nil, "", err
	}
	result, err := prog.Run(ctx, backend.Request{
		Entry: "decode_latents",
		Inputs: map[string]any{
			"coords": qTensorForBitWidth(code.BitWidth, []int{1, code.Shape.Channels, code.Shape.Height, code.Shape.Width}, coords),
			"norms":  backend.NewTensorQNorm([]int{1, code.Shape.Height, code.Shape.Width}, norms),
		},
	})
	if err != nil {
		return nil, "", err
	}
	value, ok := result.Outputs["latents"]
	if !ok {
		return nil, "", fmt.Errorf("mirage: Manta decode did not return latents")
	}
	tensor, ok := value.Data.(*backend.Tensor)
	if !ok || tensor == nil {
		return nil, "", fmt.Errorf("mirage: Manta decode returned %T, want tensor", value.Data)
	}
	if len(tensor.F32) != code.Shape.Elements() {
		return nil, "", fmt.Errorf("mirage: Manta latent length %d does not match shape %d", len(tensor.F32), code.Shape.Elements())
	}
	mode := "unknown"
	if raw, ok := value.Metadata["execution_mode"].(string); ok {
		mode = raw
	}
	return append([]float32(nil), tensor.F32...), mode, nil
}

func turboQuantDecodeModule(shape LatentShape, bitWidth int, seed int64) *mantaartifact.Module {
	bitsDType := fmt.Sprintf("q%d", bitWidth)
	coordsShape := []string{"1", itoa(shape.Channels), itoa(shape.Height), itoa(shape.Width)}
	normShape := []string{"1", itoa(shape.Height), itoa(shape.Width)}
	mod := mantaartifact.NewModule("mirage_patch_turboquant_decode")
	mod.Requirements.Capabilities = []string{
		mantaartifact.CapabilityImageOps,
		mantaartifact.CapabilityTurboQuant,
		mantaartifact.CapabilityHostFallback,
	}
	mod.EntryPoints = []mantaartifact.EntryPoint{{
		Name: "decode_latents",
		Kind: mantaartifact.EntryPointPipeline,
		Inputs: []mantaartifact.ValueBinding{
			{Name: "coords", Type: mantaTensorType(bitsDType, coordsShape)},
			{Name: "norms", Type: mantaTensorType("q_norm", normShape)},
		},
		Outputs: []mantaartifact.ValueBinding{
			{Name: "latents", Type: mantaTensorType("f16", coordsShape)},
		},
	}}
	mod.Buffers = []mantaartifact.Buffer{{Name: "latents", DType: "f16", Shape: coordsShape}}
	mod.Steps = []mantaartifact.Step{
		{
			Entry:   "decode_latents",
			Kind:    mantaartifact.StepTurboQDecode,
			Name:    "turboquant_decode",
			Inputs:  []string{"coords", "norms"},
			Outputs: []string{"latents"},
			Attributes: map[string]string{
				"bits": itoa(bitWidth),
				"seed": itoa64(seed),
			},
		},
		{Entry: "decode_latents", Kind: mantaartifact.StepReturn, Name: "return", Outputs: []string{"latents"}},
	}
	return mod
}

func mantaTensorType(dtype string, shape []string) mantaartifact.ValueType {
	return mantaartifact.ValueType{
		Kind:   mantaartifact.ValueTensor,
		Tensor: &mantaartifact.TensorType{DType: dtype, Shape: append([]string(nil), shape...)},
	}
}

func qTensorForBitWidth(bitWidth int, shape []int, data []float32) *backend.Tensor {
	switch bitWidth {
	case 2:
		return backend.NewTensorQ2(shape, data)
	case 4:
		return backend.NewTensorQ4(shape, data)
	case 8:
		return backend.NewTensorQ8(shape, data)
	default:
		return backend.NewTensorQ4(shape, data)
	}
}

func itoa(n int) string {
	return fmt.Sprintf("%d", n)
}

func itoa64(n int64) string {
	return fmt.Sprintf("%d", n)
}
