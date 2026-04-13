package mirage

import (
	"math"
	"testing"
)

func TestLatentCodeRoundTripShapeAndFiniteValues(t *testing.T) {
	shape := LatentShape{Channels: 8, Height: 3, Width: 2}
	latents := make([]float32, shape.Elements())
	for i := range latents {
		latents[i] = float32(math.Sin(float64(i)*0.37) * 3)
	}
	code, err := EncodeLatents(latents, shape, 4, 123)
	if err != nil {
		t.Fatalf("EncodeLatents: %v", err)
	}
	if err := code.Validate(); err != nil {
		t.Fatalf("LatentCode.Validate: %v", err)
	}
	if len(code.Norms) != shape.Positions() {
		t.Fatalf("norm length = %d want %d", len(code.Norms), shape.Positions())
	}
	if len(code.Coordinates) != shape.Positions()*code.PackedBytesPerPosition() {
		t.Fatalf("coordinate length = %d", len(code.Coordinates))
	}
	decoded, err := DecodeLatents(code, 123)
	if err != nil {
		t.Fatalf("DecodeLatents: %v", err)
	}
	if len(decoded) != len(latents) {
		t.Fatalf("decoded length = %d want %d", len(decoded), len(latents))
	}
	var mse float64
	for i := range decoded {
		if math.IsNaN(float64(decoded[i])) || math.IsInf(float64(decoded[i]), 0) {
			t.Fatalf("decoded[%d] is invalid: %v", i, decoded[i])
		}
		d := float64(decoded[i] - latents[i])
		mse += d * d
	}
	mse /= float64(len(decoded))
	if mse >= 10 {
		t.Fatalf("latent round-trip MSE = %g, unexpectedly high", mse)
	}
}

func TestLatentEncodeRejectsBadInputs(t *testing.T) {
	shape := LatentShape{Channels: 8, Height: 1, Width: 1}
	if _, err := EncodeLatents([]float32{1}, shape, 4, 1); err == nil {
		t.Fatalf("EncodeLatents accepted wrong tensor length")
	}
	values := make([]float32, shape.Elements())
	values[0] = float32(math.NaN())
	if _, err := EncodeLatents(values, shape, 4, 1); err == nil {
		t.Fatalf("EncodeLatents accepted NaN")
	}
	if _, err := EncodeLatents(make([]float32, shape.Elements()), shape, 3, 1); err == nil {
		t.Fatalf("EncodeLatents accepted unsupported bit width")
	}
}
