//go:build !js

package mirage

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	mantamodels "github.com/odvcencio/manta/models"
	mantaruntime "github.com/odvcencio/manta/runtime"
)

func TestMantaCodecEncodeDecodeRoundTrip(t *testing.T) {
	cfg := MantaConfig{
		ImageWidth:     16,
		ImageHeight:    16,
		LatentChannels: 4,
		HyperChannels:  4,
		BitWidth:       2,
		Lambda:         0.001,
	}
	mod, err := MantaModule(cfg)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	modulePath := filepath.Join(dir, "mirage.mll")
	weightPath := filepath.Join(dir, "mirage.weights.mll")
	if err := WriteMantaMLL(modulePath, cfg); err != nil {
		t.Fatal(err)
	}
	weights, err := mantamodels.InitMirageV1ReferenceWeights(mod, 7)
	if err != nil {
		t.Fatal(err)
	}
	if err := mantaruntime.NewWeightFile(weights).WriteFile(weightPath); err != nil {
		t.Fatal(err)
	}
	codec, err := LoadMantaCodec(context.Background(), MantaCodecOptions{
		ModulePath: modulePath,
		WeightPath: weightPath,
	})
	if err != nil {
		t.Fatal(err)
	}
	source := mantaCodecTestImage(t, 16, 16)
	file, err := codec.Encode(context.Background(), source)
	if err != nil {
		t.Fatal(err)
	}
	if file.Header.ImageWidth != 16 || file.Header.ImageHeight != 16 {
		t.Fatalf("image shape = %dx%d want 16x16", file.Header.ImageWidth, file.Header.ImageHeight)
	}
	if file.Header.LatentChannels != 4 || file.Header.BitWidth != 2 {
		t.Fatalf("unexpected header: %+v", file.Header)
	}
	if len(file.Payloads.CZ) == 0 || len(file.Payloads.CCoords) == 0 || len(file.Payloads.CNorms) == 0 {
		t.Fatalf("expected non-empty learned payloads: c_z=%d c_coords=%d c_norms=%d", len(file.Payloads.CZ), len(file.Payloads.CCoords), len(file.Payloads.CNorms))
	}
	decoded, err := codec.Decode(context.Background(), file)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Width != 16 || decoded.Height != 16 {
		t.Fatalf("decoded shape = %dx%d want 16x16", decoded.Width, decoded.Height)
	}
	psnr, err := PSNR(source, decoded)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(psnr) {
		t.Fatalf("PSNR is NaN")
	}
}

func mantaCodecTestImage(t *testing.T, width, height int) RGBImage {
	t.Helper()
	img, err := NewRGBImage(width, height)
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			img.Pix[img.offset(0, y, x)] = float32(x) / float32(img.Width-1)
			img.Pix[img.offset(1, y, x)] = float32(y) / float32(img.Height-1)
			img.Pix[img.offset(2, y, x)] = 0.25 + 0.5*float32((x+y)%7)/6
		}
	}
	return img
}
