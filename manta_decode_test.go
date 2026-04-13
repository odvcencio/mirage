package mirage

import (
	"context"
	"testing"
)

func TestDecodeMantaMatchesPatchDecode(t *testing.T) {
	img, err := NewRGBImage(8, 8)
	if err != nil {
		t.Fatalf("NewRGBImage: %v", err)
	}
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			img.Pix[img.offset(0, y, x)] = float32(x) / float32(img.Width-1)
			img.Pix[img.offset(1, y, x)] = float32(y) / float32(img.Height-1)
			img.Pix[img.offset(2, y, x)] = float32(x+y) / float32(img.Width+img.Height-2)
		}
	}
	data, err := EncodeBytesMRG(img, EncodeOptions{BitWidth: 4})
	if err != nil {
		t.Fatalf("EncodeBytesMRG: %v", err)
	}
	host, err := DecodeBytesMRG(data, DefaultDecodeOptions())
	if err != nil {
		t.Fatalf("DecodeBytesMRG: %v", err)
	}
	manta, err := DecodeBytesMRGManta(context.Background(), data, DefaultDecodeOptions())
	if err != nil {
		t.Fatalf("DecodeBytesMRGManta: %v", err)
	}
	if manta.ExecutionMode == "" {
		t.Fatalf("missing execution mode")
	}
	if !sameImage(host, manta.Image) {
		t.Fatalf("Manta decode does not match host patch decode")
	}
}

func TestTurboQuantDecodeModuleSurface(t *testing.T) {
	mod := turboQuantDecodeModule(LatentShape{Channels: 4, Height: 2, Width: 3}, 2, 99)
	if err := mod.Validate(); err != nil {
		t.Fatal(err)
	}
	if got := mod.EntryPoints[0].Name; got != "decode_latents" {
		t.Fatalf("entry = %q", got)
	}
	if got := mod.Steps[0].Kind; got != "turboquant_decode" {
		t.Fatalf("step kind = %q", got)
	}
	if got := mod.Steps[0].Attributes["seed"]; got != "99" {
		t.Fatalf("seed attr = %q", got)
	}
}

func sameImage(a, b RGBImage) bool {
	if a.Width != b.Width || a.Height != b.Height || len(a.Pix) != len(b.Pix) {
		return false
	}
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			return false
		}
	}
	return true
}
