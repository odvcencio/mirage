package mirage

import (
	"bytes"
	"testing"
)

func TestEncodeDecodeMRGRoundTrip(t *testing.T) {
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
	file, err := Encode(img, EncodeOptions{BitWidth: 4})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if file.Header.ImageWidth != 8 || file.Header.ImageHeight != 8 {
		t.Fatalf("bad image shape in header")
	}
	if file.Header.LatentChannels != PatchLatentChannels || file.Header.LatentHeight != 2 || file.Header.LatentWidth != 2 {
		t.Fatalf("bad latent shape in header: %#v", file.Header)
	}
	data, err := file.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	parsed, err := ParseFile(data)
	if err != nil {
		t.Fatalf("ParseFile: %v", err)
	}
	decoded, err := Decode(parsed, DefaultDecodeOptions())
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	psnr, err := PSNR(img, decoded)
	if err != nil {
		t.Fatalf("PSNR: %v", err)
	}
	if psnr < 14 {
		t.Fatalf("PSNR = %.2f dB, expected executable v1 smoke quality >= 14 dB", psnr)
	}
}

func TestEncodeDecodeImageReader(t *testing.T) {
	img, err := NewRGBImage(4, 4)
	if err != nil {
		t.Fatalf("NewRGBImage: %v", err)
	}
	for i := range img.Pix {
		img.Pix[i] = 0.25
	}
	var png bytes.Buffer
	if err := EncodePNG(&png, img); err != nil {
		t.Fatalf("EncodePNG: %v", err)
	}
	mrg, format, err := EncodeImageReader(png.Bytes(), EncodeOptions{BitWidth: 4})
	if err != nil {
		t.Fatalf("EncodeImageReader: %v", err)
	}
	if format != "png" {
		t.Fatalf("format = %q want png", format)
	}
	if _, err := DecodeBytesMRG(mrg, DefaultDecodeOptions()); err != nil {
		t.Fatalf("DecodeBytesMRG: %v", err)
	}
}

func TestDecodeRejectsModelMismatch(t *testing.T) {
	img, err := NewRGBImage(4, 4)
	if err != nil {
		t.Fatalf("NewRGBImage: %v", err)
	}
	file, err := Encode(img, EncodeOptions{BitWidth: 4, Seed: 99})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if _, err := Decode(file, DefaultDecodeOptions()); err == nil {
		t.Fatalf("Decode accepted a model fingerprint mismatch")
	}
	if _, err := Decode(file, DecodeOptions{Seed: 99}); err != nil {
		t.Fatalf("Decode with matching seed: %v", err)
	}
}
