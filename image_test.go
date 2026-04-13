package mirage

import (
	"bytes"
	"math"
	"testing"
)

func TestPNGImageRoundTrip(t *testing.T) {
	img, err := NewRGBImage(2, 2)
	if err != nil {
		t.Fatalf("NewRGBImage: %v", err)
	}
	img.Pix[img.offset(0, 0, 0)] = 1
	img.Pix[img.offset(1, 0, 1)] = 0.5
	img.Pix[img.offset(2, 1, 0)] = 0.25
	img.Pix[img.offset(0, 1, 1)] = 0.75

	var buf bytes.Buffer
	if err := EncodePNG(&buf, img); err != nil {
		t.Fatalf("EncodePNG: %v", err)
	}
	decoded, format, err := DecodeImage(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("DecodeImage: %v", err)
	}
	if format != "png" {
		t.Fatalf("format = %q want png", format)
	}
	if decoded.Width != img.Width || decoded.Height != img.Height {
		t.Fatalf("decoded dimensions = %dx%d", decoded.Width, decoded.Height)
	}
	mse, err := MSE(img, decoded)
	if err != nil {
		t.Fatalf("MSE: %v", err)
	}
	if mse > 1e-5 {
		t.Fatalf("PNG round-trip MSE = %g", mse)
	}
}

func TestPPMImageRoundTrip(t *testing.T) {
	img, err := NewRGBImage(3, 1)
	if err != nil {
		t.Fatalf("NewRGBImage: %v", err)
	}
	img.Pix[img.offset(0, 0, 0)] = 1
	img.Pix[img.offset(1, 0, 1)] = 1
	img.Pix[img.offset(2, 0, 2)] = 1
	var buf bytes.Buffer
	if err := EncodePPM(&buf, img); err != nil {
		t.Fatalf("EncodePPM: %v", err)
	}
	decoded, format, err := DecodeImage(bytes.NewReader(buf.Bytes()))
	if err != nil {
		t.Fatalf("DecodeImage: %v", err)
	}
	if format != "ppm" {
		t.Fatalf("format = %q want ppm", format)
	}
	mse, err := MSE(img, decoded)
	if err != nil {
		t.Fatalf("MSE: %v", err)
	}
	if mse != 0 {
		t.Fatalf("PPM round-trip MSE = %g", mse)
	}
}

func TestPSNRIdenticalIsInfinity(t *testing.T) {
	img, err := NewRGBImage(1, 1)
	if err != nil {
		t.Fatalf("NewRGBImage: %v", err)
	}
	psnr, err := PSNR(img, img)
	if err != nil {
		t.Fatalf("PSNR: %v", err)
	}
	if !math.IsInf(psnr, 1) {
		t.Fatalf("PSNR = %v want +Inf", psnr)
	}
}
