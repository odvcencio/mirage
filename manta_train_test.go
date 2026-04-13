//go:build !js

package mirage

import (
	"bytes"
	"testing"
)

func TestTrainMantaReferenceImagesFromDecodedPNGConverges(t *testing.T) {
	source := mantaReferenceTrainPNGSource(t, 24, 24)
	var encoded bytes.Buffer
	if err := EncodePNG(&encoded, source); err != nil {
		t.Fatal(err)
	}
	decoded, format, err := DecodeImage(bytes.NewReader(encoded.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	if format != "png" {
		t.Fatalf("format = %q want png", format)
	}
	result, err := TrainMantaReferenceImages([]RGBImage{decoded}, MantaReferenceTrainOptions{
		Config: MantaConfig{
			LatentChannels: 4,
			HyperChannels:  4,
			BitWidth:       2,
			Lambda:         0.001,
		},
		Steps:        24,
		LearningRate: 0.02,
		GradientClip: 0.5,
		CropSize:     16,
		WeightSeed:   7,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("loss %.6f -> %.6f, mse %.6f -> %.6f, rate %.6f -> %.6f",
		result.InitialLoss,
		result.FinalLoss,
		result.InitialMSE,
		result.FinalMSE,
		result.InitialRate,
		result.FinalRate,
	)
	if result.ImageWidth != 16 || result.ImageHeight != 16 {
		t.Fatalf("crop shape = %dx%d want 16x16", result.ImageWidth, result.ImageHeight)
	}
	if len(result.Losses) != 25 {
		t.Fatalf("loss history length = %d want 25", len(result.Losses))
	}
	if result.FinalLoss >= result.InitialLoss {
		t.Fatalf("loss did not decrease: initial=%.6f final=%.6f losses=%v", result.InitialLoss, result.FinalLoss, result.Losses)
	}
	if result.InitialLoss-result.FinalLoss < 0.03 {
		t.Fatalf("loss improvement %.6f below 0.03; losses=%v", result.InitialLoss-result.FinalLoss, result.Losses)
	}
	if result.FinalMSE >= result.InitialMSE {
		t.Fatalf("mse did not decrease: initial=%.6f final=%.6f", result.InitialMSE, result.FinalMSE)
	}
}

func TestTrainMantaReferenceImagesRejectsSmallImage(t *testing.T) {
	img := mantaReferenceTrainPNGSource(t, 8, 8)
	_, err := TrainMantaReferenceImages([]RGBImage{img}, MantaReferenceTrainOptions{CropSize: 16})
	if err == nil {
		t.Fatal("expected small image error")
	}
}

func mantaReferenceTrainPNGSource(t *testing.T, width, height int) RGBImage {
	t.Helper()
	img, err := NewRGBImage(width, height)
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Pix[img.offset(0, y, x)] = float32(x) / float32(width-1)
			img.Pix[img.offset(1, y, x)] = float32(y) / float32(height-1)
			img.Pix[img.offset(2, y, x)] = 0.25 + 0.5*float32((x+y)%7)/6
		}
	}
	return img
}
