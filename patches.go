package mirage

import "fmt"

// PatchShape returns the executable Go v1 latent shape for an image size.
func PatchShape(width, height int) (LatentShape, error) {
	if width <= 0 || height <= 0 {
		return LatentShape{}, fmt.Errorf("mirage: image dimensions must be positive")
	}
	return LatentShape{
		Channels: PatchLatentChannels,
		Height:   ceilDiv(height, PatchSize),
		Width:    ceilDiv(width, PatchSize),
	}, nil
}

// AnalyzePatches maps an RGB image to the executable Go v1 latent tensor. Each
// latent vector is one edge-padded 4x4 RGB patch remapped from [0,1] to [-1,1].
func AnalyzePatches(img RGBImage) ([]float32, LatentShape, error) {
	if err := img.validate(); err != nil {
		return nil, LatentShape{}, err
	}
	shape, err := PatchShape(img.Width, img.Height)
	if err != nil {
		return nil, LatentShape{}, err
	}
	latents := make([]float32, shape.Elements())
	for by := 0; by < shape.Height; by++ {
		for bx := 0; bx < shape.Width; bx++ {
			ch := 0
			for py := 0; py < PatchSize; py++ {
				y := clampInt(by*PatchSize+py, 0, img.Height-1)
				for px := 0; px < PatchSize; px++ {
					x := clampInt(bx*PatchSize+px, 0, img.Width-1)
					for c := 0; c < 3; c++ {
						latents[shape.offset(ch, by, bx)] = img.Pix[img.offset(c, y, x)]*2 - 1
						ch++
					}
				}
			}
		}
	}
	return latents, shape, nil
}

// SynthesizePatches maps executable Go v1 latents back to an RGB image and
// crops away the analysis-side edge padding.
func SynthesizePatches(latents []float32, shape LatentShape, width, height int) (RGBImage, error) {
	if err := shape.validate(); err != nil {
		return RGBImage{}, err
	}
	if shape.Channels != PatchLatentChannels {
		return RGBImage{}, fmt.Errorf("mirage: patch synthesis expects %d channels, got %d", PatchLatentChannels, shape.Channels)
	}
	if len(latents) != shape.Elements() {
		return RGBImage{}, fmt.Errorf("mirage: latent length %d does not match shape %d", len(latents), shape.Elements())
	}
	wantShape, err := PatchShape(width, height)
	if err != nil {
		return RGBImage{}, err
	}
	if wantShape != shape {
		return RGBImage{}, fmt.Errorf("mirage: latent shape %v does not match output image %dx%d", shape, width, height)
	}
	img, err := NewRGBImage(width, height)
	if err != nil {
		return RGBImage{}, err
	}
	for y := 0; y < height; y++ {
		by := y / PatchSize
		py := y % PatchSize
		for x := 0; x < width; x++ {
			bx := x / PatchSize
			px := x % PatchSize
			base := (py*PatchSize + px) * 3
			for c := 0; c < 3; c++ {
				v := latents[shape.offset(base+c, by, bx)]
				img.Pix[img.offset(c, y, x)] = clampFloat32((v+1)*0.5, 0, 1)
			}
		}
	}
	return img, nil
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampFloat32(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
