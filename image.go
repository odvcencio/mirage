package mirage

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"io"
	"math"
)

// RGBImage is a channel-first RGB tensor with values nominally in [0, 1].
type RGBImage struct {
	Width  int
	Height int
	Pix    []float32
}

// NewRGBImage allocates a zero-filled channel-first RGB image.
func NewRGBImage(width, height int) (RGBImage, error) {
	if width <= 0 || height <= 0 {
		return RGBImage{}, fmt.Errorf("mirage: image dimensions must be positive")
	}
	return RGBImage{
		Width:  width,
		Height: height,
		Pix:    make([]float32, 3*width*height),
	}, nil
}

func (img RGBImage) validate() error {
	if img.Width <= 0 || img.Height <= 0 {
		return fmt.Errorf("mirage: image dimensions must be positive")
	}
	want := 3 * img.Width * img.Height
	if len(img.Pix) != want {
		return fmt.Errorf("mirage: RGB image has %d values, want %d", len(img.Pix), want)
	}
	return nil
}

func (img RGBImage) offset(c, y, x int) int {
	return c*img.Width*img.Height + y*img.Width + x
}

// CenterCropRGB returns the centered width x height crop of an RGB image.
func CenterCropRGB(img RGBImage, width, height int) (RGBImage, error) {
	if err := img.validate(); err != nil {
		return RGBImage{}, err
	}
	if width <= 0 || height <= 0 {
		return RGBImage{}, fmt.Errorf("mirage: crop dimensions must be positive")
	}
	if img.Width < width || img.Height < height {
		return RGBImage{}, fmt.Errorf("mirage: image %dx%d is smaller than requested crop %dx%d", img.Width, img.Height, width, height)
	}
	out, err := NewRGBImage(width, height)
	if err != nil {
		return RGBImage{}, err
	}
	x0 := (img.Width - width) / 2
	y0 := (img.Height - height) / 2
	for c := 0; c < 3; c++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				out.Pix[out.offset(c, y, x)] = img.Pix[img.offset(c, y0+y, x0+x)]
			}
		}
	}
	return out, nil
}

// DecodeImage reads PNG, JPEG, or binary PPM (P6) into a channel-first RGBImage.
func DecodeImage(r io.Reader) (RGBImage, string, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return RGBImage{}, "", err
	}
	if bytes.HasPrefix(bytes.TrimSpace(data), []byte("P6")) {
		img, err := DecodePPM(bytes.NewReader(data))
		if err != nil {
			return RGBImage{}, "", err
		}
		return img, "ppm", nil
	}
	decoded, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return RGBImage{}, "", err
	}
	return ImageToRGB(decoded), format, nil
}

// ImageToRGB converts any Go image to a channel-first RGBImage.
func ImageToRGB(src image.Image) RGBImage {
	bounds := src.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	img, _ := NewRGBImage(width, height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := src.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			img.Pix[img.offset(0, y, x)] = float32(r) / 65535
			img.Pix[img.offset(1, y, x)] = float32(g) / 65535
			img.Pix[img.offset(2, y, x)] = float32(b) / 65535
		}
	}
	return img
}

// ToNRGBA converts a channel-first RGBImage to an 8-bit Go image.
func (img RGBImage) ToNRGBA() (*image.NRGBA, error) {
	if err := img.validate(); err != nil {
		return nil, err
	}
	out := image.NewNRGBA(image.Rect(0, 0, img.Width, img.Height))
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			out.SetNRGBA(x, y, color.NRGBA{
				R: floatToByte(img.Pix[img.offset(0, y, x)]),
				G: floatToByte(img.Pix[img.offset(1, y, x)]),
				B: floatToByte(img.Pix[img.offset(2, y, x)]),
				A: 255,
			})
		}
	}
	return out, nil
}

// EncodePNG writes an RGBImage as an 8-bit PNG.
func EncodePNG(w io.Writer, img RGBImage) error {
	nrgba, err := img.ToNRGBA()
	if err != nil {
		return err
	}
	return png.Encode(w, nrgba)
}

// EncodeJPEG writes an RGBImage as an 8-bit JPEG.
func EncodeJPEG(w io.Writer, img RGBImage, quality int) error {
	nrgba, err := img.ToNRGBA()
	if err != nil {
		return err
	}
	if quality <= 0 {
		quality = 90
	}
	if quality > 100 {
		quality = 100
	}
	return jpeg.Encode(w, nrgba, &jpeg.Options{Quality: quality})
}

// MSE returns per-sample mean squared error for two RGB images.
func MSE(a, b RGBImage) (float64, error) {
	if err := a.validate(); err != nil {
		return 0, err
	}
	if err := b.validate(); err != nil {
		return 0, err
	}
	if a.Width != b.Width || a.Height != b.Height {
		return 0, fmt.Errorf("mirage: image sizes differ")
	}
	var sum float64
	for i := range a.Pix {
		d := float64(a.Pix[i] - b.Pix[i])
		sum += d * d
	}
	return sum / float64(len(a.Pix)), nil
}

// PSNR returns peak signal-to-noise ratio in dB for images in [0, 1].
func PSNR(a, b RGBImage) (float64, error) {
	mse, err := MSE(a, b)
	if err != nil {
		return 0, err
	}
	if mse == 0 {
		return math.Inf(1), nil
	}
	return 10 * math.Log10(1/mse), nil
}

func floatToByte(v float32) byte {
	if v <= 0 || math.IsNaN(float64(v)) {
		return 0
	}
	if v >= 1 {
		return 255
	}
	return byte(math.Round(float64(v) * 255))
}
