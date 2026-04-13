package mirage

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
)

// DecodePPM reads binary PPM (P6) image data.
func DecodePPM(r io.Reader) (RGBImage, error) {
	br := bufio.NewReader(r)
	magic, err := readPPMToken(br)
	if err != nil {
		return RGBImage{}, err
	}
	if magic != "P6" {
		return RGBImage{}, fmt.Errorf("mirage: unsupported PPM magic %q", magic)
	}
	width, err := readPPMInt(br, "width")
	if err != nil {
		return RGBImage{}, err
	}
	height, err := readPPMInt(br, "height")
	if err != nil {
		return RGBImage{}, err
	}
	maxVal, err := readPPMInt(br, "maxval")
	if err != nil {
		return RGBImage{}, err
	}
	if maxVal <= 0 || maxVal > 65535 {
		return RGBImage{}, fmt.Errorf("mirage: invalid PPM maxval %d", maxVal)
	}
	img, err := NewRGBImage(width, height)
	if err != nil {
		return RGBImage{}, err
	}
	samples := width * height * 3
	if maxVal < 256 {
		data := make([]byte, samples)
		if _, err := io.ReadFull(br, data); err != nil {
			return RGBImage{}, err
		}
		for i := 0; i < width*height; i++ {
			x := i % width
			y := i / width
			img.Pix[img.offset(0, y, x)] = float32(data[i*3]) / float32(maxVal)
			img.Pix[img.offset(1, y, x)] = float32(data[i*3+1]) / float32(maxVal)
			img.Pix[img.offset(2, y, x)] = float32(data[i*3+2]) / float32(maxVal)
		}
		return img, nil
	}
	data := make([]byte, samples*2)
	if _, err := io.ReadFull(br, data); err != nil {
		return RGBImage{}, err
	}
	for i := 0; i < width*height; i++ {
		x := i % width
		y := i / width
		r := uint16(data[i*6])<<8 | uint16(data[i*6+1])
		g := uint16(data[i*6+2])<<8 | uint16(data[i*6+3])
		b := uint16(data[i*6+4])<<8 | uint16(data[i*6+5])
		img.Pix[img.offset(0, y, x)] = float32(r) / float32(maxVal)
		img.Pix[img.offset(1, y, x)] = float32(g) / float32(maxVal)
		img.Pix[img.offset(2, y, x)] = float32(b) / float32(maxVal)
	}
	return img, nil
}

// EncodePPM writes an RGBImage as binary PPM (P6) with 8-bit samples.
func EncodePPM(w io.Writer, img RGBImage) error {
	if err := img.validate(); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "P6\n%d %d\n255\n", img.Width, img.Height); err != nil {
		return err
	}
	buf := make([]byte, img.Width*img.Height*3)
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			i := (y*img.Width + x) * 3
			buf[i] = floatToByte(img.Pix[img.offset(0, y, x)])
			buf[i+1] = floatToByte(img.Pix[img.offset(1, y, x)])
			buf[i+2] = floatToByte(img.Pix[img.offset(2, y, x)])
		}
	}
	_, err := w.Write(buf)
	return err
}

func readPPMInt(r *bufio.Reader, name string) (int, error) {
	token, err := readPPMToken(r)
	if err != nil {
		return 0, err
	}
	n, err := strconv.Atoi(token)
	if err != nil {
		return 0, fmt.Errorf("mirage: invalid PPM %s %q", name, token)
	}
	return n, nil
}

func readPPMToken(r *bufio.Reader) (string, error) {
	for {
		b, err := r.ReadByte()
		if err != nil {
			return "", err
		}
		if isPPMSpace(b) {
			continue
		}
		if b == '#' {
			if _, err := r.ReadString('\n'); err != nil {
				return "", err
			}
			continue
		}
		if err := r.UnreadByte(); err != nil {
			return "", err
		}
		break
	}
	var token []byte
	for {
		b, err := r.ReadByte()
		if err != nil {
			if len(token) != 0 && err == io.EOF {
				return string(token), nil
			}
			return "", err
		}
		if isPPMSpace(b) {
			break
		}
		token = append(token, b)
	}
	return string(token), nil
}

func isPPMSpace(b byte) bool {
	switch b {
	case ' ', '\t', '\n', '\r', '\v', '\f':
		return true
	default:
		return false
	}
}
