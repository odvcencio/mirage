//go:build js && wasm

package main

import (
	"syscall/js"

	"github.com/odvcencio/mirage"
)

func main() {
	js.Global().Set("mirageDecode", js.FuncOf(decode))
	select {}
}

func decode(this js.Value, args []js.Value) any {
	if len(args) < 1 {
		return errorObject("mirageDecode expects a Uint8Array")
	}
	input := args[0]
	if input.IsUndefined() || input.IsNull() {
		return errorObject("mirageDecode expects a Uint8Array")
	}
	data := make([]byte, input.Get("byteLength").Int())
	js.CopyBytesToGo(data, input)
	img, err := mirage.DecodeBytesMRG(data, mirage.DefaultDecodeOptions())
	if err != nil {
		return errorObject(err.Error())
	}
	rgba := make([]byte, img.Width*img.Height*4)
	for y := 0; y < img.Height; y++ {
		for x := 0; x < img.Width; x++ {
			dst := (y*img.Width + x) * 4
			rgba[dst] = floatToByte(img.Pix[(0*img.Height+y)*img.Width+x])
			rgba[dst+1] = floatToByte(img.Pix[(1*img.Height+y)*img.Width+x])
			rgba[dst+2] = floatToByte(img.Pix[(2*img.Height+y)*img.Width+x])
			rgba[dst+3] = 255
		}
	}
	out := js.Global().Get("Object").New()
	out.Set("ok", true)
	out.Set("width", img.Width)
	out.Set("height", img.Height)
	bytes := js.Global().Get("Uint8ClampedArray").New(len(rgba))
	js.CopyBytesToJS(bytes, rgba)
	out.Set("rgba", bytes)
	return out
}

func errorObject(message string) js.Value {
	out := js.Global().Get("Object").New()
	out.Set("ok", false)
	out.Set("error", message)
	return out
}

func floatToByte(v float32) byte {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 255
	}
	return byte(v*255 + 0.5)
}
