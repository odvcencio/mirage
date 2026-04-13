//go:build js && wasm

package main

import (
	"context"
	"syscall/js"

	"github.com/odvcencio/mirage"
)

func main() {
	js.Global().Set("mirageDecode", js.FuncOf(decode))
	select {}
}

func decode(this js.Value, args []js.Value) any {
	if len(args) < 1 {
		return resolvedError("mirageDecode expects a Uint8Array")
	}
	input := args[0]
	if input.IsUndefined() || input.IsNull() {
		return resolvedError("mirageDecode expects a Uint8Array")
	}
	data := make([]byte, input.Get("byteLength").Int())
	js.CopyBytesToGo(data, input)
	return promise(func() js.Value {
		result, err := mirage.DecodeBytesMRGManta(context.Background(), data, mirage.DefaultDecodeOptions())
		if err != nil {
			return errorObject(err.Error())
		}
		return imageObject(result)
	})
}

func promise(fn func() js.Value) js.Value {
	executor := js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve := args[0]
		go func() {
			resolve.Invoke(fn())
		}()
		return nil
	})
	defer executor.Release()
	return js.Global().Get("Promise").New(executor)
}

func resolvedError(message string) js.Value {
	return promise(func() js.Value {
		return errorObject(message)
	})
}

func imageObject(result mirage.MantaDecodeResult) js.Value {
	img := result.Image
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
	out.Set("backend", result.ExecutionMode)
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
