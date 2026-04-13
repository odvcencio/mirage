//go:build !js || !wasm

package main

import "fmt"

func main() {
	fmt.Println("build with GOOS=js GOARCH=wasm")
}
