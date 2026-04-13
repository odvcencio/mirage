//go:build !js

package mirage

import (
	"fmt"

	mantamodels "github.com/odvcencio/manta/models"
	mantaruntime "github.com/odvcencio/manta/runtime"
	"github.com/odvcencio/manta/runtime/backend"
)

const defaultMantaReferenceTrainCropSize = 16

// MantaReferenceTrainOptions controls a tiny image-backed reference training
// run over the Manta Mirage v1 graph.
type MantaReferenceTrainOptions struct {
	Config         MantaConfig
	Steps          int
	LearningRate   float32
	GradientClip   float32
	WeightDecay    float32
	MinGDNBeta     float32
	CropSize       int
	WeightSeed     int64
	CheckpointPath string
}

// MantaReferenceTrainResult summarizes convergence for one reference run.
type MantaReferenceTrainResult struct {
	Images         int
	Steps          int
	ImageWidth     int
	ImageHeight    int
	LatentChannels int
	HyperChannels  int
	BitWidth       int
	Factorization  Factorization
	Lambda         float64
	InitialLoss    float32
	FinalLoss      float32
	InitialMSE     float32
	FinalMSE       float32
	InitialRate    float32
	FinalRate      float32
	Losses         []float32
	MSEs           []float32
	Rates          []float32
	GradientNorms  []MantaReferenceGradientNorms
	CheckpointPath string
}

// MantaReferenceGradientNorms records raw reference autograd gradient L2 norms
// by Mirage graph region before clipping.
type MantaReferenceGradientNorms struct {
	Total          float32
	Analysis       float32
	HyperAnalysis  float32
	HyperSynthesis float32
	Synthesis      float32
	Prior          float32
	Other          float32
}

// TrainMantaReferenceImages trains the Manta Mirage v1 graph on decoded RGB
// images using the CPU reference autograd path. Each image is center-cropped to
// the configured Manta input shape before it is converted to an NCHW tensor.
func TrainMantaReferenceImages(images []RGBImage, opts MantaReferenceTrainOptions) (MantaReferenceTrainResult, error) {
	if len(images) == 0 {
		return MantaReferenceTrainResult{}, fmt.Errorf("mirage: at least one training image is required")
	}
	opts = normalizeMantaReferenceTrainOptions(opts)
	mod, err := MantaModule(opts.Config)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	tensors := make([]*backend.Tensor, 0, len(images))
	for i, img := range images {
		tensor, err := rgbImageToMantaTrainTensor(img, opts.Config.ImageWidth, opts.Config.ImageHeight)
		if err != nil {
			return MantaReferenceTrainResult{}, fmt.Errorf("training image %d: %w", i, err)
		}
		tensors = append(tensors, tensor)
	}
	weights, err := mantamodels.InitMirageV1ReferenceWeights(mod, opts.WeightSeed)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	initial, err := mantamodels.MirageV1ReferenceEval(mod, weights, tensors)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	history, err := mantamodels.TrainMirageV1Reference(mod, weights, tensors, mantamodels.MirageV1ReferenceTrainConfig{
		Steps:        opts.Steps,
		LearningRate: opts.LearningRate,
		GradientClip: opts.GradientClip,
		WeightDecay:  opts.WeightDecay,
		MinGDNBeta:   opts.MinGDNBeta,
	})
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	final, err := mantamodels.MirageV1ReferenceEval(mod, weights, tensors)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	if opts.CheckpointPath != "" {
		if err := mantaruntime.NewWeightFile(weights).WriteFile(opts.CheckpointPath); err != nil {
			return MantaReferenceTrainResult{}, err
		}
	}
	return MantaReferenceTrainResult{
		Images:         len(images),
		Steps:          opts.Steps,
		ImageWidth:     opts.Config.ImageWidth,
		ImageHeight:    opts.Config.ImageHeight,
		LatentChannels: opts.Config.LatentChannels,
		HyperChannels:  opts.Config.HyperChannels,
		BitWidth:       opts.Config.BitWidth,
		Factorization:  opts.Config.Factorization,
		Lambda:         opts.Config.Lambda,
		InitialLoss:    initial.Loss,
		FinalLoss:      final.Loss,
		InitialMSE:     initial.MSE,
		FinalMSE:       final.MSE,
		InitialRate:    initial.Rate,
		FinalRate:      final.Rate,
		Losses:         append([]float32(nil), history.Losses...),
		MSEs:           append([]float32(nil), history.MSEs...),
		Rates:          append([]float32(nil), history.Rates...),
		GradientNorms:  convertMantaReferenceGradientNorms(history.GradientNorms),
		CheckpointPath: opts.CheckpointPath,
	}, nil
}

func convertMantaReferenceGradientNorms(in []mantamodels.MirageV1ReferenceGradientNorms) []MantaReferenceGradientNorms {
	if len(in) == 0 {
		return nil
	}
	out := make([]MantaReferenceGradientNorms, len(in))
	for i, item := range in {
		out[i] = MantaReferenceGradientNorms{
			Total:          item.Total,
			Analysis:       item.Analysis,
			HyperAnalysis:  item.HyperAnalysis,
			HyperSynthesis: item.HyperSynthesis,
			Synthesis:      item.Synthesis,
			Prior:          item.Prior,
			Other:          item.Other,
		}
	}
	return out
}

func normalizeMantaReferenceTrainOptions(opts MantaReferenceTrainOptions) MantaReferenceTrainOptions {
	if opts.CropSize == 0 {
		opts.CropSize = defaultMantaReferenceTrainCropSize
	}
	if opts.Config.ImageWidth == 0 {
		opts.Config.ImageWidth = opts.CropSize
	}
	if opts.Config.ImageHeight == 0 {
		opts.Config.ImageHeight = opts.CropSize
	}
	if opts.Config.ImageChannels == 0 {
		opts.Config.ImageChannels = 3
	}
	if opts.Config.LatentChannels == 0 {
		opts.Config.LatentChannels = 4
	}
	if opts.Config.HyperChannels == 0 {
		opts.Config.HyperChannels = opts.Config.LatentChannels
	}
	if opts.Config.BitWidth == 0 {
		opts.Config.BitWidth = 2
	}
	if opts.Config.Lambda == 0 {
		opts.Config.Lambda = 0.001
	}
	if opts.Steps <= 0 {
		opts.Steps = 16
	}
	if opts.LearningRate == 0 {
		opts.LearningRate = 0.01
	}
	if opts.GradientClip == 0 {
		opts.GradientClip = 1
	}
	if opts.WeightSeed == 0 {
		opts.WeightSeed = DefaultSeed
	}
	return opts
}

func rgbImageToMantaTrainTensor(img RGBImage, width, height int) (*backend.Tensor, error) {
	if err := img.validate(); err != nil {
		return nil, err
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("mirage: training crop dimensions must be positive")
	}
	if width%16 != 0 || height%16 != 0 {
		return nil, fmt.Errorf("mirage: training crop must be divisible by 16")
	}
	if img.Width < width || img.Height < height {
		return nil, fmt.Errorf("mirage: image %dx%d is smaller than requested crop %dx%d", img.Width, img.Height, width, height)
	}
	x0 := (img.Width - width) / 2
	y0 := (img.Height - height) / 2
	values := make([]float32, 3*height*width)
	for c := 0; c < 3; c++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				values[(c*height+y)*width+x] = img.Pix[img.offset(c, y0+y, x0+x)]
			}
		}
	}
	return backend.NewTensorF16([]int{1, 3, height, width}, values), nil
}
