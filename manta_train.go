//go:build !js

package mirage

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	mantaartifact "github.com/odvcencio/manta/artifact/manta"
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
	CropMode       string
	RandomCrops    int
	CropSeed       int64
	WeightSeed     int64
	ResumePath     string
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
	CropMode       string
	TrainingCrops  int
	RandomCrops    int
	CropSeed       int64
	ResumePath     string
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
// images using the CPU reference autograd path. Images are cropped according to
// opts.CropMode before they are converted to NCHW tensors.
func TrainMantaReferenceImages(images []RGBImage, opts MantaReferenceTrainOptions) (MantaReferenceTrainResult, error) {
	if len(images) == 0 {
		return MantaReferenceTrainResult{}, fmt.Errorf("mirage: at least one training image is required")
	}
	opts = normalizeMantaReferenceTrainOptions(opts)
	mod, err := MantaModule(opts.Config)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	tensors, err := mantaReferenceTrainTensors(images, opts)
	if err != nil {
		return MantaReferenceTrainResult{}, err
	}
	weights, err := mantaReferenceInitialWeights(mod, opts)
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
		CropMode:       opts.CropMode,
		TrainingCrops:  len(tensors),
		RandomCrops:    opts.RandomCrops,
		CropSeed:       opts.CropSeed,
		ResumePath:     opts.ResumePath,
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
	opts.CropMode = strings.ToLower(strings.TrimSpace(opts.CropMode))
	if opts.CropMode == "" {
		opts.CropMode = "center"
	}
	if opts.RandomCrops <= 0 {
		opts.RandomCrops = 1
	}
	if opts.CropSeed == 0 {
		opts.CropSeed = DefaultSeed
	}
	if opts.WeightSeed == 0 {
		opts.WeightSeed = DefaultSeed
	}
	return opts
}

func mantaReferenceTrainTensors(images []RGBImage, opts MantaReferenceTrainOptions) ([]*backend.Tensor, error) {
	switch opts.CropMode {
	case "center":
		tensors := make([]*backend.Tensor, 0, len(images))
		for i, img := range images {
			tensor, err := rgbImageToMantaTrainTensor(img, opts.Config.ImageWidth, opts.Config.ImageHeight)
			if err != nil {
				return nil, fmt.Errorf("training image %d: %w", i, err)
			}
			tensors = append(tensors, tensor)
		}
		return tensors, nil
	case "random":
		rng := rand.New(rand.NewSource(opts.CropSeed))
		tensors := make([]*backend.Tensor, 0, len(images)*opts.RandomCrops)
		for i, img := range images {
			for crop := 0; crop < opts.RandomCrops; crop++ {
				tensor, err := randomRGBImageToMantaTrainTensor(img, opts.Config.ImageWidth, opts.Config.ImageHeight, rng)
				if err != nil {
					return nil, fmt.Errorf("training image %d crop %d: %w", i, crop, err)
				}
				tensors = append(tensors, tensor)
			}
		}
		return tensors, nil
	default:
		return nil, fmt.Errorf("mirage: unknown crop mode %q", opts.CropMode)
	}
}

func mantaReferenceInitialWeights(mod *mantaartifact.Module, opts MantaReferenceTrainOptions) (map[string]*backend.Tensor, error) {
	if opts.ResumePath != "" {
		weights, err := mantaruntime.ReadWeightFile(opts.ResumePath)
		if err != nil {
			return nil, err
		}
		if err := validateMantaReferenceWeights(mod, weights.Weights); err != nil {
			return nil, fmt.Errorf("resume weights: %w", err)
		}
		return weights.Weights, nil
	}
	return mantamodels.InitMirageV1ReferenceWeights(mod, opts.WeightSeed)
}

func rgbImageToMantaTrainTensor(img RGBImage, width, height int) (*backend.Tensor, error) {
	if err := validateMantaTrainCrop(img, width, height); err != nil {
		return nil, err
	}
	x0 := (img.Width - width) / 2
	y0 := (img.Height - height) / 2
	return rgbImageCropToMantaTrainTensor(img, width, height, x0, y0)
}

func randomRGBImageToMantaTrainTensor(img RGBImage, width, height int, rng *rand.Rand) (*backend.Tensor, error) {
	if rng == nil {
		return nil, fmt.Errorf("mirage: random crop generator is nil")
	}
	if err := validateMantaTrainCrop(img, width, height); err != nil {
		return nil, err
	}
	x0 := 0
	if maxX := img.Width - width; maxX > 0 {
		x0 = rng.Intn(maxX + 1)
	}
	y0 := 0
	if maxY := img.Height - height; maxY > 0 {
		y0 = rng.Intn(maxY + 1)
	}
	return rgbImageCropToMantaTrainTensor(img, width, height, x0, y0)
}

func rgbImageCropToMantaTrainTensor(img RGBImage, width, height, x0, y0 int) (*backend.Tensor, error) {
	if err := validateMantaTrainCrop(img, width, height); err != nil {
		return nil, err
	}
	if x0 < 0 || y0 < 0 || x0+width > img.Width || y0+height > img.Height {
		return nil, fmt.Errorf("mirage: crop origin %d,%d is outside image %dx%d for crop %dx%d", x0, y0, img.Width, img.Height, width, height)
	}
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

func validateMantaTrainCrop(img RGBImage, width, height int) error {
	if err := img.validate(); err != nil {
		return err
	}
	if width <= 0 || height <= 0 {
		return fmt.Errorf("mirage: training crop dimensions must be positive")
	}
	if width%16 != 0 || height%16 != 0 {
		return fmt.Errorf("mirage: training crop must be divisible by 16")
	}
	if img.Width < width || img.Height < height {
		return fmt.Errorf("mirage: image %dx%d is smaller than requested crop %dx%d", img.Width, img.Height, width, height)
	}
	return nil
}

func validateMantaReferenceWeights(mod *mantaartifact.Module, weights map[string]*backend.Tensor) error {
	if mod == nil {
		return fmt.Errorf("nil module")
	}
	if len(weights) == 0 {
		return fmt.Errorf("no weights")
	}
	for _, param := range mod.Params {
		if param.Type.Tensor == nil {
			return fmt.Errorf("param %q is not a tensor", param.Name)
		}
		weight := weights[param.Name]
		if weight == nil {
			return fmt.Errorf("missing weight %q", param.Name)
		}
		shape, err := concreteMantaParamShape(param.Type.Tensor.Shape)
		if err != nil {
			return fmt.Errorf("param %q shape: %w", param.Name, err)
		}
		if !sameIntShape(weight.Shape, shape) {
			return fmt.Errorf("weight %q shape %v does not match module shape %v", param.Name, weight.Shape, shape)
		}
		if len(weight.F32) != weight.Elements() {
			return fmt.Errorf("weight %q storage has %d values, want %d", param.Name, len(weight.F32), weight.Elements())
		}
	}
	return nil
}

func concreteMantaParamShape(shape []string) ([]int, error) {
	out := make([]int, len(shape))
	for i, dim := range shape {
		n, err := strconv.Atoi(dim)
		if err != nil {
			return nil, fmt.Errorf("non-concrete dim %q", dim)
		}
		if n <= 0 {
			return nil, fmt.Errorf("non-positive dim %d", n)
		}
		out[i] = n
	}
	return out, nil
}

func sameIntShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
