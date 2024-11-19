package nn

import (
	"errors"
	"math"
	"ruinsTorch/core"
)

// MSELoss 均方误差损失
type MSELoss struct{}

// 创建 MSELoss 实例
func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

// 前向传播
func (m *MSELoss) Forward(predictions *core.CPUTensor, targets *core.CPUTensor) (*core.CPUTensor, error) {
	if len(predictions.Data) != len(targets.Data) {
		return nil, errors.New("predictions and targets must have the same size")
	}

	// 计算均方误差
	diffData := make([]float64, len(predictions.Data))
	for i := range predictions.Data {
		diffData[i] = predictions.Data[i] - targets.Data[i]
	}
	diff, _ := core.NewCPUTensorBySlice(diffData, predictions.Shape, predictions.RequiresGrad)

	squared, _ := core.Mul(diff, diff) // 平方
	loss, _ := core.Mean(squared)      // 平均值

	// 反向传播
	if predictions.RequiresGrad {
		loss.Creator = &core.Function{
			Inputs: []*core.CPUTensor{predictions, targets},
			Output: loss,
			Backward: func(grad []float64) []float64 {
				gradPredictions := make([]float64, len(predictions.Data))
				for i := range predictions.Data {
					gradPredictions[i] = grad[0] * 2 * diffData[i] / float64(len(predictions.Data))
				}
				return gradPredictions
			},
		}
	}
	return loss, nil
}

// CrossEntropyLoss 交叉熵损失
type CrossEntropyLoss struct{}

// 创建 CrossEntropyLoss 实例
func NewCrossEntropyLoss() *CrossEntropyLoss {
	return &CrossEntropyLoss{}
}

// 前向传播
func (c *CrossEntropyLoss) Forward(predictions *core.CPUTensor, targets *core.CPUTensor) (*core.CPUTensor, error) {
	if len(predictions.Shape) != 2 || len(targets.Shape) != 1 {
		return nil, errors.New("predictions must be 2D and targets must be 1D")
	}
	if predictions.Shape[0] != targets.Shape[0] {
		return nil, errors.New("batch size of predictions and targets must match")
	}

	batchSize := predictions.Shape[0]
	numClasses := predictions.Shape[1]

	// 计算交叉熵损失
	lossData := 0.0
	for i := 0; i < batchSize; i++ {
		targetClass := int(targets.Data[i])
		if targetClass < 0 || targetClass >= numClasses {
			return nil, errors.New("invalid target class index")
		}
		prob := predictions.Data[i*numClasses+targetClass]
		if prob <= 0 {
			return nil, errors.New("logarithm of zero or negative probability")
		}
		lossData -= math.Log(prob)
	}
	lossValue := lossData / float64(batchSize)
	loss, _ := core.NewCPUTensorBySlice([]float64{lossValue}, []int{1}, predictions.RequiresGrad)

	// 反向传播
	if predictions.RequiresGrad {
		loss.Creator = &core.Function{
			Inputs: []*core.CPUTensor{predictions, targets},
			Output: loss,
			Backward: func(grad []float64) []float64 {
				gradPredictions := make([]float64, len(predictions.Data))
				for i := 0; i < batchSize; i++ {
					targetClass := int(targets.Data[i])
					for j := 0; j < numClasses; j++ {
						if j == targetClass {
							gradPredictions[i*numClasses+j] = -grad[0] / (predictions.Data[i*numClasses+j] * float64(batchSize))
						} else {
							gradPredictions[i*numClasses+j] = 0
						}
					}
				}
				return gradPredictions
			},
		}
	}
	return loss, nil
}
