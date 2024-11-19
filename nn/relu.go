package nn

import "ruinsTorch/core"

type ReLU struct{}

// 创建 ReLU 模块
func NewReLU() *ReLU {
	return &ReLU{}
}

// 前向传播
func (r *ReLU) Forward(input *core.CPUTensor) (*core.CPUTensor, error) {
	resultData := make([]float64, len(input.Data))
	for i, val := range input.Data {
		if val > 0 {
			resultData[i] = val
		} else {
			resultData[i] = 0
		}
	}
	result, _ := core.NewCPUTensorBySlice(resultData, input.Shape, input.RequiresGrad)

	// 构建反向传播规则
	if input.RequiresGrad {
		result.Creator = &core.Function{
			Inputs: []*core.CPUTensor{input},
			Output: result,
			Backward: func(grad []float64) []float64 {
				gradInput := make([]float64, len(input.Data))
				for i, val := range input.Data {
					if val > 0 {
						gradInput[i] = grad[i] // 梯度保留
					} else {
						gradInput[i] = 0 // 梯度为 0
					}
				}
				return gradInput
			},
		}
	}
	return result, nil
}

// ReLU 无参数，因此 Parameters 和 ZeroGrad 可为空实现
func (r *ReLU) Parameters() []*core.CPUTensor {
	return nil
}

func (r *ReLU) ZeroGrad() {}
