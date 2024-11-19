package core

import "errors"

func Add(a, b *CPUTensor) (*CPUTensor, error) {
	if len(a.Data) != len(b.Data) {
		return nil, errors.New("CPUTensor sizes must match")
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return nil, errors.New("tensor shapes must match")
		}
	}

	// 前向传播
	resultData := make([]float64, len(a.Data))
	for i := range resultData {
		resultData[i] = a.Data[i] + b.Data[i]
	}
	result, _ := NewCPUTensorBySlice(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad)

	// 反向传播
	if result.RequiresGrad {
		result.Creator = &Function{
			Inputs: []*CPUTensor{a, b},
			Output: result,
			Backward: func(grad []float64) []float64 {
				gradA := make([]float64, len(a.Data))
				gradB := make([]float64, len(b.Data))
				copy(gradA, grad) // 加法的梯度直接传递
				copy(gradB, grad)
				return append(gradA, gradB...)
			},
		}
	}
	return result, nil
}

func Sub(a, b *CPUTensor) (*CPUTensor, error) {
	if len(a.Data) != len(b.Data) {
		return nil, errors.New("CPUTensor sizes must match")
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return nil, errors.New("tensor shapes must match")
		}
	}

	// 前向传播
	resultData := make([]float64, len(a.Data))
	for i := range resultData {
		resultData[i] = a.Data[i] - b.Data[i]
	}
	result, _ := NewCPUTensorBySlice(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad)

	// 反向传播
	if result.RequiresGrad {
		result.Creator = &Function{
			Inputs: []*CPUTensor{a, b},
			Output: result,
			Backward: func(grad []float64) []float64 {
				gradA := make([]float64, len(a.Data))
				gradB := make([]float64, len(b.Data))
				copy(gradA, grad) // 减法对 x 的梯度是 1
				for i := range grad {
					gradB[i] = -grad[i] // 减法对 y 的梯度是 -1
				}
				return append(gradA, gradB...)
			},
		}
	}
	return result, nil
}

func Mul(a, b *CPUTensor) (*CPUTensor, error) {
	if len(a.Data) != len(b.Data) {
		return nil, errors.New("CPUTensor sizes must match")
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return nil, errors.New("tensor shapes must match")
		}
	}

	// 前向传播
	resultData := make([]float64, len(a.Data))
	for i := range resultData {
		resultData[i] = a.Data[i] * b.Data[i]
	}
	result, _ := NewCPUTensorBySlice(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad)

	// 反向传播
	if result.RequiresGrad {
		result.Creator = &Function{
			Inputs: []*CPUTensor{a, b},
			Output: result,
			Backward: func(grad []float64) []float64 {
				gradA := make([]float64, len(a.Data))
				gradB := make([]float64, len(b.Data))
				for i := range grad {
					gradA[i] = grad[i] * b.Data[i] // 乘法对 x 的梯度是 y
					gradB[i] = grad[i] * a.Data[i] // 乘法对 y 的梯度是 x
				}
				return append(gradA, gradB...)
			},
		}
	}
	return result, nil
}

func Div(a, b *CPUTensor) (*CPUTensor, error) {
	if len(a.Data) != len(b.Data) {
		return nil, errors.New("CPUTensor sizes must match")
	}
	for i := range a.Shape {
		if a.Shape[i] != b.Shape[i] {
			return nil, errors.New("tensor shapes must match")
		}
	}

	// 前向传播
	resultData := make([]float64, len(a.Data))
	for i := range resultData {
		if b.Data[i] == 0 {
			return nil, errors.New("division by zero")
		}
		resultData[i] = a.Data[i] / b.Data[i]
	}
	result, _ := NewCPUTensorBySlice(resultData, a.Shape, a.RequiresGrad || b.RequiresGrad)

	// 反向传播
	if result.RequiresGrad {
		result.Creator = &Function{
			Inputs: []*CPUTensor{a, b},
			Output: result,
			Backward: func(grad []float64) []float64 {
				gradA := make([]float64, len(a.Data))
				gradB := make([]float64, len(b.Data))
				for i := range grad {
					gradA[i] = grad[i] / b.Data[i]                            // 除法对 x 的梯度是 1/y
					gradB[i] = -grad[i] * a.Data[i] / (b.Data[i] * b.Data[i]) // 对 y 的梯度是 -x/y^2
				}
				return append(gradA, gradB...)
			},
		}
	}
	return result, nil
}

func Mean(input *CPUTensor) (*CPUTensor, error) {
	// 确保张量数据不为空
	if len(input.Data) == 0 {
		return nil, errors.New("tensor has no data")
	}

	// 计算前向传播：所有元素的平均值
	sum := 0.0
	for _, val := range input.Data {
		sum += val
	}
	meanValue := sum / float64(len(input.Data))

	// 创建输出张量
	output, _ := NewCPUTensorBySlice([]float64{meanValue}, []int{1}, input.RequiresGrad)

	// 构建反向传播规则
	if input.RequiresGrad {
		output.Creator = &Function{
			Inputs: []*CPUTensor{input},
			Output: output,
			Backward: func(grad []float64) []float64 {
				// 反向传播：梯度均匀分布到每个输入
				gradInput := make([]float64, len(input.Data))
				for i := range gradInput {
					gradInput[i] = grad[0] / float64(len(input.Data)) // 平均值的梯度传播
				}
				return gradInput
			},
		}
	}
	return output, nil
}
