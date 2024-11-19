package core

import (
	"errors"
	"fmt"
)

// Tensor 定义了张量的基本操作接口
type ITensor interface {
	// 基础属性
	Shape() []int   // 获取张量的形状
	Dtype() string  // 获取数据类型
	Device() string // 获取设备类型
	Size() int      // 获取张量的总元素数

	// 数据访问
	Get(indices []int) (float64, error)     // 获取指定位置的值
	Set(indices []int, value float64) error // 设置指定位置的值

	// 形状操作
	Reshape(newShape []int) (ITensor, error)    // 重塑张量形状
	Transpose(dim1, dim2 int) (ITensor, error)  // 交换两个维度
	Slice(dim, start, end int) (ITensor, error) // 切片

	// 打印与调试
	Print() // 打印张量内容
}

type CPUTensor struct {
	Data         []float64 // 数据存储
	Shape        []int     // 张量形状
	stride       []int     // 步长，用于索引计算
	dtype        string    // 数据类型
	device       string    // 设备类型
	Grad         []float64 // 存储张量的梯度
	RequiresGrad bool      // 是否需要计算梯度
	Creator      *Function // 生成该张量的操作
}

// Function 定义
type Function struct {
	Inputs   []*CPUTensor                   // 输入张量
	Output   *CPUTensor                     // 输出张量
	Backward func(grad []float64) []float64 // 反向传播逻辑
}

func NewCPUTensorBySlice(data []float64, shape []int, requiresGrad bool) (*CPUTensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if len(data) != size {
		return nil, errors.New("Data size does not match Shape")
	}

	return &CPUTensor{
		Data:         data,
		Shape:        shape,
		stride:       computeStride(shape),
		dtype:        "float64", // 默认类型
		device:       "cpu",     // 默认设备
		RequiresGrad: requiresGrad,
		Creator:      nil,
	}, nil
}

// computeStride 计算张量的步长
func computeStride(shape []int) []int {
	stride := make([]int, len(shape))
	size := 1
	for i := len(shape) - 1; i >= 0; i-- {
		stride[i] = size
		size *= shape[i]
	}
	return stride
}

func NewOnesCPUTensor(shape []int, requiresGrad bool) (*CPUTensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	for i := range len(data) {
		data[i] = 1.0
	}
	return &CPUTensor{
		Data:         data,
		Shape:        shape,
		stride:       computeStride(shape),
		dtype:        "float64", // 默认类型
		device:       "cpu",     // 默认设备
		RequiresGrad: requiresGrad,
		Creator:      nil,
	}, nil
}

// 清空梯度
func (t *CPUTensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

func NewZerosCPUTensor(shape []int) (*CPUTensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float64, size)
	return &CPUTensor{
		Data:   data,
		Shape:  shape,
		stride: computeStride(shape),
		dtype:  "float64", // 默认类型
		device: "cpu",     // 默认设备
	}, nil
}

// Dtype 返回数据类型
func (t *CPUTensor) Dtype() string {
	return t.dtype
}

// Device 返回设备类型
func (t *CPUTensor) Device() string {
	return t.device
}

// Size 返回总元素数量
func (t *CPUTensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

func (t *CPUTensor) Print() {
	fmt.Printf("CPUTensor(Data=%v, Shape=%v, dtype=%s, device=%s)\n", t.Data, t.Shape, t.dtype, t.device)
}
