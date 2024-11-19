package main

import (
	"fmt"
	core "ruinsTorch/core"
)

func main() {
	// 1. 定义模型
	model := &core.Model{}
	weight, _ := core.NewCPUTensorBySlice([]float64{0.5}, []int{1}, true)
	bias, _ := core.NewTensor([]float64{0.0}, []int{1}, true)
	model.AddParameter(weight)
	model.AddParameter(bias)

	// 2. 定义数据
	inputs, _ := core.NewTensor([]float64{1.0}, []int{1}, false)  // 输入
	targets, _ := core.NewTensor([]float64{2.0}, []int{1}, false) // 目标

	// 3. 定义优化器
	optimizer := core.NewOptimizer(model.Parameters, 0.1)

	// 4. 训练循环
	for epoch := 0; epoch < 100; epoch++ {
		// 前向传播
		output, _ := model.Forward(inputs)

		// 计算损失
		loss, _ := core.MSELoss(output, targets)

		// 反向传播
		loss.Backward()

		// 参数更新
		optimizer.Step()

		// 清空梯度
		for _, param := range model.Parameters {
			param.ZeroGrad()
		}

		// 打印结果
		fmt.Printf("Epoch %d: Loss = %v\n", epoch, loss.Data)
	}
}
