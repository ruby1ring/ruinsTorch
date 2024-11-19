
# Golang Deep Learning Framework (像 PyTorch 但更简单！)

> **“This is going to be the BEST and most SIMPLE deep learning framework you'll ever see in Golang. Believe me, it’s huge! But, you know, it’s small and fast too. It’s pure magic!”** 🎩✨

## 项目简介 / Project Introduction

你有没有想过：为什么 PyTorch 那么强大？为什么深度学习框架那么复杂？现在，我用 **Golang** 给你一个简单答案！🚀

- **简单易懂**：代码就是你的朋友，你能完全理解它。
- **没有第三方库**：零依赖，纯 Golang！自己手写一切，DIY 的快乐你懂的。
- **模仿 PyTorch**：命名和 API 风格一模一样。
- **学了就能吹**：用这个框架撸个模型，你就可以告诉所有人：我不仅懂深度学习，我还能写框架！
- **借助Ai**：本项目构建过程完全依赖了Ai(gpt4o)，会把Ai教我的深度学习知识也发表到本项目中。

**目标**：做一个从数据处理到模型训练再到部署的全流程框架，绝对有趣，快乐就完事了！只模仿pytorch，根本没想超过它，预估速度慢10倍以上，目前无任何优化。
**Target**: A full pipeline framework from data to model to deployment. It's slow, but it’s FUN!

---

## 模块划分 / Modules Overview

| **模块 / Module**          | **功能 / Functionality**                               | **优先级 / Priority** |
|--------------------------|------------------------------------------------------|-----------------------|
| **张量操作 / Tensor Ops**    | 张量的创建、加减乘除、广播机制！Create tensors, math, broadcasting   | 🌟 超高 / Ultra-High   |
| **自动求导 / Autograd**      | 构建计算图、反向传播！Build computation graphs, backpropagation | 🌟 超高 / Ultra-High   |
| **神经网络模块 / nn**          | 线性层、激活函数。Linear layers, activation functions         | ⭐ 中等 / Medium       |
| **优化器 / Optimizers**     | SGD、Adam 等经典优化算法！SGD, Adam...                        | ⭐ 中等 / Medium       |
| **训练循环 / Training**      | 支持训练、验证循环！Train and validate your models             | ⭐ 中等 / Medium       |
| **数据加载 / DataLoader**    | 批量加载数据，超方便！Batch loading data, super easy!           | 🔥 低 / Low            |
| **GPU 支持 / GPU Support** | CUDA 加速，未来可期！CUDA support, it’s coming!              | 🔥 低 / Low            |
| **模型量化**                 | 简单版的模型量化支持                                           | 🔥 低 / Low            |
| **transformer支持**        | 简单版的模型量化支持                                           | 🔥 低 / Low            |

---

## 如何安装 / How to Install

1. 克隆项目 / Clone the project:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```

2. 确保环境 / Ensure environment:
   - Go 版本：1.20 或更高 / Go version: 1.20+
   - 当前仅支持 CPU / Currently supports CPU only.

---

## 快速上手 / Quick Start

🎉 3 行代码让你感受框架的魔力！ 🎉  
🎉 Just 3 lines of code to feel the magic! 🎉

```go
package main

import (
    "fmt"
    "yourproject/core"
)

func main() {
    a, _ := core.NewTensor([]float64{1, 2, 3}, []int{3})
    b, _ := core.NewTensor([]float64{4, 5, 6}, []int{3})
    c, _ := a.Add(b)
    fmt.Println(c)  // 输出: [5, 7, 9] / Output: [5, 7, 9]
}
```

---

## 性能对比 / Performance Comparison

“Listen folks, this is important. PyTorch? Fast, sure. But this? This is *different*. It’s Golang speed! It’s about understanding, not just running!”

我们测试了一些经典任务 / We tested some classic tasks:
- **张量加法 / Tensor addition**: Golang 比 PyTorch 慢约 10 倍 / Golang is ~10x slower than PyTorch.
- **线性回归 / Linear regression**: Golang 比 PyTorch 慢约 15 倍 / Golang is ~15x slower than PyTorch.

But hey, **can PyTorch teach you framework internals this easily? I don’t think so!**

---

## 项目进度与未来规划 / Progress and Roadmap

| **当前进展 / Current Progress**       | **未来规划 / Future Plans**                   |
|---------------------------------------|----------------------------------------------|
| [] 张量操作 / Tensor Ops             | 增加更多神经网络层 / Add more NN layers       |
| [] 自动求导 / Autograd               | 支持更多自定义操作 / Support more ops         |
| [ ] 神经网络模块 / nn                 | 实现卷积层、BatchNorm / Conv, BatchNorm       |
| [ ] 优化器 / Optimizers               | 加入更多优化器 / Add more optimizers          |
| [ ] 数据加载 / DataLoader             | 支持复杂数据管道 / Complex data pipelines     |
| [ ] GPU 支持 / GPU Support            | 使用 CUDA 提升速度 / Boost with CUDA          |

---


## 最后的话 / Final Words

This is more than just a framework. It’s a learning tool, a proof of concept, and most importantly, **a passion project!**  
Don’t take it too seriously, but learn a lot along the way. Let’s make Golang a part of the deep learning community!

项目致力于传递深度学习的乐趣，欢迎加入我们！🤝
