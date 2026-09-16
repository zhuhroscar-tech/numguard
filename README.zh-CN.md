# numguard

[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-555555?style=flat)](README.zh-CN.md)

用可复现的小型测试用例，对比朴素实现、稳定实现与独立高精度参考值。numguard 用于观察数值溢出、消减误差、整数溢出及算法状态处理错误，不是完整模型或 GPU 的性能测试工具。

## 主要功能

- 基于 NumPy 实现 softmax、log-sum-exp、归一化、归约、量化等 ML 和统计计算。
- 按适用范围比较 `float16`、`float32`、`float64`；整数类测试使用各自的评分规则。
- 逐例检查结果是否有限、误差是否在容限内，支持终端报告和 JSON。
- 自检模式同时核对预期的朴素实现失败、对照用例及稳定实现结果，便于发现回归。

## 安装

需要 Python 3.9+、NumPy 1.24+，不需要 PyTorch、TensorFlow 或 GPU。**不要运行 `pip install numguard`：PyPI 上的同名包属于另一个项目。** 请在独立环境中安装本仓库：

```bash
git clone https://github.com/zhuhroscar-tech/numguard.git
cd numguard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

也可从[发布页面](https://github.com/zhuhroscar-tech/numguard/releases)选择实际存在的 wheel，并核对对应版本的校验和。NumPy 依赖原生扩展，因此本项目不提供独立 zipapp。

## 快速使用

```bash
numguard --kernel variance --dtype float32
numguard --variant stable
numguard --json
numguard --check-naive-fails
```

不带参数时运行全部已配置用例和两种实现。朴素实现的失败是演示的一部分，因此普通报告可能返回非零退出码。`0` 表示本次检查通过，`1` 表示至少一项失败，`2` 表示用法错误。CI 应使用 `--check-naive-fails`，不要要求混合报告全部通过。

## 范围与开发

测试集经过人工挑选，并不穷尽所有输入；容限是实用阈值，不是严格误差界。结果仅针对仓库中的实现，不直接测试框架融合算子或 CUDA/MPS 行为，也不能证明模型或生产库整体正确。

详细说明见[推导文档](docs/numerical-stability.md)、[测试用例](src/numguard/fixtures.py)和[内核实现](src/numguard/kernels.py)。

```bash
python -m pytest -q
numguard --check-naive-fails
```

[MIT 许可证](LICENSE)
