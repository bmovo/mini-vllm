# mini-vllm

一个从零开始的最小 decoder-only transformer + inference engine 实现，包含 KV cache、简化 continuous batching，以及 profiling / benchmark 脚本。

## 当前完成

- token embedding
- learned positional embedding
- masked self-attention
- FFN
- lm head
- `past_key_values` / `use_cache` 增量解码接口
- 简化 continuous batching 调度器
- 吞吐 / 延迟 benchmark 脚本
- `torch.profiler` profile 脚本

## 目录结构

```text
mini_vllm/
  __init__.py
  config.py
  engine.py
  model.py
examples/
  minimal_forward.py
benchmarks/
  benchmark_engine.py
  profile_engine.py
requirements.txt
```

## Ubuntu 远程环境建议

1. 根据 CPU / CUDA 环境安装合适的 PyTorch。
2. 在仓库根目录执行示例脚本、benchmark 或 profile 脚本。

如果远程机使用 CUDA，请优先按 PyTorch 官方方式安装对应版本，而不是直接依赖默认的 `pip install torch`。

这些脚本已经会自动把仓库根目录加入 `sys.path`，所以直接执行下面的命令即可，不需要额外 `pip install -e .`。

## 最小使用示例

```python
import torch

from mini_vllm import DecoderOnlyTransformer, ModelConfig

config = ModelConfig(
    vocab_size=32000,
    max_position_embeddings=2048,
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
    intermediate_size=2048,
)
model = DecoderOnlyTransformer(config)

input_ids = torch.randint(0, config.vocab_size, (4, 32))
outputs = model(input_ids, use_cache=True)

print(outputs.logits.shape)
print(len(outputs.past_key_values))
```

## 实现说明

- `DecoderOnlyTransformer` 使用 pre-norm block：`LN -> self-attention -> residual -> LN -> FFN -> residual`
- `attention_mask` 支持两种长度：
  - 当前 token 长度 `seq_len`
  - 带 cache 时的总长度 `past_length + seq_len`
- `past_key_values` 会在每层缓存 `[batch, heads, seq, head_dim]` 的 key/value，后续接增量解码与 continuous batching 时可以直接复用
- `MiniLLMEngine.step()` 先处理 active decode，再用剩余 batch 容量处理 waiting prefill，属于一个简化版 continuous batching 策略
- 为了把不同长度的请求合成一个 decode batch，engine 会把每个请求的 KV cache 左侧补零并右对齐，再用 `attention_mask` 与显式 `position_ids` 保证不同请求的位置语义正确

## 远程执行示例

```bash
python examples/minimal_forward.py
python benchmarks/benchmark_engine.py --device cuda --num-requests 64 --prompt-length 128 --max-new-tokens 64 --max-batch-size 16
python benchmarks/profile_engine.py --device cuda --steps 30 --row-limit 30
```
