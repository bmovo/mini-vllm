# one_profiler

这份文档专门解释一件事：

当你运行 `torch.profiler` 之后，看到的 `aten::mm`、`aten::cat`、`aten::bmm` 这些东西，到底是什么意思。

这一份不讲新的代码版本，也不更新 `two.md`。

你可以把它理解成：

“读懂 profiler 输出的入门说明书”

---

## 1. 先说最重要的一句话

`aten::xxx` 不是你自己写的函数名。

它们是 PyTorch 底层实际执行的算子名字。

也就是说：

- 你写的是 `Linear`、`softmax`、`cat`
- profiler 看到的是底层真正被调用的算子
- 这些底层算子一般会显示成 `aten::xxx`

所以当你看到：

- `aten::mm`
- `aten::cat`
- `aten::bmm`
- `aten::add`

它们说的其实是：

“PyTorch 在底层实际干了这些张量运算”

---

## 2. `aten` 是什么

`aten` 可以先简单理解成：

PyTorch 张量运算的底层算子库名字。

你现在不需要深究它的历史。

只需要知道：

- `aten::mm` = PyTorch 底层做了一次矩阵乘法
- `aten::cat` = PyTorch 底层做了一次拼接
- `aten::zeros` = PyTorch 底层创建了一个全 0 张量

所以：

`aten::xxx` 本质上就是“底层操作名”。

---

## 3. 你的 profiler 表每一列是什么意思

你的输出大概长这样：

```text
Name
Self CPU %
Self CPU
CPU total %
CPU total
CPU time avg
# of Calls
```

下面逐个解释。

---

## 3.1 `Name`

表示这是什么操作。

例如：

- `aten::mm`
- `aten::cat`
- `aten::gelu`
- `engine_step`

注意：

`engine_step` 不是 PyTorch 底层算子，而是你在代码里自己用 `record_function("engine_step")` 标出来的一个逻辑阶段。

它在 [profile_engine.py](d:/0-paper/mini-vllm/benchmarks/profile_engine.py#L77) 这里。

所以：

- `aten::xxx` 是底层操作
- `engine_step` 是你人为标记的逻辑区间

---

## 3.2 `Self CPU %`

表示：

这个操作自己本身花了多少 CPU 时间，占总采样 CPU 时间的百分比。

关键是“自己本身”。

它不算这个操作内部再调用的别的子操作。

你可以把它理解成：

“纯粹这一步自己就花了多少时间”

---

## 3.3 `Self CPU`

和 `Self CPU %` 是一个意思，只不过这里显示的是绝对时间，不是百分比。

例如：

`aten::mm 232.771ms`

表示在整个 profiler 记录里，`aten::mm` 自己本身一共花了约 232 毫秒。

---

## 3.4 `CPU total %`

表示：

这个操作连同它内部触发的子操作，加起来一共花了多少 CPU 时间，占总时间的百分比。

所以：

- `Self CPU %` 更像“自己花了多少”
- `CPU total %` 更像“自己加孩子一共花了多少”

很多底层算子本身没有复杂的子调用，所以这两列可能差不多。

但对一些更高层的逻辑区间，比如 `engine_step`，`CPU total %` 会很大，因为它包住了很多底层操作。

---

## 3.5 `CPU total`

就是 `CPU total %` 对应的绝对时间版本。

例如你的输出里：

`engine_step ... CPU total 337.896ms`

意思就是：

整个 `engine_step` 逻辑区间，总共大约花了 337.896 毫秒。

---

## 3.6 `CPU time avg`

表示平均每调用一次，花多长时间。

例如：

```text
aten::mm ... CPU time avg 314.633us ... # of Calls 740
```

意思是：

- `aten::mm` 一共被调用了 740 次
- 每次平均大约花 314.633 微秒

这个指标很适合用来看：

- 是“少量重操作”慢
- 还是“大量小操作”堆起来慢

---

## 3.7 `# of Calls`

表示这个操作在采样区间内总共被调用了多少次。

它非常重要，因为有些操作单次不贵，但调用次数多到离谱，也会拖慢整体性能。

比如：

- `aten::copy_`
- `aten::slice`
- `aten::zeros`

这类操作经常单次不大，但次数很多。

---

## 4. 先读懂你这次输出里最重要的几个名字

你这次输出里最关键的是这些：

- `engine_step`
- `aten::mm`
- `aten::matmul`
- `aten::bmm`
- `aten::cat`
- `aten::gelu`
- `aten::argmax`
- `aten::native_layer_norm`
- `aten::masked_fill_`
- `aten::zeros`
- `aten::copy_`

下面逐个讲。

---

## 4.1 `engine_step`

这个不是底层算子，而是你代码里手动标出来的阶段名。

在 [profile_engine.py](d:/0-paper/mini-vllm/benchmarks/profile_engine.py#L77)：

```python
with record_function("engine_step"):
    engine.step()
```

所以它表示：

“引擎推进一步调度”这一整段逻辑

这一段里面包含了很多事情：

- 取 waiting / active 请求
- 做 prefill 或 decode
- 调模型 forward
- 更新 KV cache
- 记录生成 token

所以 `engine_step` 是一个总框。

---

## 4.2 `aten::mm`

`mm` 是 matrix multiply，矩阵乘法。

这是最重要的一个指标。

在 Transformer 里，大量时间通常都会花在矩阵乘法上。

为什么？

因为这些操作底层都会变成矩阵乘法：

- `q_proj`
- `k_proj`
- `v_proj`
- `out_proj`
- FFN 的两层线性层
- `lm_head`

你可以在 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L43) 到 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L46) 看到 attention 的线性层，在 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L24) 和 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L26) 看到 FFN 的线性层。

所以当 `aten::mm` 很高时，最常见的解释不是“哪里出 bug 了”，而是：

“模型主体计算本来就主要靠矩阵乘法”

你这次 `aten::mm` 占了大约 68.89%，这是很正常的。

---

## 4.3 `aten::matmul`

`matmul` 是更通用的矩阵乘法接口。

你可以先这样理解：

- `matmul` 像统一入口
- `mm` 是二维矩阵乘法
- `bmm` 是批量矩阵乘法

在你的 attention 代码里，像下面这种：

```python
torch.matmul(query, key.transpose(-1, -2))
torch.matmul(attn_probs, value)
```

就会对应到 `aten::matmul`，位置在 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L73) 和 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L97)。

很多时候 profiler 里你会同时看到：

- `aten::matmul`
- `aten::mm`
- `aten::bmm`

这是因为高层的 `matmul` 在不同张量形状下，底层可能会进一步走到 `mm` 或 `bmm`。

所以不要把它们理解成完全独立的三件事。

---

## 4.4 `aten::bmm`

`bmm` 是 batch matrix multiply，批量矩阵乘法。

当你不是只做一对矩阵乘法，而是一次对一整批矩阵做乘法时，就常会看到它。

在 attention 里很常见。

因为 attention 往往是：

- batch 维度
- head 维度
- seq 维度

这些组合起来后，经常会触发批量矩阵乘法。

你可以把 `aten::bmm` 理解成：

“一次性对很多组矩阵做乘法”

---

## 4.5 `aten::cat`

`cat` 是 concatenate，拼接。

这在你当前这个 mini 引擎里特别值得关注。

因为当前实现里，KV cache 的组织方式比较直接，会频繁做拼接。

例如在 attention 里，旧的 KV 和新的 KV 会拼起来，位置在 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L67) 和 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L68)。

此外在 engine 做 batch 组装时，也会把多个请求的 KV cache 拼起来，位置在 [engine.py](d:/0-paper/mini-vllm/mini_vllm/engine.py#L319)。

所以你这次看到：

`aten::cat` 大约 8.19%

这恰好说明了一件事：

当前版本除了模型本体计算，KV cache 的搬运和拼接也已经有明显成本了。

这也是为什么后面真正像 vLLM 一样做 paged KV cache 时，会更关心如何减少这种拼接开销。

---

## 4.6 `aten::gelu`

这是 FFN 里的激活函数。

对应代码在 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L25) 和 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L31)。

它的含义非常直接：

模型在执行 FFN 的非线性激活。

如果你看到它出现，不说明有问题，只说明 FFN 确实跑到了。

---

## 4.7 `aten::argmax`

`argmax` 表示从一堆分数里挑出最大的那个位置。

当前项目里，它对应“选下一个 token”的最简策略。

也就是：

- 不做采样
- 直接选 logits 里最大的 token

对应代码在 [engine.py](d:/0-paper/mini-vllm/mini_vllm/engine.py#L171) 和 [engine.py](d:/0-paper/mini-vllm/mini_vllm/engine.py#L209)。

所以 `aten::argmax` 出现说明：

生成步骤确实发生了，而不只是单纯前向。

---

## 4.8 `aten::native_layer_norm`

这是 layer normalization。

对应代码在 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L122) 、[model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L124) 和 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L163)。

它的作用你现在可以先不深挖。

现阶段只要知道：

- 这是 Transformer block 的标准组成部分
- attention 前和 FFN 前都要做
- 最后输出前也要做一次

出现它是完全正常的。

---

## 4.9 `aten::masked_fill_`

这个操作表示：

“把 mask 不允许的位置，用某个值填掉。”

在 attention 里很关键。

因为有些位置不能看：

- 未来 token 不能看
- padding 位置不能看
- batch 对齐时补出来的无效位置不能看

所以代码里会把这些位置的 attention score 填成一个极小值，让 softmax 后几乎不被选中。

对应位置在 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L80) 和 [model.py](d:/0-paper/mini-vllm/mini_vllm/model.py#L91)。

---

## 4.10 `aten::zeros`

这个表示创建一个全 0 张量。

在你当前实现里，比较常见于：

- 构造 batch 输入
- 构造 attention mask
- 给不同请求的 KV cache 做 padding

例如 [engine.py](d:/0-paper/mini-vllm/mini_vllm/engine.py#L237)、[engine.py](d:/0-paper/mini-vllm/mini_vllm/engine.py#L243)、[engine.py](d:/0-paper/mini-vllm/mini_vllm/engine.py#L264)、[engine.py](d:/0-paper/mini-vllm/mini_vllm/engine.py#L297)。

如果 `aten::zeros` 很高，通常说明：

- 临时张量分配比较多
- 有不少对齐、padding 或中间缓存构造

---

## 4.11 `aten::copy_`

这个表示数据拷贝。

当前实现里，它通常来自：

- 把 prompt 拷进 batch tensor
- 把小张量拷进更大的 padding 张量
- 做 cache 重组时的数据搬运

如果 `copy_` 调用次数很多，说明你的实现里有不少“张量搬家”操作。

这类操作不一定单次贵，但次数多了会有累计成本。

---

## 5. 结合你这次结果，应该怎么读

你这次最值得抓的不是所有行，而是前三层意思。

---

## 5.1 第一层：模型主体算力主要花在矩阵乘法

你这次：

- `aten::mm` 很高
- `aten::matmul` 很高
- `aten::bmm` 也出现了

这说明模型主要时间确实花在 attention 和线性层上。

这是正常现象。

换句话说：

你的 profiler 没有显示出“模型主体根本没跑起来”这种问题。

---

## 5.2 第二层：当前 KV cache 组织方式有额外搬运成本

你这次：

- `aten::cat` 不低
- `aten::copy_` 调用很多
- `aten::zeros` 调用很多

这说明除了真正的神经网络计算之外，还有不少时间花在：

- 拼接
- padding
- 拷贝
- 中间张量构造

这和你当前这个版本的设计是吻合的。

因为我们现在实现的是“简化 continuous batching”，不是页式 KV cache。

所以这里看到更多 `cat / zeros / copy_`，并不奇怪。

---

## 5.3 第三层：引擎逻辑确实跑到了

你看到了：

- `engine_step`
- `argmax`

这说明不是只有模型在空跑，而是：

- 引擎真的推进了 step
- 请求真的在逐 token 生成
- 最终 token 真的被选出来了

这对当前版本来说，是一个很重要的正确性信号。

---

## 6. 哪些指标高是“正常的”，哪些高了需要警惕

## 正常偏高

- `aten::mm`
- `aten::matmul`
- `aten::bmm`
- `aten::native_layer_norm`
- `aten::gelu`

因为这些本来就是模型主体计算的一部分。

## 需要特别留意

- `aten::cat`
- `aten::copy_`
- `aten::zeros`
- `aten::slice`
- `aten::clone`

因为这些更像“为了把数据组织好而付出的额外成本”。

它们不是一定有问题。

但如果后面占比越来越高，通常说明：

- batch 组织方式不够高效
- KV cache 管理方式有额外开销
- 临时张量创建太多

---

## 7. 你现在可以怎么用这份 profiler 结果

现阶段先不要试图把每一行都背下来。

你只需要先用它回答三个问题：

### 问题 1

模型主体真的跑起来了吗？

看：

- `aten::mm`
- `aten::matmul`
- `aten::bmm`

### 问题 2

引擎真的在生成，而不是只做一次前向吗？

看：

- `engine_step`
- `aten::argmax`

### 问题 3

当前实现里，额外的调度和 cache 搬运成本大不大？

看：

- `aten::cat`
- `aten::copy_`
- `aten::zeros`

---

## 8. 这一份你最该记住什么

如果这份文档最后你只能带走 4 句话，那就记这 4 句：

1. `aten::xxx` 是 PyTorch 底层实际执行的算子名
2. `mm / matmul / bmm` 主要代表模型主体的矩阵计算
3. `cat / copy_ / zeros` 主要代表数据搬运、拼接和中间张量构造成本
4. `engine_step` 不是底层算子，而是我们自己标记出来的一整个引擎步骤

---

## 9. 下一步适合补什么

如果你接下来还想继续把 profiler 看懂，我建议下一份不要立刻进 `two.md`，而是继续加一份平行文档，例如：

- `one_benchmark.md`

专门解释：

- `ttft_p50_ms`
- `ttft_p95_ms`
- `latency_p50_ms`
- `prompt_throughput_tok_per_s`
- `decode_throughput_tok_per_s`

也就是把 benchmark 输出也像这份 profiler 文档一样拆开讲清楚。
