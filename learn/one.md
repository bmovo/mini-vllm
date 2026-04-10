# one

这份文档先只回答一个问题：当前 `mini-vllm` 里每一个 class 到底在做什么。

你可以先把整个项目粗略分成两层：

- 模型层：负责“给定 token，怎么计算下一个 token 的概率”
- 引擎层：负责“多个请求什么时候一起算，KV cache 怎么保存和复用”

---

## 1. 整体结构

当前项目里主要有 9 个 class：

### 模型层

1. `ModelConfig`
2. `DecoderOnlyTransformerOutput`
3. `FeedForward`
4. `CausalSelfAttention`
5. `DecoderLayer`
6. `DecoderOnlyTransformer`

### 引擎层

1. `GenerationResult`
2. `_RequestState`
3. `MiniLLMEngine`

如果只看调用链，可以理解成：

`ModelConfig` -> `DecoderOnlyTransformer` -> `MiniLLMEngine`

也就是：

- `ModelConfig` 定义模型长什么样
- `DecoderOnlyTransformer` 负责真正的神经网络计算
- `MiniLLMEngine` 负责把多个请求组织起来，驱动模型做 prefill 和 decode

---

## 2. 每个 class 在做什么

## 2.1 `ModelConfig`

文件位置：`mini_vllm/config.py`

这个类是模型配置对象，也可以理解成“超参数说明书”。

它不参与任何实际计算，只负责保存这些信息：

- 词表大小 `vocab_size`
- 最大位置长度 `max_position_embeddings`
- 隐层维度 `hidden_size`
- decoder block 层数 `num_hidden_layers`
- attention 头数 `num_attention_heads`
- FFN 中间层大小 `intermediate_size`
- dropout 和 layer norm 的相关参数
- 是否让 `lm_head` 和 `token_embedding` 共享权重

它的 `__post_init__()` 里做了一件必要的检查：

- `hidden_size` 必须能被 `num_attention_heads` 整除

为什么要检查这个？

因为 attention 里会把 hidden dimension 切成多个 head，如果不能整除，就没法分成等宽的 heads。

一句话总结：

`ModelConfig` 负责定义模型结构参数，并在模型创建前做基础合法性校验。

---

## 2.2 `DecoderOnlyTransformerOutput`

文件位置：`mini_vllm/model.py`

这个类是模型前向计算的返回结果包装。

里面有三个字段：

- `logits`：每个位置对整个词表的预测分数
- `hidden_states`：最后一层的隐藏状态
- `past_key_values`：每层缓存下来的 KV cache

为什么要单独定义这个类，而不是直接返回一个 tuple？

因为 tuple 可读性差，后面引擎取值时不直观。定义成 dataclass 以后，调用方可以直接写：

```python
outputs.logits
outputs.past_key_values
```

一句话总结：

`DecoderOnlyTransformerOutput` 是模型输出的结构化容器，方便引擎读取 logits 和 KV cache。

---

## 2.3 `FeedForward`

文件位置：`mini_vllm/model.py`

这个类实现 Transformer block 里的 FFN 子层。

结构很简单：

1. `fc_in`：把 hidden size 投影到更大的 `intermediate_size`
2. `GELU` 激活函数
3. `fc_out`：再投影回 hidden size
4. dropout

FFN 的作用不是处理 token 与 token 的关系，而是对“每个 token 自己的表示”做更强的非线性变换。

可以这样理解：

- attention 负责“看别人”
- FFN 负责“加工自己”

一句话总结：

`FeedForward` 负责在 attention 之后进一步变换每个 token 的特征表示。

---

## 2.4 `CausalSelfAttention`

文件位置：`mini_vllm/model.py`

这个类是整个模型里最关键的一个子模块，因为它实现了：

- masked self-attention
- causal mask
- KV cache 拼接与返回

它的工作过程可以拆成几步：

### 第一步：把输入投影成 Q、K、V

输入 `hidden_states` 先分别经过：

- `q_proj`
- `k_proj`
- `v_proj`

得到 query、key、value。

然后 `_reshape_heads()` 会把张量变成多头 attention 需要的形状，大致是：

`[batch, seq, hidden] -> [batch, heads, seq, head_dim]`

### 第二步：如果有历史 cache，就把旧的 K/V 拼到前面

如果传进来了 `past_key_value`，说明当前不是第一次算，而是在增量解码。

这时会做：

- 当前步新算的 key 拼到旧 key 后面
- 当前步新算的 value 拼到旧 value 后面

这样新的 attention 就可以同时看到历史 token 和当前 token。

### 第三步：算 attention score

通过：

```python
query @ key.transpose(-1, -2)
```

得到每个 query 对所有 key 的相关性分数，再乘缩放因子 `self.scale`。

### 第四步：施加 causal mask

`_build_causal_mask()` 会生成一个“下三角可见”的 mask。

它的含义是：

- 当前 token 只能看见自己以及自己前面的 token
- 不能偷看未来 token

这就是 decoder-only 模型的“自回归”约束。

### 第五步：施加 attention mask

除了 causal mask，这里还支持额外的 `attention_mask`。

它主要用来处理：

- padding token 不该被看见
- continuous batching 时，不同请求对齐后补出来的无效位置不该被看见

### 第六步：softmax 后加权求和

对 attention score 做 softmax 得到注意力权重，再对 value 做加权求和，最后过 `out_proj`。

### 第七步：如果 `use_cache=True`，返回新的 KV cache

这里的 `present = (key, value)` 就是本层最新的 KV cache。

引擎在增量解码时，下一轮会把这个 cache 再传回来。

一句话总结：

`CausalSelfAttention` 负责完成 decoder-only 模型最核心的注意力计算，并且承担 KV cache 的读写接口。

---

## 2.5 `DecoderLayer`

文件位置：`mini_vllm/model.py`

这个类表示“一个完整的 decoder block”。

内部包含：

- `input_layernorm`
- `self_attn`
- `post_attention_layernorm`
- `mlp`

它的 forward 逻辑是：

1. 对输入做 layer norm
2. 进 self-attention
3. 做第一次 residual connection
4. 再做 layer norm
5. 进 FFN
6. 做第二次 residual connection

这就是一个典型的 pre-norm Transformer block。

为什么叫 pre-norm？

因为 layer norm 放在 attention 和 FFN 之前，而不是之后。

一句话总结：

`DecoderLayer` 是一个标准 decoder block，把 attention 和 FFN 串起来，并加上 residual 和 layer norm。

---

## 2.6 `DecoderOnlyTransformer`

文件位置：`mini_vllm/model.py`

这个类是整个最小语言模型本体。

你可以把它理解成“把所有零件组装起来的总模型”。

它内部主要包含：

- `token_embedding`
- `position_embedding`
- 多层 `DecoderLayer`
- `final_layernorm`
- `lm_head`

它的 forward 大致在做下面这些事。

### 1. 处理输入与 cache 长度

它先读取：

- `input_ids`
- `attention_mask`
- `position_ids`
- `past_key_values`

如果带了 `past_key_values`，说明当前是在做增量解码，那么 `past_length` 就不为 0。

### 2. 构造位置编码

如果外部没传 `position_ids`，它会自动生成：

- 普通前向：`0, 1, 2, ..., seq_len - 1`
- 带 cache 的增量解码：从 `past_length` 开始继续编号

这一步保证位置 embedding 能对应到正确的 token 位置。

### 3. 处理 attention mask

它支持两种 mask 长度：

- 只给当前输入长度 `seq_len`
- 直接给总长度 `past_length + seq_len`

这让它既能支持普通训练式前向，也能支持带 cache 的推理前向。

### 4. token embedding + position embedding

输入 token 会先经过词嵌入，再加上位置嵌入。

这一步得到每个 token 的初始表示。

### 5. 逐层ผ่าน decoder blocks

所有 `DecoderLayer` 会依次执行。

每一层都可能读取自己的旧 cache，并生成自己的新 cache。

### 6. 输出 logits

最后经过：

- `final_layernorm`
- `lm_head`

得到每个 token 对词表的预测分数 `logits`。

如果 `tie_word_embeddings=True`，那么 `lm_head.weight` 会直接和 `token_embedding.weight` 共用参数。

一句话总结：

`DecoderOnlyTransformer` 是模型总装层，负责把 embedding、stacked decoder layers、lm head 和 KV cache 接口整合起来。

---

## 2.7 `GenerationResult`

文件位置：`mini_vllm/engine.py`

这个类表示“一个请求最终生成完成之后的结果”。

它保存的信息包括：

- `request_id`
- 原始 prompt token
- 生成出的 output token
- 请求创建时间
- 首 token 时间
- 完成时间

它还提供两个属性：

- `ttft_seconds`
- `latency_seconds`

其中：

- TTFT = Time To First Token
- latency = 整个请求从提交到完成的总耗时

这两个指标是 benchmark 分析里最核心的延迟指标。

一句话总结：

`GenerationResult` 是请求完成后的结果对象，重点用于吞吐和延迟分析。

---

## 2.8 `_RequestState`

文件位置：`mini_vllm/engine.py`

这个类是引擎内部维护的“请求运行时状态”。

注意前面的下划线 `_`，表示它是内部类，不打算直接给外部用户使用。

它保存的不是“最终结果”，而是“一个请求在推理过程中当前进展到哪了”。

主要字段包括：

- `prompt_token_ids`
- `output_token_ids`
- `past_key_values`
- `pending_input_id`
- `first_token_at`
- `finished_at`

其中两个字段很关键：

### `past_key_values`

表示这个请求当前已经缓存下来的历史 KV。

下次 decode 时，不需要重新算全部前缀，只要喂一个新 token，再把旧 KV 带上就可以了。

### `pending_input_id`

表示“下一轮 decode 要喂给模型的 token”。

例如：

1. prefill 之后，模型会产出第一个生成 token
2. 这个 token 会被存进 `pending_input_id`
3. 下一次 decode，就把这个 token 当成模型输入

所以 `_RequestState` 实际上承担了“请求状态机”的作用。

它还有两个辅助接口：

- `cached_tokens`：当前 cache 里已经有多少 token
- `to_result()`：把内部状态转换成最终的 `GenerationResult`

一句话总结：

`_RequestState` 是单个请求在推理期间的内部状态记录器，连接 prefill、decode、KV cache 和最终结果。

---

## 2.9 `MiniLLMEngine`

文件位置：`mini_vllm/engine.py`

这个类是当前项目里“最像 vLLM 引擎”的部分。

它不负责神经网络数学本身，而是负责：

- 接收请求
- 管理 waiting / active / finished 三类请求
- 做 prefill
- 做单步 decode
- 组织简化 continuous batching
- 合并和拆分不同请求的 KV cache

你可以把它理解成：模型外面的“调度层”。

### 它维护了三种请求集合

- `_waiting`：刚提交，还没开始 prefill 的请求
- `_active`：已经做过 prefill，正在逐 token decode 的请求
- `_finished`：已经结束的请求

### `submit()`

作用是把一个新请求塞进 `_waiting` 队列。

它会创建一个 `_RequestState`，但这时还没有跑模型。

### `step()`

这是引擎最重要的方法。

每执行一次 `step()`，引擎会尝试推进一轮调度：

1. 先从 `_active` 里取出一批请求做 decode
2. 再用剩余 batch 容量，从 `_waiting` 里取请求做 prefill

这就是这里实现的“简化 continuous batching”。

它不像最完整的 vLLM 那样复杂，但已经具备一个核心思想：

- 不是等所有请求同时开始
- 而是让新请求可以不断加入已有的运行流程

### `_prefill_batch()`

这个方法处理“第一次完整看到 prompt”的阶段。

它会：

1. 把多个 prompt pad 成一个 batch
2. 调用模型 `forward(..., use_cache=True)`
3. 得到每个请求各层的 KV cache
4. 从 prompt 最后一个位置的 logits 里取出下一个 token
5. 把请求转入 active 状态

这里的重点是：

- prefill 计算量大，因为要把整段 prompt 全算一遍
- 但算完后，后面 decode 就可以复用 cache

### `_decode_batch()`

这个方法处理“已经有 cache 后，一次只解一个 token”的阶段。

它会：

1. 取出每个 active 请求的 `pending_input_id`
2. 把这些 token 合并成一个 batch
3. 把不同请求的 `past_key_values` 合并起来
4. 给模型传入 `past_key_values`、`attention_mask`、`position_ids`
5. 得到新的 logits 和新的 cache
6. 为每个请求记录新 token，并决定是否结束

这是 KV cache 真正开始发挥作用的地方。

### `_merge_past_key_values()`

因为不同请求的历史长度可能不同，所以没法直接 `torch.cat`。

这个方法会：

- 找到当前 batch 中最大的历史长度
- 把每个请求的 KV cache 左侧补零
- 右对齐到相同长度
- 再拼成一个 batch

为什么是左侧补零？

因为我们要保证“真正的历史 token”在右边对齐，这样配合显式 `position_ids` 和 `attention_mask`，不同请求的位置语义才不会乱掉。

### `_split_past_key_values()`

模型算完之后，返回的是 batched cache。

但引擎内部需要把它重新拆回“每个请求自己的 cache”。

这个方法就负责从 batched KV 里，按每个请求的有效长度切出来，重新放回各自的 `_RequestState`。

### `_record_token()`

这个方法在每生成出一个 token 后更新请求状态。

它会：

- 把 token 加进 `output_token_ids`
- 更新 `pending_input_id`
- 记录首 token 时间
- 判断是不是该结束

如果没结束，就放回 `_active`；
如果结束了，就转成 `GenerationResult` 放进 `_finished`。

一句话总结：

`MiniLLMEngine` 是整个项目的调度中枢，负责让模型真正以“推理服务”的方式运行起来。

---

## 3. 这些 class 怎么串起来工作

可以用一个请求的生命周期来理解。

### 阶段 1：创建模型

先用 `ModelConfig` 定义模型参数，然后创建 `DecoderOnlyTransformer`。

这时模型内部会组装：

- embedding
- 多层 `DecoderLayer`
- `lm_head`

### 阶段 2：创建引擎

再把模型交给 `MiniLLMEngine`。

这时引擎开始具备：

- 接收请求
- 组织 batch
- 保存 cache
- 统计延迟

### 阶段 3：提交请求

调用 `engine.submit()` 时，会创建一个 `_RequestState` 并放入 waiting 队列。

### 阶段 4：prefill

引擎执行 `_prefill_batch()`：

- prompt 进入 `DecoderOnlyTransformer`
- 每层 `DecoderLayer` 内部都会调用 `CausalSelfAttention`
- `CausalSelfAttention` 生成每层的 KV cache
- 最终 cache 被保存进 `_RequestState`

### 阶段 5：decode

下一步开始只输入一个 token：

- 引擎从 `_RequestState` 里拿出 `pending_input_id`
- 同时把 `past_key_values` 交给模型
- attention 里把旧 KV 和新 KV 拼起来
- 模型输出新的 token 和新的 cache

### 阶段 6：请求结束

当生成 token 达到上限，或者碰到 `eos_token_id`，引擎把 `_RequestState` 转成 `GenerationResult`。

所以这个项目的基本关系是：

- `DecoderOnlyTransformer` 负责“算”
- `MiniLLMEngine` 负责“调度”
- `_RequestState` 负责“记住每个请求当前到哪了”
- `GenerationResult` 负责“把最终结果交出来”

---

## 4. 你现在应该抓住的重点

如果你是第一次看这种代码，我建议先只抓住下面 4 个核心点：

1. `DecoderOnlyTransformer` 是模型本体
2. `CausalSelfAttention` 是最核心的计算模块
3. `_RequestState` 是单个请求的运行时状态
4. `MiniLLMEngine` 是把模型变成“可服务多个请求”的调度器

只要这四点先清楚，后面再看：

- KV cache 为什么能加速
- 为什么 prefill 和 decode 分开
- continuous batching 为什么能提升吞吐

就会顺很多。

---

## 5. 下一份文档适合写什么

下一步我建议把 `learn/two.md` 专门写成：

`一次请求从 submit 到生成完成，到底发生了哪些调用`

那一份会更偏“时间顺序”，也就是：

1. 请求进入 waiting
2. prefill 发生了什么
3. decode 发生了什么
4. KV cache 在哪里生成、在哪里复用
5. 请求什么时候结束

如果你要，我下一轮就按这个结构继续写。
