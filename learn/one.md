# one

这份文档重新来一版。

目标只有一个：

不要一上来追求“术语全懂”，而是先搞明白这个项目里每个 class 扮演什么角色。

如果你刚接触 LLM 推理代码，最容易卡住的点是：

- 代码里每个名字都像懂了，但连起来不知道在干嘛
- 每一句都对，但读完脑子里没有画面

所以这一版我会尽量只做三件事：

1. 先把几个必须知道的词讲清楚
2. 再讲每个 class 像什么
3. 最后讲这些 class 是怎么串起来工作的

---

## 1. 先别急着看 class，先懂 7 个词

如果下面这几个词不顺，后面所有类都会看着费劲。

## 1.1 token

你可以先把 token 理解成“文本被切出来的小块”。

例如一句话不会直接送进模型，而是先变成 token，再变成数字 id。

所以模型真正吃进去的，不是字符串，而是很多整数。

---

## 1.2 embedding

embedding 可以先理解成：

“把一个整数 token id 变成一个向量。”

因为神经网络不能直接理解 `12`、`304`、`998` 这种 token id 的语义，所以要先把它们映射成向量。

你现在不用关心向量内部长什么样，只要知道：

- 输入一串 token id
- 输出一串向量

---

## 1.3 前向计算

前向计算就是：

把输入送进模型，一层一层算下去，最后得到输出。

在这个项目里，你可以先把它理解成：

- 输入：一串 token id
- 输出：模型认为“下一个 token 应该是什么”的分数

---

## 1.4 logits

logits 是模型最后输出的一堆分数。

它还不是最终选中的 token，只是表示：

- 词表里的每个 token，有多像下一个 token

分数越大，说明模型越倾向选它。

你可以先把 logits 理解成“候选答案分数表”。

---

## 1.5 attention

attention 最粗糙、但最有用的理解是：

“当前 token 在更新自己时，会参考别的 token。”

比如一句话里，一个词想知道自己该怎么理解，不能只看自己，还得参考前面的词。

所以 attention 做的事可以先理解成：

- 当前 token 去看看别的 token
- 判断哪些 token 更重要
- 把它们的信息拿过来一点

这里的“看”不是真的看，而是“给别的 token 分配权重”。

---

## 1.6 KV cache

KV cache 可以先理解成：

“把已经算过的历史信息存起来，下次别再重复算。”

如果模型已经看过前面 100 个 token，再生成第 101 个 token 时，没必要把前 100 个全重新算一遍。

所以会把历史信息缓存起来。

这就是 KV cache 最重要的意义：

- 少做重复计算
- 让生成更快

---

## 1.7 engine

engine 可以先理解成：

“模型外面的调度器。”

模型本体只负责算。

engine 负责：

- 请求什么时候进来
- 哪些请求一起组成 batch
- 哪些请求已经生成了一半
- 哪些请求该结束了

如果把模型看成发动机，engine 就像司机和调度系统。

---

## 2. 先从更高一层看：这个项目其实只有两大部分

你可以先把当前项目拆成两块：

### 第一块：模型本体

这一块负责：

“给我一些 token，我来算下一个 token 的分数。”

对应的主要类是：

- `ModelConfig`
- `FeedForward`
- `CausalSelfAttention`
- `DecoderLayer`
- `DecoderOnlyTransformer`
- `DecoderOnlyTransformerOutput`

### 第二块：推理引擎

这一块负责：

“如果同时有很多请求进来，我该怎么组织它们去调用模型，才能更高效？”

对应的主要类是：

- `_RequestState`
- `GenerationResult`
- `MiniLLMEngine`

所以先记住一句话：

- 模型负责“怎么算”
- 引擎负责“什么时候算、哪些请求一起算”

---

## 3. 每个 class 到底像什么

这一节不追求完整，只追求脑子里先有画面。

---

## 3.1 `ModelConfig`

文件位置：`mini_vllm/config.py`

你可以把它理解成：

“模型的参数说明书。”

它里面写的是：

- 词表有多大
- 最长支持多少位置
- 隐藏层维度多大
- 一共有几层
- attention 有几个头
- FFN 中间层多大

它自己不负责计算。

它只是告诉模型：

“你应该长成什么样。”

### 你现在只需要记住

`ModelConfig` 不是模型本身，它只是模型的配置表。

---

## 3.2 `FeedForward`

文件位置：`mini_vllm/model.py`

你可以把它理解成：

“对每个 token 做一次单独加工的小模块。”

attention 负责让 token 参考别的 token。

`FeedForward` 负责在参考完之后，再把这个 token 自己的表示加工一下。

所以它更像：

- attention：交流信息
- FFN：自己消化信息

### 你现在只需要记住

`FeedForward` 不负责 token 和 token 之间的关系，它负责“单个 token 自己的特征变换”。

---

## 3.3 `CausalSelfAttention`

文件位置：`mini_vllm/model.py`

你可以把它理解成：

“让当前 token 去参考前面 token 的模块。”

这里最关键的是 `causal`。

`causal` 的意思是：

- 只能看当前位置以及前面
- 不能看未来

为什么？

因为语言模型是一个字一个字往后生成的。

如果当前 token 能偷看后面的 token，就作弊了。

所以这个类的核心工作是：

1. 让 token 之间互相参考
2. 但只允许看左边，不能看右边
3. 顺便支持 KV cache，把历史信息保存下来

### 你现在只需要记住

`CausalSelfAttention` 是模型最核心的部分，它负责“看上下文”，并且只看过去，不看未来。

---

## 3.4 `DecoderLayer`

文件位置：`mini_vllm/model.py`

你可以把它理解成：

“模型里的一层标准积木。”

这一层积木里主要装了两样东西：

- `CausalSelfAttention`
- `FeedForward`

所以一个 `DecoderLayer` 干的事就是：

1. 先让 token 看看上下文
2. 再把每个 token 自己加工一下

一个语言模型不会只有一层这样的积木，而是会堆很多层。

### 你现在只需要记住

`DecoderLayer` 就是一层 Transformer block，是模型里的重复单元。

---

## 3.5 `DecoderOnlyTransformer`

文件位置：`mini_vllm/model.py`

你可以把它理解成：

“整个语言模型本体。”

前面的 `FeedForward`、`CausalSelfAttention`、`DecoderLayer` 都只是零件。

这个类才是把所有零件组装起来的总机器。

它内部做的事情可以粗略理解成：

1. 把 token id 变成向量
2. 加上位置信息
3. 经过很多层 `DecoderLayer`
4. 最后输出 logits

它还负责处理 KV cache，也就是：

- 如果是第一次看 prompt，就正常算
- 如果前面已经算过了，就复用历史 cache

### 你现在只需要记住

`DecoderOnlyTransformer` 是“总模型”，其余模型类大多都是它内部的零件。

---

## 3.6 `DecoderOnlyTransformerOutput`

文件位置：`mini_vllm/model.py`

你可以把它理解成：

“模型输出结果的袋子。”

模型一次前向计算之后，会得到一些结果。

这个类就是专门拿来装这些结果的，主要包括：

- `logits`
- `hidden_states`
- `past_key_values`

你不用把这三个都立刻吃透。

现阶段先知道：

- `logits` 最重要，因为它决定下一个 token 倾向选什么
- `past_key_values` 很重要，因为它让后续生成更快

### 你现在只需要记住

这个类本身不做计算，它只是把模型的输出整理好。

---

## 3.7 `_RequestState`

文件位置：`mini_vllm/engine.py`

你可以把它理解成：

“一个请求在运行过程中的小档案。”

为什么需要这个档案？

因为一个请求不是一下子就完成的。

它会经历：

1. 刚提交
2. 处理 prompt
3. 生成第 1 个 token
4. 生成第 2 个 token
5. ...
6. 最后结束

所以引擎必须记住这个请求当前进行到哪一步。

这个类就是干这个的。

它会记住：

- 原始 prompt 是什么
- 已经生成了哪些 token
- 当前的 KV cache 是什么
- 下一轮该喂给模型哪个 token

### 你现在只需要记住

`_RequestState` 不是最终结果，它是“请求正在跑的时候的内部状态”。

---

## 3.8 `GenerationResult`

文件位置：`mini_vllm/engine.py`

你可以把它理解成：

“请求跑完之后的最终成绩单。”

和 `_RequestState` 不一样，它不是运行中的状态，而是结束后的结果。

它会记录：

- 原始 prompt
- 最终生成的 token
- 请求什么时候开始
- 第一个 token 什么时候出来
- 整个请求什么时候结束

这些信息后面可以用来分析：

- 首 token 延迟
- 总延迟
- 吞吐

### 你现在只需要记住

`GenerationResult` 是最终结果，`_RequestState` 是运行中状态。

---

## 3.9 `MiniLLMEngine`

文件位置：`mini_vllm/engine.py`

你可以把它理解成：

“推理调度中心。”

它本身不是神经网络。

它不负责 attention 数学计算。

它负责的是：

- 接收请求
- 组织 batch
- 决定这一步先算谁
- 保存每个请求的状态
- 复用 KV cache
- 收集最终结果

如果没有它，模型虽然能算，但只能傻乎乎地一次处理一个输入。

有了它，模型才更像一个真正的推理服务。

### 它内部最重要的三个概念

#### 1. waiting

刚提交、还没开始处理的请求。

#### 2. active

已经开始生成、但还没结束的请求。

#### 3. finished

已经结束的请求。

所以你可以把 `MiniLLMEngine` 看成：

“不断把请求从 waiting 推到 active，再从 active 推到 finished 的人。”

### 你现在只需要记住

`MiniLLMEngine` 是整个项目里最像“服务端调度器”的类。

---

## 4. 这些 class 是怎么串起来的

这部分只讲最粗的主线。

不要急着抠细节。

---

## 第一步：先创建模型

先用 `ModelConfig` 告诉程序：

“模型应该长成什么样。”

然后创建 `DecoderOnlyTransformer`。

这时模型内部会组装好：

- embedding
- 多层 `DecoderLayer`
- 最后的输出层

---

## 第二步：再创建引擎

创建 `MiniLLMEngine`，并把模型交给它。

这时相当于：

- 模型负责算
- 引擎负责调度

---

## 第三步：请求进来

当你提交一个请求时，引擎会创建一个 `_RequestState`。

此时这个请求还没完成，它只是被登记起来了。

---

## 第四步：模型开始处理 prompt

引擎会调用模型做一次前向计算。

这时在模型内部：

- token 先做 embedding
- 经过多层 `DecoderLayer`
- 每层里都有 `CausalSelfAttention` 和 `FeedForward`
- 最后得到 `DecoderOnlyTransformerOutput`

输出里最重要的两样东西是：

- logits
- `past_key_values`

---

## 第五步：开始逐个 token 生成

后面每生成一个新 token，引擎都会继续调模型。

这时和第一次不同：

- 不用把整段历史都重算
- 因为历史信息已经放进 KV cache 了

所以后续生成会更快。

---

## 第六步：请求结束

当生成到指定数量，或者遇到结束 token，请求就结束。

引擎会把 `_RequestState` 变成 `GenerationResult`。

于是：

- 运行中状态结束
- 最终结果被保存

---

## 5. 这一版你最应该抓住什么

如果你现在只能记住 5 句话，那就记这 5 句：

1. `ModelConfig` 是配置表，不是模型
2. `DecoderOnlyTransformer` 是总模型
3. `DecoderLayer` 是模型里重复堆叠的一层
4. `CausalSelfAttention` 负责看上下文，而且只能看左边
5. `MiniLLMEngine` 负责调度多个请求，不负责神经网络数学本身

只要这 5 句先稳住，后面再学：

- 为什么 KV cache 能提速
- 为什么要分 prefill 和 decode
- 为什么 continuous batching 能提高吞吐

就不会那么乱。

---

## 6. 下一份文档该怎么写

如果这份你觉得顺了，下一份最适合写成：

`一次请求从进入引擎到生成结束，中间到底发生了什么`

也就是只讲时间顺序，不讲太多数学。

建议结构会是：

1. 请求进入 `submit()`
2. 为什么先处理 prompt
3. 什么是 prefill
4. 什么是 decode
5. KV cache 在哪里产生，在哪里复用
6. 请求什么时候结束

如果你愿意，我下一轮直接写 `learn/two.md`，并且继续保持这种“零预设”的写法。
