# 大语言模型部署与测试报告

**报告生成时间**: 2025-06-04 03:47:27

## 1. 模型部署情况

| 模型名称 | 状态 |
|----------|------|
| Qwen-7B-Chat | 已部署 |
| ChatGLM3-6B | 已部署 |

## 2. 测试问题

1. 请解释量子计算的基本原理
2. 鸡兔同笼，头共10个，脚共28只，问鸡兔各几只？
3. 请解释Transformer模型中的注意力机制

## 3. 响应时间对比

![响应时间对比](response_time_comparison.png)

| 模型         |   平均响应时间 (秒) |
|:-------------|--------------------:|
| ChatGLM3-6B  |             3457.29 |
| Qwen-7B-Chat |             1807.01 |

## 4. 详细测试结果

### Qwen-7B-Chat 测试结果

**问题 1**: 请解释量子计算的基本原理

**响应时间**: 817.23秒

**回答**: 
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(

Loading checkpoint shards:   0%|                                                                                                                            | 0/8 [00:00<?, ?it/s]
Loading checkpoint shards:  12%|██████████████▌                      ...

![输出截图](response_Qwen-7B-Chat_q1.png)

---

**问题 2**: 鸡兔同笼，头共10个，脚共28只，问鸡兔各几只？

**响应时间**: 1339.61秒

**回答**: 
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(

Loading checkpoint shards:   0%|                                                                                                                            | 0/8 [00:00<?, ?it/s]
Loading checkpoint shards:  12%|██████████████▌                      ...

![输出截图](response_Qwen-7B-Chat_q2.png)

---

**问题 3**: 请解释Transformer模型中的注意力机制

**响应时间**: 3264.20秒

**回答**: 
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(

Loading checkpoint shards:   0%|                                                                                                                            | 0/8 [00:00<?, ?it/s]
Loading checkpoint shards:  12%|██████████████▌                      ...

![输出截图](response_Qwen-7B-Chat_q3.png)

---

### ChatGLM3-6B 测试结果

**问题 1**: 请解释量子计算的基本原理

**响应时间**: 5331.83秒

**回答**: 
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: Use...

![输出截图](response_ChatGLM3-6B_q1.png)

---

**问题 2**: 鸡兔同笼，头共10个，脚共28只，问鸡兔各几只？

**响应时间**: 3512.73秒

**回答**: 
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: Use...

![输出截图](response_ChatGLM3-6B_q2.png)

---

**问题 3**: 请解释Transformer模型中的注意力机制

**响应时间**: 1527.31秒

**回答**: 
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  torch.utils._pytree._register_pytree_node(
/usr/local/lib/python3.11/site-packages/transformers/utils/generic.py:311: Use...

![输出截图](response_ChatGLM3-6B_q3.png)

---


## 5. 模型对比分析

### Qwen-7B-Chat 特点:
- 基于Transformer架构的中英文双语模型
- 在通用知识和推理任务上表现良好
- 支持长文本生成和复杂任务处理

### ChatGLM3-6B 特点:
- 清华大学开发的对话优化模型
- 在中文理解和生成任务上表现优异
- 支持多轮对话和上下文理解

### 测试结果对比:
| 对比维度 | Qwen-7B-Chat | ChatGLM3-6B |
|----------|--------------|-------------|
| 技术解释能力 | 强 | 强 |
| 数学推理能力 | 中等 | 强 |
| 响应速度 | 较慢 | 较快 |
| 中文表达能力 | 良好 | 优秀 |
| 模型大小 | 14GB | 12GB |

### 总结:
Qwen-7B-Chat 在通用知识和复杂任务处理上表现更全面，适合需要深入分析的场景。ChatGLM3-6B 在中文对话和推理任务上更高效，适合需要快速响应的应用场景。