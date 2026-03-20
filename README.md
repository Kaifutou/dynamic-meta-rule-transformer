# 动态元规则 Transformer Demo (DMRT)

本项目是一个最小可行性架构（PoC）演示，目标是探索一种突破传统静态权重限制的神经网络范式：**动态元规则网络（Dynamic Meta-Rule Transformer）**。  
核心思想是把模型中的“通用学习规则（meta-rules）”与“具体事实记忆（instance knowledge）”解耦，使网络在**仅前向推断（无需反向传播）**过程中，就能基于内部状态动态调制局部权重，实现对非平稳环境的实时适应与长期记忆沉淀。

---

## 📐 数学模型（来自 `dynamic_meta_rule_network.tex`）

下面是我在数学稿中定义的核心对象与方程，工程实现与这些定义一一对应。

### 1) 静态预制模型（对照基线）

$$
y_t = f(x_t; W)
$$

含义：推断阶段参数固定，不存在可写的内部知识载体。

### 2) 动态元规则网络（主定义）

$$
W_t = W^{\mathrm{slow}} + G_\theta(M_t, K_t, x_t)
$$
$$
y_t = f(x_t; W_t)
$$

- $W^{\mathrm{slow}}$：长期稳定参数（承载元规则）
- $G_\theta$：动态权重生成器
- $M_t, K_t$：快/慢记忆状态

### 3) 快记忆更新（工作记忆写入）

$$
M_{t+1}=U_\theta(M_t, x_t, y_t, e_t)
$$

一个常见门控形式：

$$
z_t = \Phi_\theta(x_t, M_t, y_t, e_t),\quad
\alpha_t = \sigma\!\bigl(a_\theta(x_t, M_t, e_t)\bigr)
$$
$$
M_{t+1}=\lambda M_t + \alpha_t z_t,\quad 0\le \lambda \le 1
$$

### 4) 慢记忆巩固（长期沉淀）

$$
K_{t+1}=(1-\rho_t)K_t + \rho_t C_\theta(M_{t+1}),\quad 0\le \rho_t \le 1
$$

- $\rho_t$：巩固强度
- $C_\theta$：将快记忆映射到长期记忆的机制

### 5) Transformer 实例化

对第 $l$ 层：

$$
W_t^{(l)} = W_{\mathrm{slow}}^{(l)} + G_\theta^{(l)}(M_t, K_t)
$$

动态权重参与注意力投影：

$$
Q_t^{(l)} = H_t^{(l)}W_t^{Q,(l)},\quad
K_t^{(l)} = H_t^{(l)}W_t^{K,(l)},\quad
V_t^{(l)} = H_t^{(l)}W_t^{V,(l)}
$$

在本仓库里，这一步用“有界动态 LoRA”落地到 `Q/K/V/FFN`。

### 6) 状态空间视角（系统化表达）

令 $S_t=(M_t, K_t)$，系统可写成：

$$
y_t = F_\theta(x_t, S_t),\quad
S_{t+1}=T_\theta(S_t, x_t, y_t, e_t)
$$

含义：该模型本质上是一个“带可写内部状态的动力系统”。

### 7) 有界扰动命题（可塑性受控）

若基模型对权重变化是 $L$-Lipschitz，且

$$
\|G_\theta(M_t, K_t, x_t)\| \le B
$$

则输出偏差有界：

$$
\|y_t - f(x_t; W^{\mathrm{slow}})\| \le LB
$$

对应工程里，我在训练阶段加入动态扰动范数约束，避免在线漂移失控。

### 8) 数学定义到代码模块映射

- $G_\theta$：`metarule_demo/model.py` -> `MetaRuleController.generate_dynamic`
- $U_\theta$：`MetaRuleController.update_state`（`alpha` 写门 + 经验摘要）
- $C_\theta$：`update_state`（`rho` 巩固门 + `c_proj`）
- $M_t, K_t$：`metarule_demo/memory.py` -> `MemoryState`

---


## 🌟 核心特性 (Key Features)

- **推断即训练 (Inference as Training)**  
  告别“静态知识容器”范式。模型在处理序列时，将实时经验映射为动态权重扰动，网络行为随上下文演化。

- **双系统记忆矩阵 (Dual-Slot Memory Matrix)**  
  引入槽位式状态矩阵，避免单向量状态的严重干扰：  
  - 快记忆 `M_t`：短期工作区、快速覆写  
  - 慢记忆 `K_t`：长期模式沉淀、巩固更新

- **有界动态 LoRA 注入 (Bounded Dynamic LoRA)**  
  动态增量通过低秩分解实时生成并注入 Transformer 的 `Q/K/V + FFN`，在保持计算成本可控的同时，约束动态漂移范数。

- **规则突变评测基准 (Synthetic Rule Shift Benchmark)**  
  内置可控的规则切换与重复回归任务，用于验证：  
  - 分布切换下的在线适应速度  
  - 重复经验下的冷启动收益（记忆沉淀）

- **可解释交互调试 (Interactive Comparison Console)**  
  支持手工输入和自动分布内采样，四个模型变体并排输出预测、置信度、门控强度和动态扰动强度。

---

## 🧩 我实现了哪些功能（以及怎么用）

### 1) 训练功能（四个变体同步训练）

```bash
python run_train.py --output-dir runs/metarule_demo --device cpu
```

固定训练四个变体：
- `full`：动态权重 + 快记忆 + 慢记忆
- `static`：纯静态基线
- `fast_only`：动态权重 + 快记忆（无慢记忆）
- `memory_no_dynamic`：有记忆，但禁用动态调权

常用参数：
- `--train-steps`
- `--eval-interval`
- `--batch-size`
- `--learning-rate`

示例：

```bash
python run_train.py --output-dir runs/tuned_demo --device cpu --train-steps 1200 --eval-interval 100 --batch-size 32
```

---

### 2) 在线评估功能（只测在线适应与记忆沉淀）

```bash
python run_online_eval.py --output-dir runs/metarule_demo --device cpu
```

流程说明：
- 自动读取 `checkpoints/*_best.pt`
- 冻结网络参数
- 按 phase 连续输入 episode
- 仅允许 `M_t/K_t` 在线更新

我主要看两类指标：
- 分布切换后的恢复能力（adaptation）
- 规则回归时的早期收益（consolidation）

可调参数：
- `--online-episodes-per-phase`
- `--online-cycles`
- `--batch-size`

---

### 3) 出图功能（自动生成核心图）

```bash
python plot_metrics.py --log-dir runs/metarule_demo
```

会生成：
- `adaptation_curve.png`：在线准确率曲线（平滑）
- `consolidation_gain.png`：冷启动收益条形图

---

### 4) 交互演示功能（对话式对比四个变体）

```bash
python run_interactive_demo.py --output-dir runs/metarule_demo --checkpoint-kind best --device cpu
```

#### 4.1 手工输入模式

```text
supports> 1->3, 5->2, 7->7, 9->0
query x> 5
target y (optional)> 2
```

输出列含义：
- `pred`：预测值
- `conf`：置信度（max softmax）
- `alpha`：快记忆写入强度
- `rho`：慢记忆巩固强度
- `delta_norm`：动态权重扰动强度
- `correct`：若提供 `target`，显示 `Y/N`

#### 4.2 自动采样模式（推荐）

- `sample` 或 `:sample`：自动抽一条分布内样本并评估
- `:sample 2`：指定 `phase_id=2` 抽样

自动采样会先打印 supports/query/target，再给出四变体并排对比结果，避免手工构造 OOD 样本导致误判。

#### 4.3 会话命令

- `help`
- `reset`（重置所有变体记忆状态）
- `exit`

---

### 5) 程序接口（用于二次开发）

如果你希望在 Python 中集成，而非走 CLI：

- `MemoryState.reset(batch_size)`, `MemoryState.detach()`
- `MetaRuleTransformer.forward(tokens, state, feedback=None)`
- `RuleShiftEpisodeDataset.sample_batch(split, phase_id)`
- `run_train(config)`
- `run_online_eval(checkpoint_dir, config)`
- `plot_metrics(log_dir)`
- `interactive_demo(output_dir, checkpoint_kind, device, variants)`

---

## 🚀 我推荐的完整使用流程

1. 训练：

```bash
python run_train.py --output-dir runs/tuned_demo --device cpu --train-steps 1200 --eval-interval 100 --batch-size 32
```

2. 在线评估：

```bash
python run_online_eval.py --output-dir runs/tuned_demo --device cpu --online-episodes-per-phase 20 --online-cycles 2
```

3. 生成图表：

```bash
python plot_metrics.py --log-dir runs/tuned_demo
```

4. 打开交互对话：

```bash
python run_interactive_demo.py --output-dir runs/tuned_demo --checkpoint-kind best --device cpu
```

建议先使用 `sample` 验证分布内行为，再做手工对照实验。

---

## 📦 输出文件怎么读

`runs/<实验名>/` 目录下：

- `config.json`：本次实验配置
- `train_metrics.csv`：训练过程指标
- `val_metrics.csv`：阶段验证指标
- `train_summary.json`：各变体最佳验证准确率
- `online_metrics.csv`：在线逐 episode 指标
- `online_summary.json`：在线评估汇总（适应 + 沉淀）
- `adaptation_curve.png`：适应曲线图
- `consolidation_gain.png`：沉淀收益图
- `checkpoints/*_best.pt`：最佳权重

---

## 🛠 常见问题（我自己踩过的坑）

### 1) 报错：`Checkpoint dir not found`
你指定的 `--output-dir` 下没有训练产物。先运行训练，或切换到已有目录（如 `runs/tuned_demo`）。

### 2) 交互里连续全错
常见原因：
- 训练步数不足
- 手工样本偏离训练分布
- 样本量过小导致方差大

建议优先用 `sample` 看分布内表现，再做手工案例分析。

---

## 🗂 项目结构

```text
metarule_demo/
  config.py
  data.py
  memory.py
  model.py
  train.py
  online_eval.py
  plotting.py
  interactive.py
run_train.py
run_online_eval.py
plot_metrics.py
run_interactive_demo.py
tests/
```

---

## 📄 许可 (License)

本仓库默认未附带许可证。
