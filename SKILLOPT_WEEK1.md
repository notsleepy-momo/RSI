# SkillOpt × ALFWorld：第一周最小复现

## 复现边界

本实现复现 SkillOpt 的核心因果闭环，而非整篇论文的全量结果：

1. 冻结 `qwen3.5:4b` 权重；
2. 把 Markdown 技能文档作为可训练状态；
3. 在 ALFWorld 训练任务上采集完整轨迹；
4. 根据轨迹生成一个候选技能版本；
5. 分别用当前技能和候选技能重新运行独立 `valid_seen` 任务；
6. 只有成功率或环境分数严格提升才接受候选；
7. 在 `valid_unseen` 上测试最终技能。

同一批 `valid_unseen` 任务还会额外运行一次空技能（No Skill）基线，便于判断最终技能是否优于裸 Agent。

这不是官方 SkillOpt 代码的逐行移植，而是面向本地 Ollama、小模型和现有 Windows + Docker 环境的最小机制复现。

## 为什么必须分割任务

- `train`：只用于产生失败/成功轨迹和提出技能修改；
- `valid_seen`：只用于接受或拒绝候选技能；
- `valid_unseen`：只用于报告泛化结果。

如果用同一任务同时产生修改并判断是否提升，结果只能说明对该任务进行了重试或记忆，不能证明技能发生了可泛化的改进。

## 运行

先进行不调用模型的结构测试：

```powershell
cd E:\科研\ppt\25262\rsi\benchmark\ALFworld
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_skillopt_minimal.ps1 -DryRun
```

进行单任务、小步数烟雾实验：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_skillopt_minimal.ps1 `
  -Model qwen3.5:4b `
  -TrainIndices 0 `
  -ValidationIndices 0 `
  -TestIndices 0 `
  -MaxSteps 8
```

第一周的小规模正式实验建议：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_skillopt_minimal.ps1 `
  -Model qwen3.5:4b `
  -TrainIndices 0,1,2,3,4 `
  -ValidationIndices 0,1,2 `
  -TestIndices 0,1,2 `
  -MaxSteps 25
```

## 结果文件

结果位于 `results/skillopt_minimal/`：

- `episodes/*.json`：每个任务的 observation、可选动作、模型输出、实际动作和奖励；
- `candidate_edit.json`：失败诊断和候选技能；
- `validation_gate.json`：当前/候选技能的验证分数以及接受决定；
- `best_skill.md`：通过验证门控后的技能；
- `summary.json`：本次实验摘要。

每个 episode 还记录 `model_call_count`、`latency_seconds` 和 `done_reason`；终端会在每个任务开始与结束时打印进度，便于比较优化前后的速度。

## 第一周验收点

- 同一技能哈希能对应到所有使用它的轨迹；
- Actor 每一步都看到原始任务目标、最近轨迹和技能文档；
- 候选技能不会因为验证集持平而被接受；
- 测试集轨迹不参与技能生成或选择；
- 失败动作回退被明确计数，不能伪装成模型的有效决策。
- Actor 只返回整数 `action_id`，程序将编号映射为环境原始动作，避免小模型抄写长动作失败。
- 非法或越界编号会反馈给模型并最多重试 2 次；全部失败后才执行带标记的推进型启发式回退。
- 相同动作与相同观察连续出现 2 次后，该动作会在下一步被临时禁用。
- 最近 4 步无进展时禁用相关 `look/examine/inventory` 动作；默认检测到 8 步短周期后提前结束 episode。
- 动作调用使用 4096 上下文、最多 12 个输出 token，并要求 Ollama 保持模型常驻。

速度优化后的参数可以显式调整：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_skillopt_minimal.ps1 `
  -MaxSteps 20 -ActionRetries 2 -StagnationWindow 8
```

当前最小实现只有一次技能更新。完成烟雾实验后，再扩展到多 epoch、多个候选、ADD/DELETE/REPLACE 显式编辑和多随机种子对照。
