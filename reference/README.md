# reference/ —— 外部参考实现（只读，不修改）

本目录存放**别人写的**代码，作为设计参考。**不是我们的代码，不要改。**

| 文件 | 来源 | 用途 |
|---|---|---|
| `bam_action.py` | `rocPAI-Forge/microduck_rl_unilab`<br>`src/microduck_rl_unilab/tasks/microduck/bam_action.py`<br>md5 `586422eeb632d682d626da849dce9665` | ⭐ UniLab 里接入真实执行器模型的**完整参考实现**（451 行）|

**细读笔记**：`../docs/bam_action_源码细读_20260917.md`

## 为什么放在这里

上游 `Motphys/UniLab` PR #1475 把 BAM xl330-m6 电压舵机移植进 UniLab，
动机与我们一致（"执行器模型不同是训练差距主因"）。代码后迁到下游仓库。
下载一份到本地，便于对照设计，**避免网络不可达时无法查阅**。

## ⚠️ 许可与归属

上游 `Motphys/UniLab` 是 **Apache-2.0**。
本目录仅作**阅读参考**，我们的实现不复制其代码（架构思路可借鉴，物理参数完全不同）。
