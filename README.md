# dm-actuator-model

达妙 **DM-J4340-2EC (24V)** 执行器建模：参数、提取工具、以及"别人怎么做"的调研。

> **一句话**：达妙官方只发布了性能曲线的**图片**，这个仓库把它数字化成可用的模型参数。
> 顺带回答了"开源社区有没有人给执行器建模"—— **有，而且有成熟的工业范式**（见 `docs/`）。

---

## 这个仓库解决什么问题

双足机器人仿真里，电机常常被建成 **`forcerange = ±28 N·m` 的常数力矩上限**
（与速度无关）。但真机不是这样：**扭矩越大，能达到的转速越低**。

这份数据把那条被忽略的曲线补上了：

```
ω = −0.1186 · τ + 6.402   [rad/s]      r² = 0.9916
```

---

## 仓库结构

```
├── data/
│   ├── source_png/              # 达妙官方原始曲线（下载自 Gitee）
│   ├── torque_speed_curve.json  # ⭐ τ-ω 曲线（97 点）—— 建模最需要这条
│   ├── torque_eff_curve.json    # 效率 / 输入功率 / 输出功率（99 点）
│   ├── speed_const_rpm.json     # 恒转速台架（103 点，用于验证协议）
│   ├── thermal.json             # 温升曲线（54 点）
│   └── README.md                # 数据来源、提取方法、可信度
├── params/
│   ├── dm_j4340_2ec.yaml        # ⭐ 可直接使用的建模参数（带完整限制说明）
│   └── fitted_model.json        # 同一结果的机器生成版
├── tools/
│   ├── extract_curve.py         # 从 PNG 提取曲线（像素级标定）
│   ├── fit_motor_model.py       # 拟合 τ-ω / 效率 / 热模型
│   ├── verify_params.py         # 自检：参数与数据是否一致
│   └── verify_mujoco_dcmotor.py # ⭐ 实测验证参数能驱动 MuJoCo <dcmotor>
└── docs/
    ├── 执行器建模调研_20260917.md        # 调研①：社区怎么做执行器建模
    └── MuJoCo执行器建模_调研_20260917.md  # 调研②：MuJoCo 原生 <dcmotor> + mjlab（含更正）
```

---

## 快速开始

```bash
# 1) 重新提取（需要原始 PNG，已在 data/source_png/）
python3 tools/extract_curve.py --img-dir data/source_png --out data/

# 2) 拟合参数
python3 tools/fit_motor_model.py --data data --out params/fitted_model.json

# 3) 自检（改了参数务必跑）
python3 tools/verify_params.py

# 4) 验证参数能真的驱动 MuJoCo <dcmotor>（需要 mujoco）
/home/zhan/UniLab/.venv/bin/python tools/verify_mujoco_dcmotor.py
```

依赖：`numpy`, `Pillow`, `pyyaml`,（拟合热模型需要 `scipy`）

---

## 核心参数速览

| 项 | 值 | 来源 |
|---|---|---|
| **τ-ω 斜率** | −0.1186 rad/s per N·m | 实测曲线拟合（r²=0.9916） |
| **空载转速** | 6.402 rad/s（61.1 rpm） | 实测外推（τ=0） |
| **连续扭矩** | 12 N·m | 额定 |
| **峰值效率** | 65.6% @ 11.1 N·m | 实测 |
| **轻载效率** | 31% @ 3.3 N·m | 实测（很低！） |
| **热时间常数** | 263 s | 温升曲线拟合 |
| **稳态温升** | 110 °C | 同上 |

对齐 **Isaac Lab `DCMotor`** 的字段命名，可直接填进主流框架，详见 `params/dm_j4340_2ec.yaml`。

---

## ⚠️ 使用前必读的三条限制

1. **不要外推到实测区外**。曲线实测只覆盖 **3.6~19.35 N·m / 38~56 rpm**。
   按线性式算高转速（如 8.63 rad/s）会得到**负扭矩**，物理上无意义。

2. **堵转扭矩未实测**。说明书标称峰值 40 N·m，曲线只到 19.35 N·m。
   线性外推得 54 N·m，与标称**差 35%** —— 参数文件里两套都给了，按用途选。

3. **热模型只有一个工作点**（12 N·m 恒扭矩），无法反推热阻的功率依赖。

完整清单见 `params/dm_j4340_2ec.yaml` 的 `limitations:` 一节。

---

## 数据质量发现（提取时查出的）

- ⚠️ `温升.png` 与 `温升14Nm.png` 的**曲线逐像素完全相同**
  （逐像素比对：仅标题 238 px 不同，绘图区 0 差异）→ 后者是副本，不是独立工况
- ⚠️ 两张图是**不同测试协议**，不可混用：
  - `扭矩效率.png` = 自然变速加载 → **τ-ω 曲线来源**
  - `速度效率.png` = 恒 35 rpm 台架（实测 35.87±0.24 rpm）→ 效率来源
- ⚠️ 达妙**只发布 PNG，没有原始 CSV**，本仓库数据是从像素反推的（估 ±1~2%）

---

## 调研结论摘要

**有人在做，而且很成熟**：

| 层次 | 代表 | 我们能用的部分 |
|---|---|---|
| **MuJoCo 原生** | ⭐ **`<dcmotor>` 元素** | **直接可用**：填 3 个数（见上节） |
| **MuJoCo 生态库** | ⭐ **`mujocolab/mjlab`**（⭐3084） | `dc_actuator.py` / `builtin_actuator.py` / ActuatorNet |
| 工业范式（Isaac） | Isaac Lab `DCMotor` | 参考；MuJoCo 原生模型**更完整** |
| 进阶范式 | PACE（ETH，Apache-2.0） | "只扩展子类、不改内核"的策略 |
| 达妙专属 | 摩擦辨识工具 | 可补摩擦/惯量参数（⚠️ 非商用许可） |

**空白点**：没有任何开源项目发布过 DM-J4340-2EC 的 τ-ω 参数 —— **本仓库的数据是可贡献的新内容**。

详见 `docs/执行器建模调研_20260917.md` 和 `docs/MuJoCo执行器建模_调研_20260917.md`。

---

## ⭐ 怎么用起来：MuJoCo 原生 `<dcmotor>`（三个数即可）

**MuJoCo 3.10+ 原生就有 `<dcmotor>` 执行器**（含反电动势、电流饱和、电感、热模型、
齿槽转矩、LuGre 摩擦）—— 不需要自己写代码。

把 MJCF 里的 `<position>` 换成：

```xml
<actuator>
  <!-- nominal = "voltage stall_torque no_load_speed" -->
  <dcmotor name="leg_l1_joint_motor" joint="leg_l1_joint"
           nominal="24 40 6.4017"
           input="pos vel ff"      <!-- 对应达妙 MIT 的 pos/vel/t_ff -->
           saturation="12 0 0"     <!-- 连续扭矩 12 N·m -->
           forcerange="-28 28"/>
</actuator>
```

MuJoCo 会**自动推出** `K = V/ω_no_load = 3.749`、`R = K·V/τ_stall = 2.249`，
力矩公式为 `τ = (K/R)·v − (K²/R)·ω`（即官方 `τ = (Kt/R)(v − Ke·ω)`）。

验证脚本：`tools/verify_mujoco_dcmotor.py`（实测扫电压，误差 0.00%）

> 另见 `mujocolab/mjlab`（⭐3084, Apache-2.0）：MuJoCo 生态的 Isaac-Lab 风格框架，
> 有完整的执行器模块（`dc_actuator.py` / `builtin_actuator.py` / `learned_actuator.py`）。
> 详见 `docs/MuJoCo执行器建模_调研_20260917.md`。

### 仍在本仓库范围外、需要自建的部分

MuJoCo 的 `<dcmotor>` **不含**：
- 减速比（用 `<actuator gear=>` 单独给）
- 效率曲线（本仓库的 65.6% 峰值数据可做能量惩罚项）
- 达妙固件的具体行为（12-bit 量化、Kd 范围 [0,5]）

主干（τ-ω + 反电动势 + 延迟）原生就有。

---

## 数据来源与致谢

- 曲线数据：达妙官方 Gitee 仓库
  <https://gitee.com/kit-miao/DM-J4340-2EC>（`测试数据/性能曲线/V1.1/`）
- 建模范式参考：Isaac Lab（BSD-3-Clause）、PACE（Apache-2.0）

> ⚠️ 本仓库只做**数据提取与整理**，未复制任何第三方代码。
> 若引用达妙官方图片，版权归达妙所有。
