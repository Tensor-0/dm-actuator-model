# `bam_action.py` 全文细读（451 行）

> **来源**：`rocPAI-Forge/microduck_rl_unilab` → `src/microduck_rl_unilab/tasks/microduck/bam_action.py`
> **md5**：`586422eeb632d682d626da849dce9665`（本地副本 `reference/bam_action.py`）
> **上游**：`Motphys/UniLab` PR #1475 / issue #1474（2026-09-03 合并）

---

## 〇、一句话总结

**这是一份把"真实电机模型"接进 UniLab 的完整参考实现**，
451 行做完了一个电压舵机（含延迟/电池/固件 P 律/反电动势/摩擦），
**零框架契约变更**，并配 18 KB 测试。

对我们的价值：**架构可以直接抄，物理部分我们要换成达妙的 τ-ω**。

---

## 一、架构：它怎么"接上"框架（⭐ 核心可复用部分）

### 1.1 三个关键声明

```python
class BamVoltageAction(ActionTerm):
    requires_substep_state_feedback: ClassVar[bool] = True   # ← ①
```

```python
# __init__ 里
actuator_ids, actuator_names = self._entity.find_actuators(cfg.actuator_names)
joint_ids, joint_names = self._entity.find_joints_by_actuator_names(cfg.actuator_names)
# 强制 1:1 映射
if not actuator_ids or len(actuator_ids) != len(joint_ids):
    raise ValueError("requires a non-empty 1:1 actuator/joint selection")
```

```python
# apply_actions 结尾  ← ②
self._entity.data.write_ctrl(applied, actuator_ids=self._actuator_ids)
```

### 1.2 框架侧的自动接线（本机代码已确认）

`src/unilab/envs/manager_based_rl_env.py:385-410`：

```python
def _configure_action_control(self) -> None:
    if (self._cfg.sim_substeps <= 1
        or not self.action_manager.active_terms
        or not self.action_manager.requires_substep_state_feedback):   # ← 读这个 flag
        return
    try:
        self._backend.set_pre_step_control(self._apply_manager_control)  # ← 自动注册
    except NotImplementedError as exc:
        raise NotImplementedError(
            "ActionManager capability 'state-feedback actions on every physics substep' "
            f"is unavailable on backend '{self._backend.backend_type}': {exc}") from exc

def _apply_manager_control(self, backend, control):
    del backend, control
    self._sim_step_counter += 1
    self.action_manager.apply_action()      # ← 每个物理 substep 调一次
    return self._control
```

**⇒ 只要 action term 声明 `requires_substep_state_feedback = True`，
框架**自动**把 `apply_actions()` 挂到每个物理 substep 上。
不需要改框架、不需要注册别的。**

### 1.3 为什么必须逐 substep（而不是 50 Hz 控制步）

它的 docstring 说明：

> "mirroring the firmware's **200 Hz** position loop under the **50 Hz** policy"

我们的情况一样：
- dm10 `sim_dt=0.005`（200 Hz 物理）、`ctrl_dt=0.02`（50 Hz 策略）
- **τ-ω 限幅依赖瞬时 `q̇`** —— 若只在 50 Hz 算，中间 4 个 substep 用的是同一份常值
- 而高速段的 τ-ω 斜率极陡，用 20 ms 前的速度算会严重失真

---

## 二、逐段拆解：它做了什么

### 段 1：延迟缓冲（313-331 行）

```python
def _delay_append(self, q_target):
    self._delay_pointer = (self._delay_pointer + 1) % capacity
    self._delay_buffer[self._delay_pointer] = q_target
    first = self._delay_pushes == 0
    if np.any(first):
        self._delay_buffer[:, first] = q_target[first]   # 首次全填充
    self._delay_pushes += 1

def _delay_read(self):
    lag = self._env.rng.integers(self._delay_min_lag, self._delay_max_lag + 1,
                                 size=self.num_envs)          # ← 每 substep 重采样
    valid = np.minimum(lag, np.maximum(self._delay_pushes, 1) - 1)  # 夹住历史深度
    index = (self._delay_pointer - valid) % capacity
    return self._delay_buffer[index, np.arange(self.num_envs)]
```

**要点**：
- LIFO 环形缓冲，容量 = `max_lag + 1`
- **lag 每个 substep 重新采样**（不是固定值）
- 夹住 `valid`，避免访问未初始化历史
- reset 时清空 + 归零 `_delay_pushes` → 下次 append 自动回填

### 段 2：电池电压 + 负载压降（385-386 行）

```python
load = np.abs(self._prev_motor_torque).sum(axis=1, keepdims=True)
vin_eff = np.maximum(self._vin - self._vin_drop_gain * load, self._vin_min)
```

- `_vin` / `_vin_drop_gain` **startup 采样一次，跨 reset 保持**
- 压降用**上一 substep 的力矩**（不是当前的，避免隐式耦合）

### 段 3：固件 P 律 + 电流限窗 + PWM 截断（388-395 行）

```python
duty = (q_target_delayed - q) * (self._kp_fw * self._error_gain)
center = self._kt * dq / vin_eff                       # 反电动势对应的占空比中心
span   = self._resistance * self._max_current / vin_eff # 电流上限对应的半宽
duty = np.clip(duty, center - span, center + span)     # ← 电流限幅（不对称窗）
duty = np.clip(duty, -self._max_pwm, self._max_pwm)    # 物理 PWM
volts = vin_eff * duty
```

⭐ **这个 `center ± span` 窗口很巧妙**：
把"电流不超过 Imax"表达成"占空比落在以反电动势为中心的对称窗内"，
因为 `i = (v − Ke·ω)/R`，而 `v = vin·duty` → `duty = (i·R + Ke·ω)/vin`。

### 段 4：含反电动势的力矩（397-398 行）

```python
motor_torque = (self._kt * volts - self._kt**2 * dq) / self._resistance
```

即 `τ = (Kt/R)(v − Ke·ω)` —— **与我们达妙的 τ-ω 公式完全同构**。

> 注意：`self._kt**2` 是因为 Kt = Ke = kt，所以 `Kt·Ke = kt²`。

### 段 5：外力矩的有限差分估计（400-403 行）

```python
accel_fd = (dq - self._prev_joint_vel) / self._env.physics_dt
external_torque = self._armature * accel_fd - self._prev_applied_torque
```

⚠️ **这是它明说的"近似边界"**：拿不到 `qfrc_*`，只能用
`I_eff·(dq_t − dq_{t−1})/dt − τ_applied_prev` 反推。
`I_eff` 只用标称 `dof_armature`（**故意不伪造连杆惯量**）。

### 段 6：摩擦预算 + 静摩擦的"力矩域复刻"（405-418 行）

```python
stribeck_coeff = np.exp(-((np.abs(dq) / cfg.dtheta_stribeck) ** cfg.stribeck_alpha))
frictionloss = self._friction_budget(prev_motor_torque, external_torque, stribeck_coeff)

# ⭐ 关键：在力矩域复刻求解器的静摩擦 clip
static = (np.abs(dq) < _DQ_EPS) & (np.abs(motor_torque) < frictionloss)
applied = np.where(static, 0.0, motor_torque - frictionloss * np.sign(dq))
applied = applied - cfg.friction_viscous * dq
```

**这段是全文最值得学的地方**：
上游把摩擦写进 `dof_frictionloss` 让**求解器**做静摩擦判定；
UniLab 没这个通道，于是**在力矩域手写**：
- 准静态（`|dq| < 1e-3`）**且**电机力矩打不破摩擦 → 输出 **0**（卡住）
- 否则 → 干摩擦**反向**于运动
- 粘性项无条件减去

---

## 三、文件结构一览

| 行 | 内容 |
|---|---|
| 1-38 | 模块 docstring（**含两条近似边界的诚实记录**）|
| 40-73 | 常量（`_XL330_ERROR_GAIN`、`_DQ_EPS`）|
| 76-110 | 校验小工具（`_real` / `_int` / `_range`）|
| 113-157 | **`BamVoltageActionCfg`**（dataclass，含全部拟合参数默认值）|
| 160-165 | 类声明 + `requires_substep_state_feedback = True` |
| 167-217 | `__init__`：id 映射、缓冲分配、startup 采样 |
| 219-265 | `_validate_cfg`：逐参数校验 |
| 267-295 | 5 个 property（含 `applied_torque` —— 对外暴露"最接近真值"的量）|
| 297-311 | `process_actions`：`q_des = action·scale + default_joint_pos` |
| 313-331 | 延迟缓冲 append / read |
| 333-363 | `_friction_budget` |
| **365-425** | **`apply_actions`：核心管线** |
| 427-448 | `reset` |

---

## 四、⭐ 对我们 dm10 的可复用度

| 组件 | 我们能直接抄吗 | 说明 |
|---|---|---|
| **架构**（flag + `write_ctrl` + 1:1 映射） | ✅ **完全可抄** | 与电机型号无关 |
| **延迟缓冲** | ✅ **可抄** | 我们也有指令延迟问题 |
| **`center ± span` 电流限窗** | ✅ 可抄 | 通用 DC 电机 |
| **反电动势力矩公式** | ✅ **可抄**（改参数） | 与达妙 τ-ω 同构 |
| **静摩擦力矩域复刻** | 🟡 可抄方法 | 但我们暂无摩擦参数 |
| 电池模型 / 固件 P 律 | ❌ 不需要 | 达妙是 CAN 直驱，非舵机 |
| BAM m6 摩擦参数 | ❌ 不适用 | 那是 xl330 的拟合值 |
| **外力矩有限差分** | 🟡 备选 | 我们可能不需要 |

### ⚠️ 两个必须注意的差异

**1. 达妙的 τ-ω 可以用 MuJoCo 原生 `<dcmotor>`，不必手写**

它手写整条电气链是因为 xl330 的固件 P 律/PWM 语义特殊。
我们的达妙电机**有官方 τ-ω 曲线**，MuJoCo 的 `<dcmotor>` 原生就实现了
`τ = (Kt/R)(v − Ke·ω)` + 电流饱和（见 `mujoco-native-dcmotor-exists`）。

**2. 但要注意 actuator 类型冲突**

- BAM 路线：XML 保持 `<motor>`（`forcerange` 做最终截断），力矩在 Python 算
- `<dcmotor>` 路线：XML 换成 `<dcmotor>`，MuJoCo 内部算

**两者不能混**（`<dcmotor>` 的 `gainprm/biasprm` 语义特殊，
BAM 那套"从 `ctrl_range` 读范围"的假设不再成立）。

---

## 五、给我的三条设计启示

### 1. 抄架构，不抄物理
`requires_substep_state_feedback` + `write_ctrl` 这套是通用骨架。
物理部分我们**要么全用原生 `<dcmotor>`，要么全自己写**，不要半半。

### 2. 诚实标注近似（它在 docstring 里专列一节）
它把"拿不到 `qfrc_*`、摩擦写不进去"两条写在**模块开头**，还说明了
"稳态无差、快速暂态略低估"的**影响范围**。
我们的 τ-ω 模型也该这样标注边界（实测区 3.6~19.35 N·m、不可外推）。

### 3. 留一个 `applied_torque` 这样的 property
它明确说：
> "Closest observable to the upstream `actuator_force`" —— 对外的"最接近真值"的量

**这正是我们 Q2 需要的力矩通道** —— 不用去 backend 抠 `actuator_force`，
**让 action term 自己把算出来的力矩存下来并暴露**。

---

## 六、结论

**这份代码证明了两件事**：
1. **"在 UniLab 里接入真实执行器模型"是可行的、有人做过的**（零契约变更）
2. **"读不到真力矩"是框架的既定限制** —— 连上游都只能近似 + 标注

**⇒ 我们的 Q2 方案可以据此定稿**：
- 若走 `<dcmotor>` 原生路线 → 架构照抄，物理交给 MuJoCo
- 力矩通道 → 学它，**在 action term 里存一份并暴露**，不去碰 backend
