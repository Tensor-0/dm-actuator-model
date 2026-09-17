#!/usr/bin/env python3
"""负载工况下验证 MuJoCo `<dcmotor>` 复现官方 τ-ω 曲线。

为什么需要这个（而不只是空载验证）
----------------------------------
官方台架是**加载测试**：扭矩从 3.6 一路加载到 19.35 N·m 测出来的。
而 `verify_mujoco_dcmotor.py` 只验了**空载**（扫电压看稳态转速）。

空载只碰到 τ-ω 曲线的一个点（τ≈0）。要证明整条线对，必须**加负载**。

做法
----
在关节上挂一个**恒定反力矩**（用 `qfrc_applied` 模拟测功机加载），
扫不同的负载值，等系统稳定后读 (ω, τ)，看是否落在 τ-ω 直线上。

⚠️ 版本注意：本机 MuJoCo 3.11 用 `input="voltage"`（3.12+ 改为 `pos vel`）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        import mujoco
    except ImportError:
        print("❌ 需要 mujoco。试：/home/zhan/UniLab/.venv/bin/python 跑本脚本")
        return 1

    p = yaml.safe_load((ROOT / "params" / "dm_j4340_2ec.yaml").read_text(encoding="utf-8"))
    V, TAU, W = (float(x) for x in p["mujoco_dcmotor"]["nominal"].split())
    K, R = V / W, (V / W) * V / TAU

    print(f"MuJoCo {mujoco.__version__}")
    print(f"nominal = {V} V / {TAU} N·m / {W} rad/s   →  K={K:.5f}, R={R:.5f}")
    print()

    # 电压进、力矩出；用 qfrc_applied 当测功机
    xml = f"""
    <mujoco>
      <option integrator="implicitfast" gravity="0 0 0"/>
      <worldbody><body name="b">
        <joint name="j" type="hinge" damping="0"/>
        <geom type="capsule" size=".05" fromto="0 0 0 0 0 .3" density="500"/>
      </body></worldbody>
      <actuator>
        <dcmotor name="m1" joint="j" nominal="{V} {TAU} {W}" input="voltage"/>
      </actuator>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    def settle(voltage: float, load_nm: float, steps: int = 200_000) -> tuple[float, float]:
        """跑到稳态，返回 (ω, τ)。load_nm 是施加在关节上的反向负载。"""
        data.qpos[0] = 0.0
        data.qvel[0] = 0.0
        data.qfrc_applied[0] = 0.0
        mujoco.mj_forward(model, data)
        for i in range(steps):
            data.ctrl[0] = voltage
            data.qfrc_applied[0] = -load_nm      # 恒定反力矩
            mujoco.mj_step(model, data)
            # 早期退出：速度稳定
            if i > 20_000 and i % 2000 == 0 and abs(data.qacc[0]) < 1e-9:
                break
        return float(data.qvel[0]), float(data.actuator_force[0])

    # 官方曲线的实测区间
    curve = json.loads((ROOT / "data" / "torque_speed_curve.json").read_text(encoding="utf-8"))
    tau_min = min(r["torque_Nm"] for r in curve)
    tau_max = max(r["torque_Nm"] for r in curve)

    print(f"官方曲线实测区: {tau_min:.2f} ~ {tau_max:.2f} N·m")
    print()
    print(f"{'负载设定':>9} {'稳态ω':>10} {'实测τ':>9} {'ω预测':>10} {'ω误差%':>9}")
    print("-" * 52)

    worst = 0.0
    rows = []
    # 在实测区间内取荷载点
    for load in np.linspace(tau_min, tau_max, 9):
        w, tau = settle(V, float(load))
        # 用实测 τ 反推预测 ω（τ-ω 直线：ω = (V - τ·R/K)/K）
        w_pred = (V - tau * R / K) / K
        err = (w - w_pred) / w_pred * 100 if w_pred else float("nan")
        worst = max(worst, abs(err))
        rows.append({"load_Nm": float(load), "omega": w, "tau": tau, "omega_pred": w_pred})
        print(f"{load:9.2f} {w:10.4f} {tau:9.3f} {w_pred:10.4f} {err:9.2f}")

    print()
    print(f"最大误差 {worst:.2f}%")

    # 与官方曲线逐点比
    print()
    print("与官方提取曲线比对（97 点，按扭矩最近邻）:")
    ct = np.array([r["torque_Nm"] for r in curve])
    cw = np.array([r["omega_rad_s"] for r in curve])
    sim_w = []
    for t in ct:
        # 由 τ-ω 直线预测（模型自身）
        sim_w.append((V - t * R / K) / K)
    sim_w = np.array(sim_w)
    rel = np.abs(sim_w - cw) / cw * 100
    print(f"  平均 {rel.mean():.2f}%   最大 {rel.max():.2f}%   (官方曲线像素提取误差 ±1~2%)")
    print()
    # 对照：若用说明书标称的 40 会差多少（说明 stall_torque 选择的重要性）
    TAU_DS = 40.0
    sim_ds = W * (1.0 - ct / TAU_DS)
    rel_ds = np.abs(sim_ds - cw) / cw * 100
    print(f"  ⚠️ 若用说明书标称 stall=40：平均 {rel_ds.mean():.2f}%  最大 {rel_ds.max():.2f}%")

    # 判据：平均误差应在官方曲线的像素提取噪声量级（±1~2%），
    # 不该定得比这更严 —— 否则是在拟合提取噪声。
    ok = worst < 1.0 and rel.mean() < 3.0
    print()
    if ok:
        print("✅ 负载工况下 τ-ω 复现良好")
        return 0
    print("❌ 偏差过大，请检查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
