#!/usr/bin/env python3
"""验证 params/ 里的参数能真的驱动 MuJoCo `<dcmotor>`。

为什么需要这个
--------------
调研报告说"MuJoCo 原生 `<dcmotor>` 能表达 τ-ω 曲线"——这是一个**断言**。
本脚本把它变成**可复现的验证**：用参数文件里的 nominal 建一个模型，
扫电压、测稳态转速，与理论值比对。

也顺带验证一个容易搞错的点：`nominal` 里的三个数
**共同定义** τ-ω 直线（斜率 = ω_no_load/τ_stall），不是各自独立可调的。

需要能 `import mujoco`（建议用 UniLab 的 venv）：
    /home/zhan/UniLab/.venv/bin/python tools/verify_mujoco_dcmotor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    try:
        import mujoco
    except ImportError:
        print("❌ 需要 mujoco。试：/home/zhan/UniLab/.venv/bin/python 跑本脚本")
        return 1

    p = yaml.safe_load((ROOT / "params" / "dm_j4340_2ec.yaml").read_text(encoding="utf-8"))
    mj = p["mujoco_dcmotor"]
    V, TAU, W = (float(x) for x in mj["nominal"].split())

    print(f"MuJoCo {mujoco.__version__}")
    print(f"参数: nominal = {V} V / {TAU} N·m / {W} rad/s   (input={mj['input']})")
    print()

    # input="pos vel ff" 时 ctrl 语义是 [pos, vel, ff]，不便扫电压；
    # 这里用 input="voltage" 的等价模型来验证电气特性。
    xml = f"""
    <mujoco>
      <option integrator="implicitfast"/>
      <worldbody><body name="b">
        <joint name="j" type="hinge"/>
        <geom type="capsule" size=".05" fromto="0 0 0 0 0 .3" density="500"/>
      </body></worldbody>
      <actuator>
        <dcmotor name="m1" joint="j" nominal="{V} {TAU} {W}" input="voltage"/>
      </actuator>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # --- 1. 检查 MuJoCo 推出的 K / R ---
    # 由 nominal="V τ_stall ω_no_load" 推出 (官方文档原文):
    #     K = V / ω_no_load
    #     R = K · V / τ_stall
    K = V / W
    R = K * V / TAU
    got = model.actuator_gainprm[0]
    print("① 派生参数")
    print(f"   期望  K = V/ω_no_load = {K:.5f}")
    print(f"   期望  R = K·V/τ_stall = {R:.5f}")
    print(f"   实测  gainprm[0] = {got[0]:.5f}   (= R 本身)")
    print(f"   实测  gainprm[1] = {got[1]:.5f}   (= K)")
    print()

    # --- 1b. 数值反推力矩公式（不靠猜 gainprm 语义）---
    def force_at(ctrl_v: float, qvel: float) -> float:
        data.qpos[0] = 0.0
        data.qvel[0] = qvel
        data.ctrl[0] = ctrl_v
        data.act = 0.0
        data.act_dot = 0.0
        mujoco.mj_forward(model, data)
        return float(data.actuator_force[0])

    d_tau_d_v = (force_at(V, 0.0) - force_at(0.0, 0.0)) / V
    d_tau_d_w = (force_at(V, 6.0) - force_at(V, 0.0)) / 6.0
    print("② 力矩公式（数值反推）")
    print(f"   dτ/dv   = {d_tau_d_v:.5f}   理论 K/R  = {K/R:.5f}  "
          f"{'✅' if abs(d_tau_d_v - K/R) < 1e-3 else '❌'}")
    print(f"   dτ/dω   = {d_tau_d_w:.5f}   理论 −K²/R = {-K*K/R:.5f}  "
          f"{'✅' if abs(d_tau_d_w + K*K/R) < 1e-3 else '❌'}")
    print("   ⇒ τ = (K/R)·v − (K²/R)·ω   ← 即官方 τ = (Kt/R)(v − Ke·ω)")
    print()

    # --- 2. 扫电压，验证 τ-ω 行为 ---
    print("③ 空载稳态转速（跑到稳态）")
    print(f"   {'电压 V':>7} {'实测 ω':>10} {'理论 V/K':>10} {'误差%':>8}")
    ok = True
    for v in (V, 18.0, 12.0, 6.0, 3.0):
        data.qpos[0] = 0.0
        data.qvel[0] = 0.0
        mujoco.mj_forward(model, data)
        for _ in range(60000):
            data.ctrl[0] = v
            mujoco.mj_step(model, data)
        w = float(data.qvel[0])
        pred = v / K
        err = (w - pred) / pred * 100
        if abs(err) > 0.5:
            ok = False
        print(f"   {v:7.1f} {w:10.4f} {pred:10.4f} {err:8.2f}")

    print()
    # --- 3. τ-ω 端点 ---
    print("④ τ-ω 直线端点")
    print(f"   τ=0 → ω = {V/K:.4f} rad/s   (文件 no_load_speed = {W}) "
          f"{'✅' if abs(V/K-W)<1e-3 else '❌'}")
    print(f"   ω=0 → τ = {K*V/R:.4f} N·m   (文件 stall_torque = {TAU}) "
          f"{'✅' if abs(K*V/R-TAU)<1e-3 else '❌'}")
    print()

    if ok:
        print("✅ MuJoCo `<dcmotor>` 复现了参数文件的 τ-ω 行为（误差 < 0.5%）")
        return 0
    print("❌ 有偏差，请检查")
    return 1


if __name__ == "__main__":
    sys.exit(main())
