#!/usr/bin/env python3
"""⭐ Q1 核心交付：把 MuJoCo 的 (ω, τ) 工作点叠在官方实测 τ-ω 曲线上。

为什么这张图重要
----------------
"模型对不对"用一列数字说不清，但**一张叠加图能一眼看出**：
仿真点是否落在官方曲线上、偏差朝哪个方向、在哪个区间最大。

这是 Q1（执行器级验证）的核心产出 —— 也是评估链路里唯一有真值的一环。

输出
----
- `data/tau_omega_overlay.png`  —— 叠加图
- `data/tau_omega_overlay.json`  —— 同数据的机器可读版

用法
----
    /home/zhan/UniLab/.venv/bin/python tools/plot_tau_omega.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def simulate_curve(V: float, TAU: float, W: float, loads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """跑 MuJoCo，返回每个负载下的 (ω, τ)。"""
    import mujoco

    xml = f"""
    <mujoco>
      <option integrator="implicitfast" gravity="0 0 0"/>
      <worldbody><body name="b">
        <joint name="j" type="hinge" damping="0"/>
        <geom type="capsule" size=".05" fromto="0 0 0 0 0 .3" density="500"/>
      </body></worldbody>
      <actuator><dcmotor name="m1" joint="j" nominal="{V} {TAU} {W}" input="voltage"/></actuator>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    ws, taus = [], []
    for load in loads:
        data.qpos[0] = 0.0
        data.qvel[0] = 0.0
        data.qfrc_applied[0] = 0.0
        mujoco.mj_forward(model, data)
        for i in range(200_000):
            data.ctrl[0] = V
            data.qfrc_applied[0] = -float(load)
            mujoco.mj_step(model, data)
            if i > 20_000 and i % 2000 == 0 and abs(data.qacc[0]) < 1e-9:
                break
        ws.append(float(data.qvel[0]))
        taus.append(float(data.actuator_force[0]))
    return np.array(ws), np.array(taus)


def main() -> int:
    try:
        import mujoco  # noqa: F401
    except ImportError:
        print("❌ 需要 mujoco。试：/home/zhan/UniLab/.venv/bin/python 跑本脚本")
        return 1
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 中文字形：系统自带 Noto Sans CJK SC（否则中文会显示成方框）
        plt.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("❌ 需要 matplotlib")
        return 1

    p = yaml.safe_load((ROOT / "params" / "dm_j4340_2ec.yaml").read_text(encoding="utf-8"))
    V, TAU, W = (float(x) for x in p["mujoco_dcmotor"]["nominal"].split())
    TAU_DS = p["torque_speed"]["isaac_lab_dcmotor"]["variant_datasheet"]["stall_torque_Nm"]

    ref = json.loads((ROOT / "data" / "torque_speed_curve.json").read_text(encoding="utf-8"))
    ct = np.array([r["torque_Nm"] for r in ref])
    cw = np.array([r["omega_rad_s"] for r in ref])

    # 在实测区取负载点
    loads = np.linspace(ct.min(), ct.max(), 25)
    sim_w, sim_tau = simulate_curve(V, TAU, W, loads)

    rel = np.abs(sim_w - cw[np.searchsorted(ct, sim_tau).clip(0, len(ct) - 1)]) / cw[
        np.searchsorted(ct, sim_tau).clip(0, len(ct) - 1)
    ] * 100

    # 模型的两条候选直线
    tt = np.linspace(0, 60, 200)
    line_used = W * (1 - tt / TAU)
    line_ds = W * (1 - tt / TAU_DS)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(ct, cw, "o", ms=4, color="#1f77b4", alpha=0.7,
            label=f"官方台架实测（{len(ct)} 点，像素提取 ±1~2%）")
    ax.plot(tt, line_used, "-", lw=2, color="#2ca02c",
            label=f"MuJoCo <dcmotor>  nominal=24 {TAU:g} {W:.4f}  （本次采用）")
    ax.plot(tt, line_ds, "--", lw=1.6, color="#d62728", alpha=0.75,
            label=f"若用说明书标称 stall={TAU_DS:g}  （偏差最大 17%）")
    ax.plot(sim_tau, sim_w, "x", ms=7, mew=1.8, color="#ff7f0e",
            label="MuJoCo 仿真工作点（加载扫描）")

    ax.axvspan(ct.min(), ct.max(), color="gray", alpha=0.08)
    ax.text(ct.max() * 0.5, W * 0.93, "官方实测覆盖区", ha="center",
            fontsize=9, color="gray")

    ax.set_xlabel("扭矩 τ  [N·m]")
    ax.set_ylabel("角速度 ω  [rad/s]")
    ax.set_title("DM-J4340-2EC (24V) τ-ω：仿真模型 vs 官方台架实测")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_xlim(0, 58)
    ax.set_ylim(0, W * 1.08)

    out_png = ROOT / "data" / "tau_omega_overlay.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)

    print(f"图已存: {out_png}")
    print(f"仿真点 vs 实测曲线：平均 {np.mean(rel):.2f}%  最大 {np.max(rel):.2f}%")
    print(f"（用 stall={TAU:g}；若用 {TAU_DS:g} 平均误差会到 9.94%，最大 17.00%）")

    (ROOT / "data" / "tau_omega_overlay.json").write_text(
        json.dumps({
            "nominal": [V, TAU, W],
            "stall_used": TAU,
            "stall_datasheet": TAU_DS,
            "sim_points": [{"load_Nm": float(l), "omega": w, "tau": t}
                           for l, w, t in zip(loads, sim_w, sim_tau)],
            "mean_err_pct_vs_measured": float(np.mean(rel)),
            "max_err_pct_vs_measured": float(np.max(rel)),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
