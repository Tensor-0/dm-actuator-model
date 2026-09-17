#!/usr/bin/env python3
"""把提取出的曲线数据拟合成可用的电机模型参数。

输出三组参数：
  1. **τ-ω 线性模型** —— 对齐 Isaac Lab `DCMotor` 的字段命名
     `τ_max(q̇) = clip(τ_stall·(1 − q̇/q̇_max), −∞, τ_con)`
  2. **效率模型** —— 峰值/额定/轻载三点（Isaac Lab 没建这一项）
  3. **热模型** —— 一阶 `T(t) = T_inf − (T_inf−T0)·e^(−t/τ)`

⚠️ 关于 stall_torque 的重要取舍
--------------------------------
Isaac Lab 的 τ-ω 直线由 **(q̇_max, τ_stall) 两点定义**，斜率 = τ_stall/q̇_max。
也就是说 **q̇_max 和 τ_stall 不能各自独立取值**——给定 q̇_max 后，
选定的 τ_stall 就唯一确定了斜率。

本机实测斜率是 −0.1186 rad/s per N·m，若配 τ_stall=40（说明书标称）会得到
斜率 40/6.40 = 6.25 N·m/(rad/s)，与实测的 1/0.1186 = 8.43 **相差 35%**。
所以本脚本**同时输出两套**，由使用方按用途选：

  - `datasheet`：τ_stall=40（标称），斜率被改写 → 保守，不会高估高速段能力
  - `measured` ：τ_stall=54（实测外推），斜率忠于曲线 → 忠于本次数据

两者都在**实测区间内**表现接近（差异主要在高速外推段）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_tau_omega(rows: list[dict]) -> dict:
    """线性拟合 τ-ω，并给出 Isaac Lab DCMotor 需要的四个量。"""
    t = np.array([r["torque_Nm"] for r in rows], dtype=float)
    w = np.array([r["omega_rad_s"] for r in rows], dtype=float)
    slope, intercept = np.polyfit(t, w, 1)
    r = float(np.corrcoef(t, w)[0, 1])

    no_load = float(intercept)                  # τ=0 时的 ω
    stall_meas = float(-intercept / slope)      # ω=0 时的 τ（外推！）

    return {
        "omega_per_torque": float(slope),       # dω/dτ  [rad/s per N·m]
        "no_load_speed_rad_s": no_load,
        "no_load_speed_rpm": no_load * 60 / (2 * np.pi),
        "stall_torque_measured_Nm": stall_meas,
        "r_squared": r ** 2,
        "pearson_r": r,
        "n_points": len(rows),
        "measured_range": {
            "torque_Nm": [float(t.min()), float(t.max())],
            "omega_rad_s": [float(w.min()), float(w.max())],
        },
    }


def efficiency_summary(rows: list[dict]) -> dict:
    """效率三点 + 峰值位置。"""
    t = np.array([r["torque_Nm"] for r in rows], dtype=float)
    e = np.array([r["eff_pct"] for r in rows], dtype=float)
    i = int(np.argmax(e))

    def at(target):
        j = int(np.argmin(np.abs(t - target)))
        return {"torque_Nm": float(t[j]), "eff_pct": float(e[j])}

    return {
        "peak": {"torque_Nm": float(t[i]), "eff_pct": float(e[i])},
        "at_rated_12Nm": at(12.0),
        "at_low_load_3_3Nm": at(3.3),
        "note": "轻载效率很低（<35%）；这是准直驱电机小扭矩工况的固有特性",
    }


def fit_thermal(rows: list[dict]) -> dict:
    """一阶热模型拟合。"""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return {"error": "需要 scipy"}

    t = np.array([r["t_s"] for r in rows], dtype=float)
    T = np.array([r["temp_C"] for r in rows], dtype=float)

    def model(t, t_inf, t0, tau):
        return t_inf - (t_inf - t0) * np.exp(-t / tau)

    p, _ = curve_fit(model, t, T, p0=[105.0, 50.0, 250.0], maxfev=40000)
    resid = float(np.sqrt(np.mean((model(t, *p) - T) ** 2)))
    return {
        "T_inf_C": float(p[0]),
        "T_0_C": float(p[1]),
        "tau_s": float(p[2]),
        "rms_residual_C": resid,
        "n_points": len(rows),
        "caveat": "单一工作点（12 N·m 恒扭矩）的温升，未做功率相关性标定",
    }


def corner_velocity(no_load: float, tau_stall: float, tau_con: float) -> float:
    """连续扭矩线与 τ-ω 直线的交点速度。

    由 Isaac Lab 的模型式反解：
        tau_max(qd) = tau_stall * (1 - qd/qd_max)
    令 tau_max = tau_con 得
        qd_c = qd_max * (1 - tau_con/tau_stall)

    ⚠️ 该值**依赖 tau_stall**，所以两套变体的 corner_velocity 不同——
    不是同一个数。早期版本错误地用了 (tau_con - b)/slope，那会算出负值。
    """
    return float(no_load * (1.0 - tau_con / tau_stall))


def main() -> None:
    ap = argparse.ArgumentParser(description="拟合电机模型参数")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tau-con", type=float, default=12.0, help="连续扭矩 [N·m]")
    ap.add_argument("--tau-stall-datasheet", type=float, default=40.0,
                    help="说明书标称峰值扭矩 [N·m]")
    args = ap.parse_args()

    d = args.data
    ts = json.loads((d / "torque_speed_curve.json").read_text(encoding="utf-8"))
    te = json.loads((d / "torque_eff_curve.json").read_text(encoding="utf-8"))
    th = json.loads((d / "thermal.json").read_text(encoding="utf-8"))

    tau_omega = fit_tau_omega(ts)
    eff = efficiency_summary(te)
    therm = fit_thermal(th)

    s, b = tau_omega["omega_per_torque"], tau_omega["no_load_speed_rad_s"]
    result = {
        "tau_omega": tau_omega,
        "tau_omega_model": {
            "equation_rad_s": f"omega = {s:.6f}*tau + {b:.6f}",
            "equation_rpm": f"rpm = {s*60/(2*np.pi):.6f}*tau + {b*60/(2*np.pi):.6f}",
            "valid_range": tau_omega["measured_range"],
        },
        "isaac_lab_dcmotor": {
            "_comment": "两套互斥：τ-ω 直线的斜率由 (no_load_speed, stall_torque) 决定，不能各自独立取",
            "no_load_speed_rad_s": b,
            "continuous_torque_Nm": args.tau_con,
            "variant_datasheet": {
                "stall_torque_Nm": args.tau_stall_datasheet,
                "implied_slope_rad_s_per_Nm": -args.tau_stall_datasheet / b,
                "corner_velocity_rad_s": corner_velocity(b, args.tau_stall_datasheet, args.tau_con),
                "provenance": "说明书标称值（未实测）",
            },
            "variant_measured": {
                "stall_torque_Nm": tau_omega["stall_torque_measured_Nm"],
                "implied_slope_rad_s_per_Nm": s,
                "corner_velocity_rad_s": corner_velocity(b, tau_omega["stall_torque_measured_Nm"], args.tau_con),
                "provenance": "⚠️ 由曲线线性外推，超出实测区（实测只到 19.35 N·m）",
            },
        },
        "efficiency": eff,
        "thermal": therm,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"τ-ω: ω = {s:.6f}·τ + {b:.6f} rad/s  (r²={tau_omega['r_squared']:.4f})")
    print(f"  空载转速 {tau_omega['no_load_speed_rpm']:.2f} rpm")
    print(f"  峰值效率 {eff['peak']['eff_pct']:.2f}% @ {eff['peak']['torque_Nm']:.2f} N·m")
    print(f"  热 tau={therm.get('tau_s', 0):.1f}s  T_inf={therm.get('T_inf_C', 0):.1f}°C")
    print(f"→ 写入 {args.out}")


if __name__ == "__main__":
    main()
