#!/usr/bin/env python3
"""自检脚本：验证 params/ 里的派生量确实与 data/ 一致。

为什么要这个
------------
参数文件里有很多**派生量**（隐含斜率、拐点速度、外推堵转扭矩）。
这些量最容易在手抄时出错——本项目就真的错过一次：
`corner_velocity` 曾用错公式算出负值，`implied_slope` 曾差 0.001。

本脚本把「YAML 声明值」与「从 data/ 重算的值」逐项比对，
不通过就报错退出。改了参数请务必跑一遍。

    python3 tools/verify_params.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
TOL = 0.01          # 相对容差

failures: list[str] = []
checks = 0


def check(label: str, yaml_val: float, calc_val: float, tol: float = TOL) -> None:
    global checks
    checks += 1
    if yaml_val is None or calc_val is None:
        failures.append(f"{label}: 缺值 yaml={yaml_val} calc={calc_val}")
        return
    if abs(yaml_val - calc_val) > tol * max(1.0, abs(calc_val)):
        failures.append(f"{label}: yaml={yaml_val} 但重算={calc_val} (容差 {tol})")
    else:
        print(f"  ✅ {label}: {calc_val}")


def main() -> int:
    yml = yaml.safe_load((ROOT / "params" / "dm_j4340_2ec.yaml").read_text(encoding="utf-8"))
    ts = json.loads((ROOT / "data" / "torque_speed_curve.json").read_text(encoding="utf-8"))
    te = json.loads((ROOT / "data" / "torque_eff_curve.json").read_text(encoding="utf-8"))
    th = json.loads((ROOT / "data" / "thermal.json").read_text(encoding="utf-8"))

    # ---- τ-ω ----
    t = np.array([r["torque_Nm"] for r in ts])
    w = np.array([r["omega_rad_s"] for r in ts])
    slope, intercept = np.polyfit(t, w, 1)
    r = float(np.corrcoef(t, w)[0, 1])

    print("τ-ω 模型")
    tsy = yml["torque_speed"]
    il = tsy["isaac_lab_dcmotor"]
    check("no_load_speed_rad_s", il["no_load_speed_rad_s"], intercept)
    assert f"{slope:.6f}" in tsy["equation_rad_s"], f"方程串不符: {tsy['equation_rad_s']}"

    # 实测范围
    mr = tsy["measured_range"]
    check("range.torque_min", mr["torque_Nm"][0], float(t.min()))
    check("range.torque_max", mr["torque_Nm"][1], float(t.max()))
    check("range.omega_min", mr["omega_rad_s"][0], float(w.min()))
    check("range.omega_max", mr["omega_rad_s"][1], float(w.max()))

    # Isaac Lab 两套变体
    tau_con = il["continuous_torque_Nm"]
    for name, tau_stall_key in [("variant_datasheet", "stall_torque_Nm"),
                                ("variant_measured", "stall_torque_Nm")]:
        v = il[name]
        ts_tau = v[tau_stall_key]
        # 该变体下的直线斜率由 (no_load, tau_stall) 定
        #   datasheet 变体 → 用 tau_stall/no_load
        #   measured  变体 → 用实测斜率
        if name == "variant_datasheet":
            implied = ts_tau / intercept
            check(f"{name}.implied_slope", v["implied_slope_Nm_per_rad_s"], implied)
        # 拐点速度 = qd_max*(1 - tau_con/tau_stall)
        cv = intercept * (1.0 - tau_con / ts_tau)
        check(f"{name}.corner_velocity", v["corner_velocity_rad_s"], cv)

    # 外推堵转（仅 measured 变体应等于该外推值）
    stall_extrap = -intercept / slope
    check("measured.stall(外推)", il["variant_measured"]["stall_torque_Nm"], stall_extrap, tol=0.05)

    # ---- 效率 ----
    print("效率")
    te_t = np.array([r["torque_Nm"] for r in te])
    te_e = np.array([r["eff_pct"] for r in te])
    i = int(np.argmax(te_e))
    check("peak.eff_pct", yml["efficiency"]["peak"]["eff_pct"], float(te_e[i]))
    check("peak.torque_Nm", yml["efficiency"]["peak"]["torque_Nm"], float(te_t[i]))
    j = int(np.argmin(np.abs(te_t - 12.0)))
    check("at_rated.eff_pct", yml["efficiency"]["at_rated"]["eff_pct"], float(te_e[j]))

    # ---- 热 ----
    print("热模型")
    try:
        from scipy.optimize import curve_fit
        th_t = np.array([r["t_s"] for r in th])
        th_T = np.array([r["temp_C"] for r in th])

        def model(t_, t_inf, t0, tau):
            return t_inf - (t_inf - t0) * np.exp(-t_ / tau)

        p, _ = curve_fit(model, th_t, th_T, p0=[105.0, 50.0, 250.0], maxfev=40000)
        check("T_inf_C", yml["thermal"]["T_inf_C"], float(p[0]))
        check("tau_s", yml["thermal"]["tau_s"], float(p[2]))
    except ImportError:
        print("  ⚠️ 跳过（无 scipy）")

    print()
    print(f"τ-ω r² = {r**2:.4f}, {len(ts)} 点")
    if failures:
        print(f"❌ {len(failures)}/{checks} 项不一致：")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"✅ 全部 {checks} 项一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
