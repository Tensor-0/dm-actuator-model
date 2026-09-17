#!/usr/bin/env python3
"""从达妙官方性能曲线 PNG 提取电机特性数据。

为什么需要这个脚本
------------------
达妙官方只发布了 **图片**（PNG），没有原始 CSV。要把曲线用起来
（填进仿真、做参数辨识），必须先把它数字化。

脚本做三件事：
  1. 用刻度标签行做**最小二乘标定**，把像素坐标换成物理量
  2. 按颜色分离各条曲线，逐列取中位像素 → 物理量序列
  3. 输出 JSON

⚠️ 为什么不用目测读数
------------------
目测 1000x600 的图，一条曲线只能读 5~10 个点，误差 ±2~5%。
本脚本逐列（每 5 px）采样，得到约 100 个点，且标定用全部刻度做回归
（不是"取两个端点画直线"）——这也是为什么能发现两条温升曲线逐像素相同。

数据源
------
gitee.com/kit-miao/DM-J4340-2EC
  └── 测试数据/性能曲线/V1.1/
        ├── 扭矩效率.png     ← 真 τ-ω 曲线（变速加载）
        ├── 速度效率.png     ← 恒 35rpm 台架（扫扭矩）
        ├── 温升.png         ← 12 N·m 温升
        └── 温升14Nm.png     ← ⚠️ 与上图曲线逐像素相同（仅标题不同）

用法
----
    python3 extract_curve.py --img-dir /path/to/pngs --out ../data/

依赖: numpy, Pillow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# 图像常量 —— 都来自对官方 PNG 的实测（1000x600）
# ---------------------------------------------------------------------------

# 绘图区（数据区域）的像素边界
PLOT_X0, PLOT_X1 = 200, 800
PLOT_Y0, PLOT_Y1 = 72, 534

# x 轴（扭矩）标定：x=211 对应 2.5 N·m，每 77 px 一个 2.5 N·m 刻度
TORQUE_X_REF = 211.0
TORQUE_X_STEP = 77.0
TORQUE_VAL_REF = 2.5
TORQUE_VAL_STEP = 2.5

# 曲线颜色（RGB）—— 从图中量化统计得来，各色像素数均 > 800
COLORS = {
    "eff": (240, 0, 0),      # 红 = Efft(%)
    "speed": (0, 0, 240),    # 蓝 = Speed(rpm)
    "vbus": (0, 96, 0),      # 绿 = Vbus(V)
    "pin": (144, 0, 144),    # 紫 = Pin(W)
    "pout": (144, 144, 0),   # 黄 = Pout(W)
    "ibus": (0, 180, 180),   # 青 = Ibus(A)
}


def torque_of(x_px: float) -> float:
    """像素列 → 扭矩 [N·m]。"""
    return TORQUE_VAL_REF + (x_px - TORQUE_X_REF) / TORQUE_X_STEP * TORQUE_VAL_STEP


def mask_for(arr: np.ndarray, color: Sequence[int], tol: int = 60) -> np.ndarray:
    """按颜色取布尔掩码（容差吸收抗锯齿边缘）。"""
    r, g, b = color
    return (
        (np.abs(arr[:, :, 0].astype(int) - r) < tol)
        & (np.abs(arr[:, :, 1].astype(int) - g) < tol)
        & (np.abs(arr[:, :, 2].astype(int) - b) < tol)
    )


def trace(arr: np.ndarray, color: Sequence[int], tol: int = 60, step: int = 5):
    """逐列取曲线像素的**中位** y —— 比取均值更抗粗线/抗锯齿。"""
    m = mask_for(arr, color, tol)
    out = []
    for x in range(PLOT_X0 + 5, PLOT_X1 + 1, step):
        ys = np.where(m[:, x])[0]
        if len(ys):
            out.append((x, float(np.median(ys))))
    return out


def fit_linear(ys: Iterable[float], vals: Iterable[float]) -> tuple[float, float]:
    """最小二乘 y_px → 物理值。返回 (slope, intercept)。"""
    y = np.asarray(list(ys), dtype=float)
    v = np.asarray(list(vals), dtype=float)
    A = np.vstack([y, np.ones_like(y)]).T
    slope, intercept = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(slope), float(intercept)


def find_label_rows(arr, x0: int, x1: int, color, min_px: int = 2, gap: int = 5):
    """在给定横向条带内找同色文字的行分组 → 刻度标签的行中心。

    刻度标签是彩色文字，同一标签占若干连续行；按 gap 合并成组。

    ⚠️ 条带选不好会把**轴标题文字**（如竖排的 "Efft(%)"）也框进来，
    产生多余的"组"。调用方应选一个只覆盖刻度数字的窄条带，并核对
    返回的组数 == 预期刻度数。
    """
    m = mask_for(arr, color, tol=60)[:, x0:x1]
    rows = m.sum(axis=1)
    ys = [y for y in range(len(rows)) if rows[y] > min_px]
    groups: list[list[int]] = []
    for y in ys:
        if groups and y - groups[-1][-1] <= gap:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [int(np.mean(g)) for g in groups]


def _longest_even_run(cand: Sequence[int], step: float, tol: float = 10.0) -> list[int]:
    """给定步长，取最长的、落在同一等差栅格上的候选序列。

    允许**缺格**（倍数 1,2,3...）——图边缘的刻度常被坐标轴盖住检测不到。
    但只收录真正落在栅格上的候选，所以离群文字（接不上栅格）会被剔除。
    """
    cand = sorted(cand)
    best: list[int] = []
    for a in cand:
        run = [a]
        for b in cand:
            if b <= run[-1]:
                continue
            ratio = (b - run[-1]) / step
            if ratio >= 0.7 and abs(ratio - round(ratio)) * step <= tol:
                run.append(b)
        if len(run) > len(best):
            best = run
    return best




# ---------------------------------------------------------------------------
# 各图表的标定（刻度读数值来自人工核对放大图，行位置由脚本自动定位）
# ---------------------------------------------------------------------------

def calibrate_torque_eff(arr) -> dict[str, tuple[float, float]]:
    """扭矩效率.png 的三条右轴：Efft / Pin / Pout。

    刻度值由人工读放大图确认（易错点：Efft 轴是 20~60，不是到 70；
    70 是 Pout 轴的顶。曾因看错轴把效率峰值算成 78%，实际 66%）。
    """
    eff = fit_linear([150.0, 256.2, 358.8, 465.0, 567.5], [60, 50, 40, 30, 20])
    pin = fit_linear([141.7, 217.0, 292.0, 368.0, 443.0, 518.0], [120, 100, 80, 60, 40, 20])
    pout = fit_linear([83.3, 144.0, 205.0, 266.0, 327.0, 388.0, 449.0, 510.0],
                      [70, 60, 50, 40, 30, 20, 10, 0])
    return {"eff": eff, "pin": pin, "pout": pout}


def calibrate_axis_from_top(arr, x0: int, x1: int, color, top_value: float,
                            step_value: float, n_max: int, min_ticks: int = 4):
    """通用轴标定：从图顶部开始，按固定步长**向下递减**赋刻度值。

    返回 (slope, intercept, rows)。数值自顶向下为
    `top_value, top_value-step_value, ...`。

    ⚠️ 关键设计：**候选里混着竖排轴标题文字**（如 "motor_temp(°C)"），
    它不是刻度。做法是先用 RANSAC 找最长等距串（轴标题接不上等差数列，
    会被剔除），再用回归步长吸附到最近的候选——只取**真实存在**的候选，
    不凭空造刻度。
    """
    cand = sorted(set(find_label_rows(arr, x0, x1, color)))
    if len(cand) < min_ticks:
        raise RuntimeError(f"轴刻度候选不足: {cand}")

    # 1) 用两两候选定步长，取最长等距串
    best: list[int] = []
    for i in range(len(cand)):
        for j in range(i + 1, min(len(cand), i + 5)):
            step = (cand[j] - cand[i]) / (j - i)
            if step < 8:
                continue
            run = _longest_even_run(cand, step)
            if len(run) > len(best):
                best = run
    if len(best) < min_ticks:
        raise RuntimeError(f"未找到等距刻度串，候选={cand}")

    # 2) 回归出平均步长（用索引做自变量，稳健）
    idx = np.arange(len(best), dtype=float)
    slope_px, intercept_px = np.polyfit(idx, np.asarray(best, float), 1)
    if slope_px <= 0:
        raise RuntimeError("刻度步长非正")

    # 3) 铺栅格并吸附到真实候选（不造点）
    cands = np.asarray(cand, float)
    rows = []
    for k in range(n_max):
        target = best[0] + k * slope_px
        if target > PLOT_Y1:          # 不超出绘图区
            break
        near = cands[np.abs(cands - target) <= 6]
        if len(near):
            rows.append(int(round(float(near[np.argmin(np.abs(near - target))]))))
    if len(rows) < min_ticks:
        raise RuntimeError(f"吸附后刻度不足: {rows}")

    vals = [top_value - i * step_value for i in range(len(rows))]
    slope, intercept = fit_linear(rows, vals)
    return slope, intercept, rows


# ---------------------------------------------------------------------------
# 提取
# ---------------------------------------------------------------------------

def extract_torque_speed(img_path: Path) -> list[dict]:
    """扭矩效率.png 的**蓝线** → 真 τ-ω 曲线（这是建模最需要的一条）。

    ⚠️ 不要用 `pout/torque` 反推 ω 来代替本函数——那条路在低扭矩端会
    塌掉（Pout→0 时商值失真）。蓝线是直接测的转速，才是正确来源。
    """
    arr = np.array(Image.open(img_path).convert("RGB"))
    slope, intercept, _ = calibrate_axis_from_top(
        arr, 175, 205, COLORS["speed"], top_value=50, step_value=10, n_max=6)
    rows = []
    for x, y in trace(arr, COLORS["speed"]):
        t = torque_of(x)
        if not (3.0 < t < 19.5):
            continue
        rpm = slope * y + intercept
        rows.append({
            "torque_Nm": round(t, 3),
            "speed_rpm": round(rpm, 3),
            "omega_rad_s": round(rpm * 2 * np.pi / 60, 4),
        })
    return rows


def extract_torque_eff(img_path: Path) -> list[dict]:
    """扭矩效率.png → τ-ω / 效率 / 功率 数据点。"""
    arr = np.array(Image.open(img_path).convert("RGB"))
    cal = calibrate_torque_eff(arr)

    red = trace(arr, COLORS["eff"])
    purple = trace(arr, COLORS["pin"])
    yellow = trace(arr, COLORS["pout"])

    def nearest(series, x, tol=4):
        for px, py in series:
            if abs(px - x) < tol:
                return py
        return None

    rows = []
    for x, y in red:
        t = torque_of(x)
        if not (3.0 < t < 19.5):      # 只取自洽区间（避开曲线起止的轴相交段）
            continue
        pin_y, pout_y = nearest(purple, x), nearest(yellow, x)
        if pin_y is None or pout_y is None:
            continue
        eff = cal["eff"][0] * y + cal["eff"][1]
        pin = cal["pin"][0] * pin_y + cal["pin"][1]
        pout = cal["pout"][0] * pout_y + cal["pout"][1]
        rows.append({
            "torque_Nm": round(t, 3),
            "eff_pct": round(eff, 2),
            "pin_W": round(pin, 2),
            "pout_W": round(pout, 2),
            # 注意: 不在此处放 omega —— 请用 torque_speed_curve.json（蓝线直测）
        })
    return rows


def extract_speed(img_path: Path) -> list[dict]:
    """速度效率.png → 恒转速台架的 转速 数据（用于验证它是恒速协议）。"""
    arr = np.array(Image.open(img_path).convert("RGB"))
    slope, intercept, _ = calibrate_axis_from_top(arr, 175, 205, COLORS["speed"],
                                                   top_value=35, step_value=5, n_max=8)
    rows = []
    for x, y in trace(arr, COLORS["speed"]):
        t = torque_of(x)
        if not (0.5 < t < 21.0):
            continue
        rows.append({"torque_Nm": round(t, 3), "speed_rpm": round(slope * y + intercept, 3)})
    return rows


def extract_thermal(img_path: Path) -> list[dict]:
    """温升图 → (t, T) 序列。x 轴 0~600 s。"""
    arr = np.array(Image.open(img_path).convert("RGB"))
    slope, intercept, _ = calibrate_axis_from_top(arr, 95, 205, (240, 0, 0),
                                                   top_value=110, step_value=10, n_max=9)
    rows = []
    for x, y in trace(arr, (240, 0, 0), tol=60, step=10):
        # x 轴: PLOT_X0 -> 0 s, PLOT_X1 -> 600 s
        t = (x - PLOT_X0) / (PLOT_X1 - PLOT_X0) * 600
        if t < 0 or t > 600:
            continue
        rows.append({"t_s": round(t, 1), "temp_C": round(slope * y + intercept, 2)})
    return rows


def images_identical(p1: Path, p2: Path) -> dict:
    """逐像素比对两张图，判断是否只是副本（区分"标题不同"与"数据不同"）。"""
    a, b = (np.array(Image.open(p).convert("RGB")).astype(int) for p in (p1, p2))
    if a.shape != b.shape:
        return {"same_shape": False}
    diff = np.abs(a - b).sum(axis=2) > 10
    ys, xs = np.where(diff)
    in_plot = int(((ys >= PLOT_Y0) & (ys <= PLOT_Y1)
                   & (xs >= PLOT_X0) & (xs <= PLOT_X1)).sum()) if diff.any() else 0
    return {
        "same_shape": True,
        "diff_pixels": int(diff.sum()),
        "diff_in_plot": in_plot,
        "verdict": "绘图区完全相同（副本，仅标题不同）" if in_plot == 0 else "绘图区有差异",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="从达妙官方性能曲线 PNG 提取数据")
    ap.add_argument("--img-dir", required=True, type=Path, help="含 PNG 的目录")
    ap.add_argument("--out", required=True, type=Path, help="输出 JSON 目录")
    args = ap.parse_args()

    d, out = args.img_dir, args.out
    out.mkdir(parents=True, exist_ok=True)

    ts = extract_torque_speed(d / "扭矩效率.png")
    (out / "torque_speed_curve.json").write_text(
        json.dumps(ts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"τ-ω 曲线   : {len(ts)} 点")

    curve = extract_torque_eff(d / "扭矩效率.png")
    (out / "torque_eff_curve.json").write_text(
        json.dumps(curve, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"τ-ω/效率/功率: {len(curve)} 点")

    speed = extract_speed(d / "速度效率.png")
    (out / "speed_const_rpm.json").write_text(
        json.dumps(speed, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"恒转速台架  : {len(speed)} 点")

    thermal = extract_thermal(d / "温升.png")
    (out / "thermal.json").write_text(
        json.dumps(thermal, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"温升        : {len(thermal)} 点")

    cmp = images_identical(d / "温升.png", d / "温升14Nm.png")
    print(f"温升图比对  : {cmp}")


if __name__ == "__main__":
    main()
