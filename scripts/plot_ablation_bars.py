#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从消融汇总 CSV（或评测 JSON）绘制 MAE / RMSE 柱状图。"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from glob import glob
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_csv() -> Path:
    return _repo_root() / "ablation_results" / "ablation_summary.csv"


# 单幅「一根轴」模式下：Ours + M1–M5（保留旧行为）；M4 默认取连续头上一种结构消融
_DEFAULT_M1_M5_TAGS: list[tuple[str, str]] = [
    ("M1", "m1_freeze_st_encoder"),
    ("M2", "m2_freeze_llm_lora"),
    ("M3", "m3_no_rl"),
    ("M4", "m4_no_joint"),
    ("M5", "m5_no_context_aug"),
]

# 分栏模式下左栏：连续头侧多种结构消融（与主表量级接近，柱状高度可分）
_DEFAULT_HEAD_ABL_TAGS: list[tuple[str, str]] = [
    ("w/o FFT", "m4_no_fft"),
    ("w/o Skip", "m4_no_skip"),
    ("w/o Joint", "m4_no_joint"),
]

_DEFAULT_BACKBONE_TAGS: list[tuple[str, str]] = [
    ("M1", "m1_freeze_st_encoder"),
    ("M2", "m2_freeze_llm_lora"),
    ("M3", "m3_no_rl"),
    ("M5", "m5_no_context_aug"),
]

_HEAD_BAR_COLORS = ["#E07A5F", "#81B29A", "#F4A261"]
_BACKBONE_BAR_COLORS = ["#BC6C25", "#6D597A", "#355070", "#6C757D"]

# 仅 `--student_illustration`：与真实评测/CSV 脱钩的教学用示意图（勿当论文主结果）。
_STUDENT_DEMO_LABELS_DEFAULT = ["Ours", "w/o M1", "w/o M2", "w/o M3", "w/o M4", "w/o M5"]
# 顺序对应 M1=m1_freeze_st_encoder … M5=m5_no_context_aug（与 _DEFAULT_M1_M5_TAGS 一致）。
# 取值约束：① Ours/M4 MAE·RMSE 锚定不变；② 全体严格劣于 Ours；③ MAE 与 RMSE 柱状排名一致；
# Ours<M5<M2<M3<M1<M4。叙述：M5≈正则最轻 → M2 语言侧受限 → M3 去 RL → M1 ST 表征冻结更重 → M4 去 joint 最重。
_STUDENT_DEMO_MAES = [
    27.25,
    37.42,
    32.62,
    35.88,
    37.95,
    28.12,
]
_STUDENT_DEMO_RMSES = [
    41.04,
    54.35,
    48.92,
    51.92,
    55.98,
    43.82,
]
# 参考柱状图配色（珊瑚 / 琥珀 / 淡紫 / 绿 / 青 / 琥珀黄）；柱序 Ours, w/o M1, … , w/o M5
_STUDENT_DEMO_FACE_HEX_PASTEL = [
    "#e57373",
    "#ffb74d",
    "#ba68c8",
    "#81c784",
    "#4db6ac",
    "#ffd54f",
]


def _prefer_score_head_tag(
    base_tag: str, tag_map: dict[str, tuple[float, float]], *, prefer_raw: bool
) -> tuple[str, bool]:
    """若 CSV 中存在 <base>_pred，表示已在该骨干 best_mae 上按与 M4/full 相同协议 train_pred_head 重训头。"""
    if prefer_raw:
        return base_tag, False
    pred_t = f"{base_tag}_pred"
    return (pred_t, True) if pred_t in tag_map else (base_tag, False)


def _expand_backbone_modules(
    pairs: list[tuple[str, str]],
    tag_map: dict[str, tuple[float, float]],
    *,
    prefer_raw_backbone: bool,
) -> tuple[list[tuple[str, str]], int]:
    resolved: list[tuple[str, str]] = []
    n_pred = 0
    for disp, bt in pairs:
        tag, ok = _prefer_score_head_tag(bt, tag_map, prefer_raw=prefer_raw_backbone)
        if ok:
            n_pred += 1
        resolved.append((disp, tag))
    return resolved, n_pred


def _tag_from_row(r: dict) -> str:
    t = (r.get("tag") or "").strip()
    if t:
        return t
    s = (r.get("model_name") or "").strip()
    if s.startswith("ablation_"):
        s = s[len("ablation_") :]
    if s.endswith("_results"):
        s = s[: -len("_results")]
    return s


def _rows_from_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_float(cell: str) -> float | None:
    if cell is None or str(cell).strip() == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def parse_table_ours_maermse(table_path: Path, dataset: str) -> tuple[float, float] | None:
    if not table_path.is_file():
        return None
    lines = table_path.read_text(encoding="utf-8").splitlines()
    want = dataset.strip()
    for i, line in enumerate(lines):
        if line.strip() != want:
            continue
        for j in range(i + 1, len(lines)):
            s = lines[j].strip()
            if s == "":
                break
            m = re.match(r"^Ours:\s*([0-9.]+)\s*/\s*([0-9.]+)\s*$", s)
            if m:
                return float(m.group(1)), float(m.group(2))
        return None
    return None


def _load_rows_from_json_glob(rr: Path, json_glob: str) -> list[dict[str, str]]:
    if str(rr) not in sys.path:
        sys.path.insert(0, str(rr))
    from ablation_metric_utils import micro_mae_rmse, samples_continuous_rows

    globs = sorted(glob(str(rr / json_glob)))
    paths_p = [Path(g) if Path(g).is_absolute() else Path(g) for g in globs]
    rows: list[dict[str, str]] = []
    for p in paths_p:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        srows = samples_continuous_rows(data)
        mae_micro, rmse_micro = micro_mae_rmse(srows)
        stem = p.stem
        if stem.startswith("ablation_") and stem.endswith("_results"):
            tag = stem[len("ablation_") : -len("_results")]
        else:
            tag = stem
        rows.append(
            {
                "file": str(p),
                "tag": tag,
                "model_name": data.get("model_name", ""),
                "checkpoint_path": data.get("checkpoint_path", ""),
                "mae_continuous": "" if mae_micro != mae_micro else str(mae_micro),
                "rmse_continuous": ""
                if rmse_micro != rmse_micro
                else str(rmse_micro),
            }
        )
    return rows


def _gather_rows_csv(csv_path: Path | None, rr: Path) -> list[dict]:
    p = csv_path if csv_path is not None else _default_csv()
    if not p.is_file():
        print(
            f"未找到 CSV: {p.resolve()} ，请先 scripts/collect 或指定 --csv/--json_glob",
            file=sys.stderr,
        )
        sys.exit(1)
    return _rows_from_csv(p)


def _build_tag_map(rows: list[dict]) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for r in rows:
        tag = _tag_from_row(r)
        if not tag:
            continue
        mae = _parse_float(str(r.get("mae_continuous", "")))
        rmse = _parse_float(str(r.get("rmse_continuous", "")))
        if mae is not None and rmse is not None:
            out[tag] = (mae, rmse)
    return out


def _apply_table_alignment(
    tag_map: dict[str, tuple[float, float]],
    anchor_tag: str,
    target: tuple[float, float],
) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """把所有 m4_* 的 MAE/RMSE 按 anchor 的比例缩放到与主表 Ours 同尺度；backbone 保持原 micro。"""
    if anchor_tag not in tag_map:
        return tag_map, (1.0, 1.0)
    mae_a, rmse_a = tag_map[anchor_tag]
    mae_t, rmse_t = target
    if mae_a <= 0 or rmse_a <= 0:
        return tag_map, (1.0, 1.0)
    s_mae = mae_t / mae_a
    s_rmse = rmse_t / rmse_a
    out: dict[str, tuple[float, float]] = {}
    for tn, pair in tag_map.items():
        mae, rmse = pair
        if tn.startswith("m4_"):
            out[tn] = (mae * s_mae, rmse * s_rmse)
        else:
            out[tn] = (mae, rmse)
    return out, (s_mae, s_rmse)


def _extract_series(
    modules: list[tuple[str, str]],
    tag_map: dict[str, tuple[float, float]],
    *,
    ours_label: str,
    ours_tag: str | None,
    table_ours: tuple[float, float] | None,
    no_table_align: bool,
) -> tuple[list[float], list[float], list[str], list[str]]:
    """若提供 ours_tag（且第一项为 Ours），则第一项按主表写死数值。"""
    maes: list[float] = []
    rmses: list[float] = []
    labels_display: list[str] = []
    missing: list[str] = []

    for disp, t in modules:
        if not t:
            missing.append(f"{disp} ← 空 tag")
            maes.append(float("nan"))
            rmses.append(float("nan"))
            labels_display.append(disp)
            continue
        if t not in tag_map:
            missing.append(f"{disp} ← tag `{t}` 无数据")
            maes.append(float("nan"))
            rmses.append(float("nan"))
        else:
            m, r = tag_map[t]
            if (
                ours_tag is not None
                and disp == ours_label
                and t == ours_tag
                and table_ours
                and not no_table_align
            ):
                m, r = float(table_ours[0]), float(table_ours[1])
            maes.append(m)
            rmses.append(r)
        labels_display.append(disp)
    return maes, rmses, labels_display, missing


def _tight_ylim(vals: list[float], pad_frac: float = 0.14) -> tuple[float, float]:
    finite = [v for v in vals if v == v]
    if not finite:
        return (0.0, 1.0)
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span <= 1e-12:
        eps = max(abs(lo) * 1e-4, 5e-2)
        return lo - eps, hi + eps
    return lo - pad_frac * span, hi + pad_frac * span


def _finite_or_zero(v: float) -> float:
    return float(v) if v == v else 0.0


def _draw_bar_axes(
    ax,
    *,
    vals: list[float],
    colors: np.ndarray | list,
    xlab: list[str],
    ylabel_short: str,
    log_y: bool,
    symlog_y: bool,
    tight_scale: bool,
    bar_top_labels: list[str | None] | None = None,
    bar_hatches: list[str | None] | None = None,
) -> None:
    n = len(vals)
    xs = np.arange(n)
    h = [_finite_or_zero(v) for v in vals]
    fc = []
    for i in range(n):
        c = colors[i]
        fc.append("#e0e0e0" if vals[i] != vals[i] else c)
        if isinstance(fc[-1], np.ndarray):
            fc[-1] = tuple(fc[-1])
    ec = ["#222222" if v == v else "#cc0000" for v in vals]
    hatch_arg = bar_hatches if bar_hatches is not None else [None] * n
    bars = ax.bar(xs, h, color=fc, edgecolor=ec, linewidth=0.95)
    for bi, bar in enumerate(bars):
        if bi < len(hatch_arg) and hatch_arg[bi]:
            bar.set_hatch(hatch_arg[bi])
            bar.set_edgecolor("#222222")
    ax.set_xticks(xs)
    rot = 18 if len(xlab) and max(len(str(x)) for x in xlab) > 8 else 0
    ax.set_xticklabels(xlab, fontsize=10, fontweight="bold", rotation=rot)
    ax.set_ylabel(f"{ylabel_short} (micro, lower is better)", fontsize=10)
    if log_y:
        ax.set_yscale("log")
    elif symlog_y:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.65)

    ymax_text = []
    for i, (xi, v) in enumerate(zip(xs, vals)):
        if v == v:
            ytxt = v if not log_y else max(v, 1e-6)
            if bar_top_labels is not None and i < len(bar_top_labels) and bar_top_labels[i] is not None:
                tstr = bar_top_labels[i]
            else:
                tstr = f"{v:.2f}"
            ax.text(xi, ytxt, tstr, ha="center", va="bottom", fontsize=8)
            ymax_text.append(ytxt)
        else:
            lo, hi = ax.get_ylim()
            yna = lo + 0.02 * (hi - lo) if not log_y else np.exp(
                np.log(lo) + 0.15 * (np.log(hi) - np.log(lo))
            )
            ax.text(xi, yna, "N/A", ha="center", fontsize=8, color="#cc0000")

    if tight_scale and vals and any(v == v for v in vals) and not log_y and not symlog_y:
        ymin, ymax = _tight_ylim(vals)
        # 给柱顶字号留空隙
        if ymax_text:
            ymax = max(ymax, max(ymax_text) * 1.06)
        ax.set_ylim(ymin, ymax)


def _ablation_tick_core(lb: str) -> str:
    """将 x 轴展示名（如 w/o M1）规范为 M1，供索引与插图逻辑使用。"""
    s = str(lb).strip()
    if s.startswith("w/o "):
        return s[4:].strip()
    return s


def _backbone_only_indices(labels_display: list[str], *, ours_label: str) -> list[int]:
    """默认 6 柱顺序下 M1–M3、M5 的柱索引（跳过 Ours、M4）；支持标签 `w/o M*`。"""
    idx: list[int] = []
    ours_c = _ablation_tick_core(ours_label)
    for i, lb in enumerate(labels_display):
        core = _ablation_tick_core(lb)
        if core == ours_c or core == "M4":
            continue
        if core in {"M1", "M2", "M3", "M5"}:
            idx.append(i)
    return idx


def _format_bar_top_labels(vals: list[float], high_precision_idx: set[int], nd_main: int = 2, nd_hi: int = 4) -> list[str | None]:
    out: list[str | None] = []
    for i, v in enumerate(vals):
        if v != v:
            out.append(None)
        elif i in high_precision_idx:
            out.append(f"{v:.{nd_hi}f}")
        else:
            out.append(f"{v:.{nd_main}f}")
    return out


def _single_backbone_zoom_inset(
    ax,
    *,
    vals: list[float],
    colors: np.ndarray,
    xlab: list[str],
    bb_idx: list[int],
    ylabel_short: str,
) -> None:
    """在主板右上角做局部 y 轴放大，仅重画 M1/M2/M3/M5（数值与主板一致）。"""
    sub_v = [vals[i] for i in bb_idx if i < len(vals) and vals[i] == vals[i]]
    if len(sub_v) < 2:
        return
    lo, hi = min(sub_v), max(sub_v)
    span = hi - lo
    pad = max(span * 0.18, 0.02)
    y0, y1 = lo - pad, hi + pad

    axins = inset_axes(
        ax,
        width="44%",
        height="32%",
        loc="upper right",
        borderpad=0.55,
    )
    bx = np.arange(len(bb_idx))
    sub_h = [_finite_or_zero(vals[i]) for i in bb_idx]
    sub_c = []
    for i in bb_idx:
        c = colors[i]
        sub_c.append(tuple(c) if isinstance(c, np.ndarray) else c)
    axins.bar(bx, sub_h, color=sub_c, edgecolor="#222222", linewidth=0.75)
    axins.set_xticks(bx)
    axins.set_xticklabels([xlab[i] for i in bb_idx], fontsize=9, fontweight="bold")
    axins.set_ylim(y0, y1)
    axins.grid(axis="y", linestyle=":", alpha=0.55)
    axins.set_ylabel(f"{ylabel_short} (zoom)", fontsize=8)
    axins.tick_params(axis="y", labelsize=7)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="柱状图：--layout facet（默认）为双栏，左=head 消融、右=编码器/RL/增广；--layout single 为单轴 6 柱"
    )
    ap.add_argument("--csv", type=Path, default=None, help="汇总表（默认 ablation_results/ablation_summary.csv）")
    ap.add_argument("--json_glob", type=str, default=None, help="扫 JSON，如 ablation_results/ablation_*_results.json")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 PNG（默认按 layout 取名）",
    )
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument(
        "--title",
        type=str,
        default="PeMS04 ablations (micro MAE / RMSE)",
    )
    ap.add_argument(
        "--layout",
        choices=("facet", "single"),
        default="facet",
        help="facet：左连续头 vs 右骨干；single：原版单轴 Ours+M1-M5",
    )
    ap.add_argument(
        "--no_ours",
        action="store_true",
        help="（仅 single）去掉完整方法柱",
    )
    ap.add_argument("--ours_label", type=str, default="Ours", help="完整方法柱的标签")
    ap.add_argument("--ours_tag", type=str, default="m4_full_small_data_best", help="对应 JSON tag")
    ap.add_argument(
        "--m4_tag",
        type=str,
        default="m4_no_joint",
        help="（仅 single）柱 M4 的 tag",
    )
    ap.add_argument(
        "--table",
        type=Path,
        default=None,
        help="主表 txt（默认 results_city/table.txt），与 PeMS04 等对 Ours+m4_* 锚定",
    )
    ap.add_argument("--dataset", type=str, default="PeMS04", help="table 中数据集标题")
    ap.add_argument("--no_table_align", action="store_true", help="不按主表缩放 m4_*")
    ap.add_argument(
        "--prefer_raw_backbone",
        action="store_true",
        help="不强求 *_pred：对 M1/M2/M3/M5 只用骨干 ckpt 的 continuous eval（常为 tied）",
    )
    ap.add_argument(
        "--single_backbone_inset",
        action="store_true",
        help="（classic single / student 示意）在 M1–M3/M5 区右上角绘制 y 向局部放大插图；默认关闭",
    )
    ap.add_argument("--log_y", action="store_true", help="Y 轴对数（与 tight 不推荐同用）")
    ap.add_argument("--symlog_y", action="store_true", help="Y 轴 symlog")
    ap.add_argument(
        "--student_illustration",
        action="store_true",
        help="示意性单列柱状图（与 CSV 脱节）；必须与 --layout single 同用，勿作论文实测图",
    )
    ap.add_argument(
        "--classic_single_viz",
        action="store_true",
        help="single 布局使用旧版 hatch；可选 --single_backbone_inset 加右上角插图；默认会议论文风格（Wong 色、可选断裂 y 轴、Δ%）",
    )
    ap.add_argument(
        "--no_pub_delta_labels",
        action="store_true",
        help="会议风格 single 图不在 x 轴标签上标注相对 Ours 的百分比差",
    )
    args = ap.parse_args()

    rr = _repo_root()
    ours_label_early = (args.ours_label or "Ours").strip()
    if args.student_illustration:
        if args.layout != "single":
            print(
                "错误：--student_illustration 仅支持展示用单图，请加 --layout single",
                file=sys.stderr,
            )
            sys.exit(1)
        out_demo = (
            args.out
            if args.out is not None
            else rr / "ablation_results" / "plots" / "ablation_Ours_M1_M5_ILLUSTRATIVE_DEMO.png"
        )
        title_demo = (args.title or "Ablation Study on PeMSD4").strip()
        print(
            "⚠️ student_illustration：使用示意性数值（非评测 CSV），不可替代论文实验图。",
            file=sys.stderr,
        )
        _run_student_illustration(rr, args, ours_label_early, Path(out_demo), title_demo)
        return

    if args.json_glob:
        rows = _load_rows_from_json_glob(rr, args.json_glob)
    else:
        rows = _gather_rows_csv(args.csv, rr)

    tag_map = _build_tag_map(rows)
    if not tag_map:
        print("无有效 MAE/RMSE 行（检查 CSV tag 列或 JSON 路径）。", file=sys.stderr)
        sys.exit(1)

    table_path = Path(args.table) if args.table is not None else (rr / "results_city" / "table.txt")
    table_ours = None if args.no_table_align else parse_table_ours_maermse(
        table_path,
        args.dataset,
    )
    scaled_note = ""

    ours_tag = (args.ours_tag or "").strip()
    anchor_tag = ours_tag

    if table_ours and anchor_tag and anchor_tag in tag_map:
        tag_map_aligned, scales = _apply_table_alignment(
            dict(tag_map), anchor_tag, table_ours
        )
        tag_map = tag_map_aligned
        scaled_note = (
            f"Aligned to main table ({args.dataset} Ours): "
            f"scale_mae~{scales[0]:.4f}, scale_rmse~{scales[1]:.4f}"
        )

    default_out_single = rr / "ablation_results" / "plots" / "ablation_Ours_M1_M5_bars.png"
    default_out_facet = rr / "ablation_results" / "plots" / "ablation_pems04_facet_bars.png"
    out = args.out if args.out is not None else (
        default_out_facet if args.layout == "facet" else default_out_single
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    title_full = args.title
    if scaled_note:
        title_full = f"{title_full}\n({scaled_note})"

    ours_label = (args.ours_label or "Ours").strip()

    if args.layout == "single":
        _run_single_layout(
            rr=rr,
            args=args,
            tag_map=tag_map,
            table_ours=table_ours,
            ours_tag=ours_tag,
            ours_label=ours_label,
            title_full=title_full,
            out=out,
        )
        return

    head_modules = [(ours_label, ours_tag)] + list(_DEFAULT_HEAD_ABL_TAGS)
    backbone_modules, bb_pred_used = _expand_backbone_modules(
        list(_DEFAULT_BACKBONE_TAGS),
        tag_map,
        prefer_raw_backbone=args.prefer_raw_backbone,
    )
    if not bb_pred_used and not args.prefer_raw_backbone:
        print(
            "Note: CSV 中尚无 m*_pred 行；骨干柱仍来自 best_mae 直连 eval。"
            "若 M1=M2=M3 tied，请先跑 ./scripts/run_ablations.sh score_heads_parallel 再 collect。",
            file=sys.stderr,
        )
    elif bb_pred_used:
        print(
            f"M1/M2/M3/M5 使用 *_pred（骨干上 train_pred_head 重训头的 continuous 指标）。",
            file=sys.stderr,
        )

    hm, hr, hl, herr = _extract_series(
        head_modules,
        tag_map,
        ours_label=ours_label,
        ours_tag=ours_tag,
        table_ours=table_ours,
        no_table_align=args.no_table_align,
    )
    bm, br, _, berr = _extract_series(
        backbone_modules,
        tag_map,
        ours_label=ours_label,
        ours_tag=None,
        table_ours=table_ours,
        no_table_align=True,
    )

    def _fatal_missing(name: str, miss: list[str], vals: list[float]) -> None:
        if miss and all(v != v for v in vals):
            print(f"错误：{name} 全无数据。", file=sys.stderr)
            for line in miss:
                print(f"  {line}", file=sys.stderr)
            sys.exit(1)

        for line in miss:
            print(f"⚠️ {line}", file=sys.stderr)

    _fatal_missing("Head panel", herr, hm)
    _fatal_missing("Backbone panel", berr, bm)

    n_h = len(hl)
    head_colors_arr = np.empty((n_h, 4))
    head_colors_arr[0] = mcolors.to_rgba("#2E86AB")
    for idx, hex_c in enumerate(_HEAD_BAR_COLORS):
        if 1 + idx < n_h:
            head_colors_arr[1 + idx] = mcolors.to_rgba(hex_c)

    n_b = len(backbone_modules)
    backbone_colors_arr = np.empty((n_b, 4))
    for i in range(n_b):
        backbone_colors_arr[i] = mcolors.to_rgba(_BACKBONE_BAR_COLORS[i % len(_BACKBONE_BAR_COLORS)])

    # M1-M3 三根柱仍完全一致时提示（多为未训 *_pred）
    if len(bm) >= 3 and all(bm[i] == bm[i] for i in range(3)):
        span3 = max(bm[0], bm[1], bm[2]) - min(bm[0], bm[1], bm[2])
        if span3 < 5e-4:
            print(
                "Note: M1-M3 MAE tied. Run score_heads_parallel (+ collect) for separable continuous metrics.",
                file=sys.stderr,
            )

    if ours_label and hm and hm[0] == hm[0]:
        for lab, xm in zip(hl[1:], hm[1:]):
            if xm == xm and xm < hm[0] - 1e-9:
                print(
                    f"⚠️ {lab} MAE ({xm:.4f}) 低于 Ours ({hm[0]:.4f})。",
                    file=sys.stderr,
                )
        for lab, xm in zip([t[0] for t in backbone_modules], bm):
            if xm == xm and xm < hm[0] - 1e-9:
                print(
                    f"⚠️ 骨干栏 {lab} MAE ({xm:.4f}) 低于 head 面板 Ours（若不对请检查缩放/协议）。",
                    file=sys.stderr,
                )

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.9), constrained_layout=False)
    fig.suptitle(title_full, fontsize=12, fontweight="bold", y=0.995)

    _draw_bar_axes(
        axes[0, 0],
        vals=hm,
        colors=head_colors_arr,
        xlab=hl,
        ylabel_short="MAE",
        log_y=args.log_y,
        symlog_y=args.symlog_y,
        tight_scale=True,
    )
    axes[0, 0].set_title("(a) Continuous prediction head", fontsize=11, pad=8)

    _draw_bar_axes(
        axes[0, 1],
        vals=bm,
        colors=backbone_colors_arr,
        xlab=[x[0] for x in backbone_modules],
        ylabel_short="MAE",
        log_y=args.log_y,
        symlog_y=args.symlog_y,
        tight_scale=True,
    )
    axes[0, 1].set_title("(b) Encoder / RL / context aug.", fontsize=11, pad=8)

    _draw_bar_axes(
        axes[1, 0],
        vals=hr,
        colors=head_colors_arr,
        xlab=hl,
        ylabel_short="RMSE",
        log_y=args.log_y,
        symlog_y=args.symlog_y,
        tight_scale=True,
    )

    _draw_bar_axes(
        axes[1, 1],
        vals=br,
        colors=backbone_colors_arr,
        xlab=[x[0] for x in backbone_modules],
        ylabel_short="RMSE",
        log_y=args.log_y,
        symlog_y=args.symlog_y,
        tight_scale=True,
    )

    leg_el = []
    leg_el.append(Patch(facecolor="#2E86AB", edgecolor="#222222", linewidth=0.95, label=ours_label))
    for lbl, hc in zip([t[0] for t in _DEFAULT_HEAD_ABL_TAGS], _HEAD_BAR_COLORS):
        leg_el.append(Patch(facecolor=hc, edgecolor="#222222", linewidth=0.95, label=lbl))
    for lbl, hc in zip([t[0] for t in backbone_modules], _BACKBONE_BAR_COLORS[: len(backbone_modules)]):
        leg_el.append(Patch(facecolor=hc, edgecolor="#222222", linewidth=0.95, label=f"B: {lbl}"))

    fig.legend(
        handles=leg_el,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.94),
        borderaxespad=0.5,
    )
    plt.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.09, hspace=0.45, wspace=0.26)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ 已保存 {out.resolve()} （layout=facet：左=head，右=backbone）")


# Wong 色盲友好（≈250 色域），顺序对应 Ours, M1, M2, M3, M4, M5
_WONG_BARS_HEX6 = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def _apply_pub_style_rc(dpi: int) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
        }
    )


def _reset_pub_style_rc() -> None:
    plt.rcdefaults()


def _apply_ieee_transactions_demo_rc(dpi: int) -> None:
    """IEEE Transactions / JSAC / TMC 期刊图常见规范：衬线体、细轴线、浅灰实线 y 向网格。"""
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "Nimbus Roman",
                "STIXGeneral",
            ],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.35,
            "grid.alpha": 0.42,
            "axes.axisbelow": True,
        }
    )


def _apply_journal_pastel_bar_rc(dpi: int) -> None:
    """通信/深度学习期刊柱状 baseline 示意图风格：sans、横向点线网格、despine。"""
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Nimbus Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.titleweight": "normal",
        }
    )


def _m4_index(labels_display: list[str]) -> int:
    if "M4" in labels_display:
        return labels_display.index("M4")
    return min(4, max(0, len(labels_display) - 1))


def _use_y_break(vals: list[float], m4_idx: int) -> bool:
    if m4_idx < 0 or m4_idx >= len(vals) or not (vals[m4_idx] == vals[m4_idx]):
        return False
    rest = [vals[i] for i in range(len(vals)) if i != m4_idx and vals[i] == vals[i]]
    if not rest:
        return False
    rmax = max(rest)
    return float(vals[m4_idx]) > rmax * 1.06


def _break_thresholds(vals: list[float], m4_idx: int) -> tuple[float, float, float]:
    """返回 (y_lo, break_top, break_bot_u)；用于下/上两段 y 轴。"""
    v4 = float(vals[m4_idx])
    rest = [float(vals[i]) for i in range(len(vals)) if i != m4_idx and vals[i] == vals[i]]
    rest_min = min(rest)
    rest_max = max(rest)
    span_s = rest_max - rest_min
    pad = max(0.25 * span_s, 0.35)
    y_lo = max(0.0, rest_min - pad)
    break_top = rest_max + max(0.08 * span_s, 0.12)
    mid = rest_max + 0.32 * (v4 - rest_max)
    gap = max(0.45 * (v4 - rest_max), 0.9)
    break_bot_u = v4 - gap
    break_bot_u = max(break_bot_u, break_top + mid * 0.02 + 0.25)
    if break_bot_u >= v4 - 1e-3:
        break_bot_u = v4 - gap
    return y_lo, break_top, break_bot_u


def _draw_axis_break_twospine(ax_low, ax_high) -> None:
    """两段轴衔接处在右侧画断裂符号（常见于会议论文柱状图）。"""
    kw = dict(color="0.25", lw=1.0, clip_on=False, zorder=99)
    trans_l = blended_transform_factory(ax_low.transAxes, ax_low.transData)
    trans_h = blended_transform_factory(ax_high.transAxes, ax_high.transData)
    ys_lo_lo, ys_lo_hi = ax_low.get_ylim()
    ys_hi_lo, ys_hi_hi = ax_high.get_ylim()
    h_low = max((ys_lo_hi - ys_lo_lo) * 0.012, 1e-6)
    h_high = max((ys_hi_hi - ys_hi_lo) * 0.012, 1e-6)
    x0, x1 = 1.0, 1.022
    ax_low.plot([x0, x1], [ys_lo_hi - h_low, ys_lo_hi], transform=trans_l, **kw)
    ax_low.plot([x0, x1], [ys_lo_hi, ys_lo_hi - h_low], transform=trans_l, **kw)
    ax_high.plot([x0, x1], [ys_hi_lo, ys_hi_lo + h_high], transform=trans_h, **kw)
    ax_high.plot([x0, x1], [ys_hi_lo + h_high, ys_hi_lo], transform=trans_h, **kw)


def _all_vals_finite(vals: list[float]) -> bool:
    return bool(vals) and all(v == v for v in vals)


def _plot_one_metric_pub_axes(
    ax_t,
    ax_b,
    vals: list[float],
    labels_display: list[str],
    colors_rgba: list,
    ylabel: str,
    ours_idx: int,
    m4_idx: int,
    *,
    show_delta: bool,
    show_xticklabels: bool,
    show_xlabel: bool,
) -> None:
    """上行 ax_t（仅断裂轴时放大 M4）、下行 ax_b 为主区；柱高仍为真实评测值。"""
    n = len(vals)
    xs = np.arange(n)
    w = 0.62
    break_top = break_bot_u = 0.0
    use_break = _use_y_break(vals, m4_idx) and _all_vals_finite(vals)

    if use_break:
        y_lo, break_top, break_bot_u = _break_thresholds(vals, m4_idx)
        v4 = float(vals[m4_idx])
        for i in range(n):
            if vals[i] != vals[i]:
                continue
            c = colors_rgba[i]
            if i == m4_idx:
                ax_b.bar(i, break_top, width=w, color=c, edgecolor="black", linewidth=0.65, zorder=3)
            else:
                ax_b.bar(i, float(vals[i]), width=w, color=c, edgecolor="black", linewidth=0.65, zorder=3)
        ax_b.set_ylim(y_lo, break_top)
        ax_t.bar(
            m4_idx,
            v4 - break_bot_u,
            width=w,
            bottom=break_bot_u,
            color=colors_rgba[m4_idx],
            edgecolor="black",
            linewidth=0.65,
            zorder=3,
        )
        ax_t.set_ylim(break_bot_u, v4 + max(0.04 * v4, 0.25))
        ax_t.set_xticks(xs)
        ax_t.tick_params(axis="x", length=0)
        ax_t.spines["bottom"].set_visible(False)
        ax_b.spines["top"].set_visible(False)
        plt.setp(ax_t.get_xticklabels(), visible=False)
        _draw_axis_break_twospine(ax_b, ax_t)
    else:
        ax_t.set_visible(False)
        for i in range(n):
            if vals[i] != vals[i]:
                continue
            ax_b.bar(i, float(vals[i]), width=w, color=colors_rgba[i], edgecolor="black", linewidth=0.65)
        ymin = min(float(vals[i]) for i in range(n) if vals[i] == vals[i])
        ymax = max(float(vals[i]) for i in range(n) if vals[i] == vals[i])
        ax_b.set_ylim(ymin - 0.06 * (ymax - ymin + 0.1), ymax + 0.08 * (ymax - ymin + 0.1))

    ax_b.set_ylabel(ylabel, fontsize=10)
    ax_b.set_xticks(xs)
    tick_labs = list(labels_display)
    if show_delta and vals[ours_idx] == vals[ours_idx]:
        ref = float(vals[ours_idx])
        tick_labs = []
        for i, lb in enumerate(labels_display):
            if vals[i] != vals[i]:
                tick_labs.append(lb)
                continue
            if i == ours_idx:
                tick_labs.append(lb)
                continue
            dlt = 100.0 * (float(vals[i]) / ref - 1.0)
            sgn = "+" if dlt >= 0 else ""
            tick_labs.append(f"{lb}\n({sgn}{dlt:.2f}%)")
    ax_b.set_xticklabels(tick_labs, fontsize=9, fontweight="semibold")

    plt.setp(ax_b.get_xticklabels(), visible=show_xticklabels)
    ax_b.tick_params(axis="x", labelbottom=show_xticklabels)
    ax_b.set_xlabel("Method" if show_xlabel else "", fontsize=10)

    for i in range(n):
        if vals[i] != vals[i]:
            continue
        v = float(vals[i])
        ytxt = v
        if use_break and i == m4_idx:
            ytxt = break_top * 0.985
        prec = 3 if (i != ours_idx and i != m4_idx and use_break) else 2
        ax_b.text(i, ytxt, f"{v:.{prec}f}", ha="center", va="bottom", fontsize=7.5)
    if use_break and vals[m4_idx] == vals[m4_idx]:
        v4 = float(vals[m4_idx])
        ax_t.text(m4_idx, v4, f"{v4:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax_b.grid(axis="y", linestyle="--", alpha=0.35)
    if use_break:
        ax_t.grid(axis="y", linestyle="--", alpha=0.35)


def _run_single_layout_conference(
    *,
    args: argparse.Namespace,
    maes: list[float],
    rmses: list[float],
    labels_display: list[str],
    title_full: str,
    out: Path,
) -> None:
    """ICLR/CVPR/NeurIPS 常见：中性网格、无色盲风险配色；M4 远高于其余时可选断裂 y 轴。"""
    _apply_pub_style_rc(args.dpi)
    try:
        n = len(maes)
        ours_idx = 0
        m4_idx = _m4_index(labels_display)
        colors_rgba = [mcolors.to_rgba(_WONG_BARS_HEX6[i % len(_WONG_BARS_HEX6)]) for i in range(n)]

        fig = plt.figure(figsize=(6.6, 5.2))
        fig.suptitle(title_full, fontsize=11, fontweight="bold", y=0.98)
        gs = fig.add_gridspec(
            4,
            1,
            height_ratios=[0.38, 1.08, 0.38, 1.08],
            hspace=0.12,
            left=0.11,
            right=0.98,
            top=0.89,
            bottom=0.10,
        )
        ax_mae_b = fig.add_subplot(gs[1])
        ax_rmse_b = fig.add_subplot(gs[3], sharex=ax_mae_b)
        ax_mae_t = fig.add_subplot(gs[0], sharex=ax_mae_b)
        ax_rmse_t = fig.add_subplot(gs[2], sharex=ax_mae_b)
        plt.setp(ax_mae_t.get_xticklabels(), visible=False)
        plt.setp(ax_rmse_t.get_xticklabels(), visible=False)

        show_d = not getattr(args, "no_pub_delta_labels", False)
        _plot_one_metric_pub_axes(
            ax_mae_t,
            ax_mae_b,
            list(maes),
            labels_display,
            colors_rgba,
            "(a) MAE",
            ours_idx,
            m4_idx,
            show_delta=show_d,
            show_xticklabels=False,
            show_xlabel=False,
        )
        _plot_one_metric_pub_axes(
            ax_rmse_t,
            ax_rmse_b,
            list(rmses),
            labels_display,
            colors_rgba,
            "(b) RMSE",
            ours_idx,
            m4_idx,
            show_delta=show_d,
            show_xticklabels=True,
            show_xlabel=True,
        )
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(
            f"✅ Conference-style single figure → {out.resolve()} "
            "(Wong palette, optional y-break, Δ% vs Ours on ablations)"
        )
    finally:
        _reset_pub_style_rc()


def _student_bar_axes_pastel(
    ax,
    *,
    vals: list[float],
    xlab: list[str],
    colors_hex: list[str],
    ylabel: str,
    bar_top_labels: list[str | None] | None,
    log_y: bool,
    symlog_y: bool,
    show_errorbars: bool = False,
    yerr: list[float] | None = None,
) -> None:
    """示意 baseline 对照图：粉彩无描边柱、点线 y 网格、斜排 x 标签；可选误差条。"""
    n = len(vals)
    if n == 0:
        return
    xs = np.arange(n)
    h = [_finite_or_zero(v) for v in vals]
    fc = [colors_hex[i] if i < len(colors_hex) and vals[i] == vals[i] else "#e8e8e8" for i in range(n)]

    err_trim = (yerr or [])[:n]
    use_err = (
        show_errorbars
        and (not log_y)
        and (not symlog_y)
        and len(err_trim) >= n
    )
    err_kw = dict(elinewidth=0.65, capthick=0.65, ecolor="#111111", zorder=2.5)

    bar_kw: dict = dict(
        width=0.68,
        color=fc,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    if use_err:
        bar_kw["yerr"] = err_trim
        bar_kw["capsize"] = 3.0
        bar_kw["error_kw"] = err_kw

    ax.bar(xs, h, **bar_kw)

    lo = np.inf
    hi = -np.inf
    for i, v in enumerate(vals):
        if v != v:
            continue
        lo = min(lo, float(v))
        top = float(v) + (float(err_trim[i]) if use_err else 0.0)
        hi = max(hi, top)

    ax.set_xticks(xs)
    ax.set_xticklabels(xlab, fontsize=8.5, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel, fontsize=10)
    if log_y:
        ax.set_yscale("log")
    elif symlog_y:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.72)
    ax.grid(axis="x", visible=False)

    if lo < np.inf and hi > -np.inf:
        span = max(hi - lo, 1e-6)
        ax.set_ylim(max(0.0, lo - 0.06 * span), hi + 0.12 * span)

    for i, (xi, v) in enumerate(zip(xs, vals)):
        if v != v:
            continue
        ycap = float(v) + (float(err_trim[i]) if use_err else 0.0)
        if bar_top_labels is not None and i < len(bar_top_labels) and bar_top_labels[i] is not None:
            tstr = bar_top_labels[i]
        else:
            tstr = f"{float(v):.2f}"
        pad = 0.018 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.text(xi, ycap + pad, tstr, ha="center", va="bottom", fontsize=8, color="#222222")


def _run_student_illustration(
    _unused_rr: Path,
    args: argparse.Namespace,
    ours_label: str,
    out: Path,
    title_full: str,
) -> None:
    """教学用示意图（模块顶注释说明排序与锚点）；非 CSV 实测。"""
    out.parent.mkdir(parents=True, exist_ok=True)
    labels_display = list(_STUDENT_DEMO_LABELS_DEFAULT)
    if ours_label != "Ours":
        labels_display[0] = ours_label
    maes = list(map(float, _STUDENT_DEMO_MAES))
    rmses = list(map(float, _STUDENT_DEMO_RMSES))
    pastel = list(_STUDENT_DEMO_FACE_HEX_PASTEL)
    n = len(maes)
    colors_rgba = np.array([mcolors.to_rgba(pastel[i % len(pastel)]) for i in range(n)])

    _apply_journal_pastel_bar_rc(args.dpi)
    try:
        panel_w = min(7.45, max(5.35, 0.98 * n + 1.18))
        fig_w = min(19.2, 2.12 * panel_w + 0.65)
        fig_h = max(5.15, min(9.5, 4.85 + 0.12 * n))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharex=True)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.30, wspace=0.28)
        fig.suptitle(title_full, fontsize=10.5, fontweight="normal", y=0.97)

        bb_idx = _backbone_only_indices(labels_display, ours_label=labels_display[0])
        hi_set = set(bb_idx)
        labs_mae = _format_bar_top_labels(maes, hi_set)
        labs_rmse = _format_bar_top_labels(rmses, hi_set)

        show_inset = (
            bool(getattr(args, "single_backbone_inset", False))
            and len(bb_idx) >= 2
            and not args.log_y
            and not args.symlog_y
        )

        _student_bar_axes_pastel(
            ax0,
            vals=maes,
            xlab=labels_display,
            colors_hex=pastel,
            ylabel="MAE",
            bar_top_labels=labs_mae,
            log_y=args.log_y,
            symlog_y=args.symlog_y,
            show_errorbars=False,
        )
        ax0.set_title("MAE", fontsize=10, pad=6)
        _student_bar_axes_pastel(
            ax1,
            vals=rmses,
            xlab=labels_display,
            colors_hex=pastel,
            ylabel="RMSE",
            bar_top_labels=labs_rmse,
            log_y=args.log_y,
            symlog_y=args.symlog_y,
            show_errorbars=False,
        )
        ax1.set_title("RMSE", fontsize=10, pad=6)

        if show_inset:
            _single_backbone_zoom_inset(
                ax0,
                vals=maes,
                colors=colors_rgba,
                xlab=labels_display,
                bb_idx=bb_idx,
                ylabel_short="MAE",
            )
            _single_backbone_zoom_inset(
                ax1,
                vals=rmses,
                colors=colors_rgba,
                xlab=labels_display,
                bb_idx=bb_idx,
                ylabel_short="RMSE",
            )
            fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.28, wspace=0.26)

        ax0.tick_params(axis="x", labelbottom=True)
        ax1.tick_params(axis="x", labelbottom=True)
        fig.supxlabel("Ablation variant", fontsize=10, y=0.06)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.05)
        plt.close(fig)
        print(f"✅ 示意消融图已保存 {out.resolve()}（非评测数据，见脚本内注释）")
    finally:
        _reset_pub_style_rc()


def _run_single_layout(
    *,
    rr: Path,
    args: argparse.Namespace,
    tag_map: dict[str, tuple[float, float]],
    table_ours: tuple[float, float] | None,
    ours_tag: str,
    ours_label: str,
    title_full: str,
    out: Path,
) -> None:
    modules: list[tuple[str, str]] = []
    if not args.no_ours:
        modules.append((ours_label, ours_tag))
    for label, t in _DEFAULT_M1_M5_TAGS:
        if label == "M4":
            tid = (args.m4_tag or "m4_no_joint").strip()
            modules.append((label, tid))
        else:
            bt, _usedp = _prefer_score_head_tag(t, tag_map, prefer_raw=args.prefer_raw_backbone)
            modules.append((label, bt))

    maes, rmses, labels_display, missing = _extract_series(
        modules,
        tag_map,
        ours_label=ours_label,
        ours_tag=ours_tag,
        table_ours=table_ours,
        no_table_align=args.no_table_align,
    )

    if missing and all(v != v for v in maes):
        print("错误：无任何匹配行。", file=sys.stderr)
        for line in missing:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)
    for line in missing:
        print(f"⚠️ {line}", file=sys.stderr)

    use_pub_single = (
        not getattr(args, "classic_single_viz", False)
        and not args.log_y
        and not args.symlog_y
    )
    if use_pub_single:
        _run_single_layout_conference(
            args=args,
            maes=list(maes),
            rmses=list(rmses),
            labels_display=labels_display,
            title_full=title_full,
            out=out,
        )
        return

    n = len(modules)
    colors = np.empty((n, 4))
    if not args.no_ours and n > 0:
        colors[0] = mcolors.to_rgba("#2E86AB")
        rest = plt.cm.Set2(np.linspace(0, 1, max(1, n - 1)))
        if n > 1:
            colors[1:] = rest
    else:
        colors = plt.cm.Set2(np.linspace(0, 1, max(1, n)))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(max(7.5, 1.2 * len(modules)), 8.0))
    fig.suptitle(title_full, fontsize=12, fontweight="bold")

    bb_idx = _backbone_only_indices(labels_display, ours_label=ours_label)
    hi_set = set(bb_idx)
    labs_mae = _format_bar_top_labels(maes, hi_set)
    labs_rmse = _format_bar_top_labels(rmses, hi_set)

    bar_h: list[str | None] | None = None
    if not args.no_ours and n == 6 and labels_display and labels_display[0] == ours_label:
        # Ours、M4 无斜线；M1–M3、M5 用不同 hatch 便于区分（柱高不变）
        bar_h = [None, "//", "xx", "++", None, ".."]

    show_inset = (
        bool(getattr(args, "single_backbone_inset", False))
        and len(bb_idx) >= 2
        and not args.log_y
        and not args.symlog_y
    )

    _draw_bar_axes(
        ax0,
        vals=list(maes),
        colors=colors,
        xlab=labels_display,
        ylabel_short="MAE",
        log_y=args.log_y,
        symlog_y=args.symlog_y,
        tight_scale=False,
        bar_top_labels=labs_mae,
        bar_hatches=bar_h,
    )
    if show_inset:
        _single_backbone_zoom_inset(
            ax0,
            vals=list(maes),
            colors=colors,
            xlab=labels_display,
            bb_idx=bb_idx,
            ylabel_short="MAE",
        )

    _draw_bar_axes(
        ax1,
        vals=list(rmses),
        colors=colors,
        xlab=labels_display,
        ylabel_short="RMSE",
        log_y=args.log_y,
        symlog_y=args.symlog_y,
        tight_scale=False,
        bar_top_labels=labs_rmse,
        bar_hatches=bar_h,
    )
    if show_inset:
        _single_backbone_zoom_inset(
            ax1,
            vals=list(rmses),
            colors=colors,
            xlab=labels_display,
            bb_idx=bb_idx,
            ylabel_short="RMSE",
        )

    ax1.set_xlabel("Ours vs module ablations")
    if show_inset:
        fig.subplots_adjust(left=0.09, right=0.97, top=0.9, bottom=0.08, hspace=0.32)
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(
        f"✅ 已保存 {out.resolve()}  （{len(modules)} 柱 layout=single"
        + ("：Ours + M1–M5" if not args.no_ours else "")
        + "；M1/M2/M3/M5：柱顶 4 位小数 + 不同 hatch"
        + ("；右上角插图（--single_backbone_inset）" if args.single_backbone_inset else "")
        + "）"
    )


if __name__ == "__main__":
    main()
