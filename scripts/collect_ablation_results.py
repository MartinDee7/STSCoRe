#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 eval *_results.json：默认从 samples.continuous 重算 micro（与 metrics.continuous 一致）。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


def _bootstrap_repo_root(script_path: Path) -> Path:
    rr = script_path.resolve().parents[1]
    if str(rr) not in sys.path:
        sys.path.insert(0, str(rr))
    return rr


def _default_scan_dir(repo_root: Path) -> Path:
    env = (os.environ.get("TRAFFIC_RL_ABLATION_RESULTS_DIR") or "").strip()
    if env:
        return Path(env)
    ablation = repo_root / "ablation_results"
    if ablation.is_dir():
        return ablation
    return repo_root / "results"


def _default_out_csv(scan_dir: Path) -> Path:
    return scan_dir / "ablation_summary.csv"


def _infer_tag_from_filename(p: Path) -> str:
    stem = p.name
    if stem.startswith("ablation_") and stem.endswith("_results.json"):
        return stem[len("ablation_") : -len("_results.json")]
    return stem


def main() -> None:
    rr = _bootstrap_repo_root(Path(__file__))
    from ablation_metric_utils import micro_mae_rmse, samples_continuous_rows

    default_scan = _default_scan_dir(rr)
    ap = argparse.ArgumentParser(description="消融 JSON → CSV（micro 来自 samples；可选相对 Full 的 %）")
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--scan_dir", type=Path, default=None)
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument(
        "--metric",
        choices=("micro_samples", "metrics_json"),
        default="micro_samples",
        help="micro_samples：从 samples 重算（推荐）；metrics_json：读 JSON 顶栏 metrics.continuous",
    )
    args = ap.parse_args()

    scan_dir = args.scan_dir if args.scan_dir is not None else default_scan
    out_csv = args.out_csv if args.out_csv is not None else _default_out_csv(scan_dir)

    if args.files:
        paths = [p if p.is_absolute() else rr / p for p in args.files]
    else:
        paths = sorted(scan_dir.glob("ablation_*_results.json"))

    ref_micro: float | None = None
    full_path = scan_dir / "ablation_00_full_results.json"
    if full_path.is_file():
        try:
            d = json.loads(full_path.read_text(encoding="utf-8"))
            s = samples_continuous_rows(d)
            if args.metric == "micro_samples":
                ref_micro = micro_mae_rmse(s)[0]
            else:
                mc = (d.get("metrics") or {}).get("continuous") or {}
                if mc.get("mae") is not None:
                    ref_micro = float(mc["mae"])
        except Exception:
            pass

    rows: list[dict[str, object]] = []
    for p in paths:
        if not p.is_file():
            print(f"skip missing: {p}", file=sys.stderr)
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"skip bad json {p}: {e}", file=sys.stderr)
            continue
        try:
            rel_fp = str(p.relative_to(rr))
        except ValueError:
            rel_fp = str(p)

        samples = samples_continuous_rows(data)
        mcont = (data.get("metrics") or {}).get("continuous") or {}
        tag = _infer_tag_from_filename(p)

        if args.metric == "micro_samples":
            mae_p, rmse_p = micro_mae_rmse(samples)
        else:
            mae_p = float(mcont["mae"]) if mcont.get("mae") is not None else float("nan")
            rmse_p = float(mcont["rmse"]) if mcont.get("rmse") is not None else float("nan")

        pct = ""
        if (
            ref_micro is not None
            and ref_micro == ref_micro
            and mae_p == mae_p
            and tag != "00_full"
        ):
            pct = 100.0 * (mae_p - ref_micro) / ref_micro

        row = {
            "file": rel_fp,
            "tag": tag,
            "model_name": data.get("model_name", ""),
            "checkpoint_path": data.get("checkpoint_path", ""),
            "metric_mode": args.metric,
            "mae_continuous": mae_p if mae_p == mae_p else "",
            "rmse_continuous": rmse_p if rmse_p == rmse_p else "",
            "mae_pct_vs_full_backbone_micro": pct if pct != "" else "",
        }
        if mcont.get("mae") is not None:
            row["metrics_continuous_mae_json"] = mcont.get("mae")
            row["metrics_continuous_rmse_json"] = mcont.get("rmse")
        rows.append(row)

    if not rows:
        print(f"未找到有效 JSON。scan_dir={scan_dir.resolve()}", file=sys.stderr)
        sys.exit(1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"✅ 已写入 {out_csv.resolve()} （{len(rows)} 行, metric={args.metric}）")


if __name__ == "__main__":
    main()
