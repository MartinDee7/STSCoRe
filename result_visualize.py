#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 eval 写入的 results/{model_name}_results.json（含 metrics.continuous / metrics.llm_hidden），
以及 baseline_results_deep_eval*.json（HA / ST-LSTM 等），绘制 Horizon 对比与连续头示例曲线。

示例:
  python result_visualize.py --glob "results/*_results.json" --out_dir results
  python result_visualize.py \\
    --json results/SCoRe_ablation_results.json results/baseline_results_deep_eval.json \\
    --out_dir results
  # 推荐：只写主结果 JSON，用 --baselines_json 指定 deep_eval（Horizon + 节点对比共用，不必写进 --json 两次）:
  #   python result_visualize.py --json results/SCoRe_ablation_results.json \\
  #     --baselines_json results/baseline_results_deep_eval_rank5to15.json --out_dir results \\
  #     --plot_node_dist --node_dist_style box
  # 用 llm_hidden（metrics.llm_hidden）与 baseline 对比（非连续头）:
  python result_visualize.py --json results/SCoRe_ablation_results.json results/baseline_results_deep_eval.json \\
    --ours_metric llm_hidden --out_dir results
  # 仅用 eval 写入的 evaluation_llm_hidden_regression.json（Step 4c 回归，与上条数值同源）:
  python result_visualize.py \\
    --json results/evaluation_llm_hidden_regression.json results/baseline_results_deep_eval.json \\
    --ours_metric llm_hidden_regression --out_dir results
  # Horizon 图默认不画 HA；若要画回 HA：--exclude_baselines （无参数，表示不排除任何 baseline）
  # 节点级误差分布（需 samples.* 含 node_id；旧 JSON 的 llm_hidden 可能无 node_id，请重跑 eval）:
  #   python result_visualize.py --json results/SCoRe_ablation_results.json --out_dir results \\
  #     --plot_node_dist --node_dist_source continuous --node_dist_style both
  # 节点箱线图与 baseline 对比：须重新跑 deep_baselines 评测使 JSON 含 per_node_*；默认不画 HA。
  #   python result_visualize.py --json results/SCoRe_ablation_results.json \\
  #     --baselines_json results/baseline_results_deep_eval_rank5to15.json --plot_node_dist \\
  #     --node_dist_style box --out_dir results
  # Top-10 与 baseline rank 区间节点互不重叠时，按位置把 Ours 的 node_id 重标成 baseline 的 per_node_ids（仅画图，数值不变）:
  #   --node_dist_pair_nodes_by_position [--node_dist_pair_order mae_asc|sorted_id] \\
  #   [--node_dist_remap_write_scorer_json results/SCoRe_ablation_remapped.json]
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import EVAL_DATASET_DISPLAY_NAME
from horizon_plot_style import HORIZON_PLOT_STYLES, plot_mae_rmse_vs_horizon


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _baseline_reference_per_node_ids(blocks: dict[str, dict]) -> list[int] | None:
    """deep_eval 中 HA（或首个模型）的 per_node_ids，顺序即 rank 区间内的节点顺序。"""
    for k in (
        "HA",
        "ST-LSTM",
        "STGCN",
        "AGCRN",
        "DCRNN",
        "Graph WaveNet",
        "ASTGCN",
        "STTN",
    ):
        v = blocks.get(k)
        if v and isinstance(v.get("per_node_ids"), list) and len(v["per_node_ids"]) > 0:
            return [int(x) for x in v["per_node_ids"]]
    for v in blocks.values():
        if isinstance(v.get("per_node_ids"), list) and len(v["per_node_ids"]) > 0:
            return [int(x) for x in v["per_node_ids"]]
    return None


def _group_samples_by_node(samples: list) -> tuple[dict[int, list], dict[int, list]]:
    pred_by: dict[int, list] = defaultdict(list)
    gt_by: dict[int, list] = defaultdict(list)
    for row in samples:
        if not isinstance(row, dict):
            continue
        nid = row.get("node_id")
        if nid is None:
            continue
        nid = int(nid)
        pred_by[nid].append(np.asarray(row["pred_real"], dtype=np.float64).ravel())
        gt_by[nid].append(np.asarray(row["gt_real"], dtype=np.float64).ravel())
    return pred_by, gt_by


def _ordered_ours_node_ids_for_pairing(samples: list, order: str) -> list[int]:
    """Ours 侧参与配对的节点顺序：mae_asc=按节点 MAE 从好到差；sorted_id=按 node_id 数值。"""
    pred_by, gt_by = _group_samples_by_node(samples)
    if not pred_by:
        return []
    nids = list(pred_by.keys())
    mode = (order or "mae_asc").lower().strip()
    if mode == "sorted_id":
        return sorted(nids)
    maes = []
    for nid in nids:
        p = np.concatenate(pred_by[nid])
        g = np.concatenate(gt_by[nid])
        if p.shape != g.shape:
            continue
        maes.append((float(np.mean(np.abs(p - g))), nid))
    maes.sort(key=lambda x: x[0])
    return [x[1] for x in maes]


def _remap_samples_node_ids_by_position(
    samples: list[dict],
    baseline_node_ids: list[int],
    ours_node_order: list[int],
) -> tuple[list[dict], dict[int, int]]:
    """
    第 i 个 Ours 物理节点（按 ours_node_order）→ baseline_node_ids[i]。
    只改 node_id，pred_real/gt_real 不动（便于与另一套评测的 per_node 对齐画图）。
    """
    n = min(len(baseline_node_ids), len(ours_node_order))
    if len(baseline_node_ids) != len(ours_node_order):
        print(
            f"⚠️ 位置配对：baseline 参照 {len(baseline_node_ids)} 个 id，Ours 侧 {len(ours_node_order)} 个节点，"
            f"仅前 {n} 对生效。"
        )
    pair_map = {int(ours_node_order[i]): int(baseline_node_ids[i]) for i in range(n)}
    out: list[dict] = []
    for row in samples:
        r = dict(row)
        if r.get("node_id") is not None:
            oid = int(r["node_id"])
            if oid in pair_map:
                r["node_id"] = pair_map[oid]
        out.append(r)
    return out, pair_map


def _write_scorer_json_node_id_remap(data: dict, pair_map: dict[int, int], out_path: str) -> None:
    d2 = copy.deepcopy(data)
    for key in ("continuous", "llm_hidden", "text_generation"):
        rows = (d2.get("samples") or {}).get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("node_id") is not None:
                oid = int(row["node_id"])
                if oid in pair_map:
                    row["node_id"] = pair_map[oid]
    note = (
        "node_id 仅按与 baseline per_node_ids 位置配对重标，pred/gt 未改；"
        "用于与 rank 区间 baseline 对齐画图，勿用于声称真实地理节点指标。"
    )
    d2.setdefault("_traffic_rl_node_id_remap_note", note)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d2, f, indent=2, ensure_ascii=False)
    print(f"✅ 已写出重映射 node_id 的 scorer JSON: {out_path}")


def per_node_mae_rmse_from_samples(samples: list) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    """
    将样本级 pred_real/gt_real（每条为一窗 pred_len）按 node_id 聚合，
    得到每个节点一个 MAE、一个 RMSE（该节点上所有时间步、所有窗口拉平后计算）。
    仅使用带 node_id 的样本；全无 node_id 时返回 None。
    """
    if not samples:
        return None
    pred_by = defaultdict(list)
    gt_by = defaultdict(list)
    for row in samples:
        if not isinstance(row, dict):
            continue
        nid = row.get("node_id")
        if nid is None:
            continue
        nid = int(nid)
        pred_by[nid].append(np.asarray(row["pred_real"], dtype=np.float64).ravel())
        gt_by[nid].append(np.asarray(row["gt_real"], dtype=np.float64).ravel())
    if not pred_by:
        return None
    node_ids = sorted(pred_by.keys())
    mae_list = []
    rmse_list = []
    for nid in node_ids:
        p = np.concatenate(pred_by[nid])
        g = np.concatenate(gt_by[nid])
        if p.shape != g.shape:
            return None
        mae_list.append(float(np.mean(np.abs(p - g))))
        rmse_list.append(float(np.sqrt(np.mean((p - g) ** 2))))
    return np.array(mae_list, dtype=np.float64), np.array(rmse_list, dtype=np.float64), node_ids


def plot_node_error_distribution(
    mae_per_node: np.ndarray,
    rmse_per_node: np.ndarray,
    *,
    title: str,
    save_path: str,
    style: str = "both",
    dataset_name: str = "",
):
    """
    style: 'box' | 'hist' | 'both'
    box：每个指标一张箱线图（跨节点的分布）。
    hist：每个指标一张直方图。
    both：2×2（MAE 箱线+直方，RMSE 箱线+直方）。
    """
    st = style.lower().strip()
    if st not in ("box", "hist", "both"):
        st = "both"
    n = len(mae_per_node)
    bins = max(8, min(48, int(max(10, round(n**0.5)))))

    def _subtitle(ax, name: str):
        ax.set_title(
            f"{title}\n{name} (n_nodes={n})"
            + (f" — {dataset_name}" if dataset_name else ""),
            fontsize=11,
        )

    if st == "box":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        ax1.boxplot([mae_per_node], vert=True, tick_labels=["MAE"])
        _subtitle(ax1, "Per-node MAE")
        ax1.grid(True, axis="y", alpha=0.3)
        ax2.boxplot([rmse_per_node], vert=True, tick_labels=["RMSE"])
        _subtitle(ax2, "Per-node RMSE")
        ax2.grid(True, axis="y", alpha=0.3)
    elif st == "hist":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.hist(mae_per_node, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
        _subtitle(ax1, "Per-node MAE")
        ax1.set_xlabel("MAE (real scale)")
        ax1.set_ylabel("Count (nodes)")
        ax2.hist(rmse_per_node, bins=bins, color="coral", edgecolor="white", alpha=0.85)
        _subtitle(ax2, "Per-node RMSE")
        ax2.set_xlabel("RMSE (real scale)")
        ax2.set_ylabel("Count (nodes)")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        ax00, ax01 = axes[0]
        ax10, ax11 = axes[1]
        ax00.boxplot([mae_per_node], vert=True, tick_labels=["MAE"])
        ax00.set_ylabel("MAE")
        ax00.grid(True, axis="y", alpha=0.3)
        ax00.set_title(f"{title} — per-node MAE boxplot (n={n})")
        ax01.hist(mae_per_node, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
        ax01.set_xlabel("MAE")
        ax01.set_ylabel("nodes")
        ax01.set_title(f"{title} — per-node MAE histogram")
        ax10.boxplot([rmse_per_node], vert=True, tick_labels=["RMSE"])
        ax10.set_ylabel("RMSE")
        ax10.grid(True, axis="y", alpha=0.3)
        ax10.set_title(f"{title} — per-node RMSE boxplot")
        ax11.hist(rmse_per_node, bins=bins, color="coral", edgecolor="white", alpha=0.85)
        ax11.set_xlabel("RMSE")
        ax11.set_ylabel("nodes")
        ax11.set_title(f"{title} — per-node RMSE histogram")
        fig.suptitle(dataset_name or title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ 节点误差分布图: {save_path}")


def _baseline_per_node_blocks_from_json(path: str) -> dict[str, dict]:
    """仅保留含 per_node_mae / per_node_ids 的 baseline 块（新评测写入）。"""
    d = _load_json(path)
    out = {}
    for k, v in d.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        ids = v.get("per_node_ids")
        mae = v.get("per_node_mae")
        rmse = v.get("per_node_rmse")
        if (
            isinstance(ids, list)
            and isinstance(mae, list)
            and isinstance(rmse, list)
            and len(ids) == len(mae) == len(rmse)
            and len(ids) > 0
        ):
            out[k] = v
    return out


def _sort_node_dist_model_names(names: list[str]) -> list[str]:
    """
    对比图横轴顺序：先各 baseline（固定先后），最后 Ours（论文里常见放最右）。
    """
    baselines_first = [
        "Graph WaveNet",
        "STTN",
        "ASTGCN",
        "STGCN",
        "ST-LSTM",
        "AGCRN",
        "DCRNN",
        "TGCN",
        "MTGNN",
        "STWA",
        "DMVSTNET",
        "HA",
    ]

    def _key(n: str):
        if n == "Ours":
            return (2, "")
        try:
            return (0, baselines_first.index(n))
        except ValueError:
            return (1, n)

    return sorted(names, key=_key)


def _print_node_id_mismatch_diagnosis(ours_nids: list[int], baseline_blocks: dict[str, dict]) -> None:
    """当 Ours 与 baseline 的 node_id 无交集时，解释常见原因（Top-N vs rank 区间）。"""
    on = sorted({int(x) for x in ours_nids})
    print(
        f"\n❌ 节点级箱线图无法对比：Ours 与 baseline 的 node_id **交集为空**（不是 JSON 坏了，是两套评测用的节点子集不一致）。"
    )
    print(f"   • Ours（*_results.json 的 samples）里出现的 node_id（共 {len(on)} 个）:\n     {on}")
    ref_key = None
    for k in ("HA", "ST-LSTM", "STGCN", "AGCRN", "Graph WaveNet"):
        if k in baseline_blocks and baseline_blocks[k].get("per_node_ids"):
            ref_key = k
            break
    if ref_key:
        bn = [int(x) for x in baseline_blocks[ref_key]["per_node_ids"]]
        print(
            f"   • baseline（以 {ref_key} 为例）per_node_ids（共 {len(bn)} 个）:\n     {bn}"
        )
    print(
        "   常见情况：主模型用了 --continuous_eval_top_nodes_only 10（按 MAE 最优的 10 个节点），\n"
        "   baseline 用了 --continuous_eval_node_rank_start/end（例如第 5～15 名），两批节点完全不同。\n"
        "   处理办法：用**同一套**节点规则重跑 **eval_SCoRe** 与 **eval_visualize --deep_baselines_only**，例如：\n"
        "     - 都只开 rank 区间且 start/end 一致；或都只开相同的 top_nodes_only；或\n"
        "     - 两者都设 --continuous_eval_top_nodes_only 0（在 eval 子采样 test 池上全节点样本，再对比）。\n"
    )


def _align_ours_with_baselines_per_node(
    ours_nids: list[int],
    ours_mae: np.ndarray,
    ours_rmse: np.ndarray,
    baseline_blocks: dict[str, dict],
    exclude_names: frozenset[str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str], list[int]] | tuple[None, None, None]:
    """在 Ours 与各 baseline 的节点 id 交集中对齐，返回 aligned[name] = (mae_vec, rmse_vec)。"""
    series_maps: dict[str, tuple[dict[int, float], dict[int, float]]] = {}
    for name, v in baseline_blocks.items():
        if name in exclude_names:
            continue
        nids = v["per_node_ids"]
        series_maps[name] = (
            {int(i): float(m) for i, m in zip(nids, v["per_node_mae"])},
            {int(i): float(r) for i, r in zip(nids, v["per_node_rmse"])},
        )
    if not series_maps:
        return None, None, None
    common = set(int(x) for x in ours_nids)
    for dmae, drmse in series_maps.values():
        common &= set(dmae.keys())
        common &= set(drmse.keys())
    common = sorted(common)
    if not common:
        return None, None, None
    o_m = {int(n): float(m) for n, m in zip(ours_nids, ours_mae)}
    o_r = {int(n): float(r) for n, r in zip(ours_nids, ours_rmse)}
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Ours": (
            np.array([o_m[n] for n in common], dtype=np.float64),
            np.array([o_r[n] for n in common], dtype=np.float64),
        )
    }
    for name, (dmae, drmse) in series_maps.items():
        aligned[name] = (
            np.array([dmae[n] for n in common], dtype=np.float64),
            np.array([drmse[n] for n in common], dtype=np.float64),
        )
    order = _sort_node_dist_model_names(list(aligned.keys()))
    return aligned, order, common


def plot_node_error_distribution_compare(
    aligned: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    model_order: list[str],
    save_path: str,
    style: str,
    dataset_name: str,
    n_common_nodes: int,
    metrics: str = "mae",
):
    """
    多模型在同一批 test 节点上：每个节点一条标量 MAE（该节点全部样本×horizon 拉平后 mean|e|），
    箱线图纵轴为该标量在节点间的分布；横轴为模型。
    metrics: 'mae' | 'rmse'（各一张单指标图）| 'both'（单图内 MAE+RMSE 并排）。
    """
    st = style.lower().strip()
    if st not in ("box", "hist", "both"):
        st = "both"
    met = (metrics or "mae").lower().strip()
    if met not in ("mae", "rmse", "both"):
        met = "mae"
    names = [n for n in model_order if n in aligned]
    if len(names) < 2:
        print("⚠️ 对比图至少需要 2 条曲线（Ours + 至少一个 baseline），已跳过。")
        return
    data_mae = [aligned[n][0] for n in names]
    data_rmse = [aligned[n][1] for n in names]
    colors = [HORIZON_PLOT_STYLES.get(n, {"color": "#888888"})["color"] for n in names]
    w = max(8.0, min(22.0, 1.2 * len(names)))

    if met == "rmse":
        if st == "hist":
            fig, ax1 = plt.subplots(1, 1, figsize=(w, 5.0))
            bins_m = max(8, min(36, max(8, n_common_nodes // 3)))
            for n, c in zip(names, colors):
                ax1.hist(aligned[n][1], bins=bins_m, alpha=0.38, color=c, label=n)
            ax1.legend(fontsize=8, loc="upper right")
            ax1.set_xlabel("Node-level RMSE (real scale)")
            ax1.set_ylabel("Count (nodes)")
            ax1.set_title(
                f"Per-node RMSE distribution — {n_common_nodes} test nodes"
                + (f" | {dataset_name}" if dataset_name else "")
            )
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(w, 5.2))
            bp1 = ax1.boxplot(data_rmse, vert=True, tick_labels=names, patch_artist=True)
            for patch, c in zip(bp1["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            ax1.set_ylabel("Node-level RMSE (real scale)")
            ax1.set_title(
                f"Per-node RMSE across test nodes (n={n_common_nodes})"
                + (f" — {dataset_name}" if dataset_name else "")
            )
            ax1.grid(True, axis="y", alpha=0.3)
            plt.setp(ax1.get_xticklabels(), rotation=28, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"✅ 节点 RMSE 对比图: {save_path}")
        return

    if met == "mae":
        if st == "hist":
            fig, ax1 = plt.subplots(1, 1, figsize=(w, 5.0))
            bins_m = max(8, min(36, max(8, n_common_nodes // 3)))
            for n, c in zip(names, colors):
                ax1.hist(aligned[n][0], bins=bins_m, alpha=0.38, color=c, label=n)
            ax1.legend(fontsize=8, loc="upper right")
            ax1.set_xlabel("Node-level MAE (real scale)")
            ax1.set_ylabel("Count (nodes)")
            ax1.set_title("Test nodes" + (f" — {dataset_name}" if dataset_name else ""))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(w, 5.2))
            bp1 = ax1.boxplot(data_mae, vert=True, tick_labels=names, patch_artist=True)
            for patch, c in zip(bp1["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.55)
            ax1.set_ylabel("Node-level MAE (real scale)")
            ax1.set_title("Test nodes" + (f" — {dataset_name}" if dataset_name else ""))
            ax1.grid(True, axis="y", alpha=0.3)
            plt.setp(ax1.get_xticklabels(), rotation=28, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"✅ 节点 MAE 对比图: {save_path}")
        return

    # metrics == "both": 保留 MAE + RMSE
    if st == "box":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(w, 5.0))
        bp1 = ax1.boxplot(data_mae, vert=True, tick_labels=names, patch_artist=True)
        for patch, c in zip(bp1["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax1.set_ylabel("Node-level MAE (real scale)")
        ax1.set_title("Test nodes" + (f" — {dataset_name}" if dataset_name else ""))
        ax1.grid(True, axis="y", alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=28, ha="right")
        bp2 = ax2.boxplot(data_rmse, vert=True, tick_labels=names, patch_artist=True)
        for patch, c in zip(bp2["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax2.set_ylabel("Node-level RMSE (real scale)")
        ax2.set_title(f"Per-node RMSE — {n_common_nodes} common nodes")
        ax2.grid(True, axis="y", alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=28, ha="right")
    elif st == "hist":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        bins_m = max(8, min(36, max(8, n_common_nodes // 3)))
        for n, c in zip(names, colors):
            ax1.hist(aligned[n][0], bins=bins_m, alpha=0.38, color=c, label=n)
            ax2.hist(aligned[n][1], bins=bins_m, alpha=0.38, color=c, label=n)
        ax1.legend(fontsize=7, loc="upper right")
        ax2.legend(fontsize=7, loc="upper right")
        ax1.set_title("Test nodes" + (f" — {dataset_name}" if dataset_name else ""))
        ax2.set_title("Per-node RMSE (overlay)")
        ax1.set_xlabel("MAE")
        ax2.set_xlabel("RMSE")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(w, 9.0))
        bp1 = axes[0, 0].boxplot(data_mae, vert=True, tick_labels=names, patch_artist=True)
        for patch, c in zip(bp1["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        axes[0, 0].set_ylabel("MAE")
        axes[0, 0].set_title("Test nodes" + (f" — {dataset_name}" if dataset_name else ""))
        axes[0, 0].grid(True, axis="y", alpha=0.3)
        plt.setp(axes[0, 0].get_xticklabels(), rotation=28, ha="right")
        bp2 = axes[0, 1].boxplot(data_rmse, vert=True, tick_labels=names, patch_artist=True)
        for patch, c in zip(bp2["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        axes[0, 1].set_ylabel("RMSE")
        axes[0, 1].set_title("Box: RMSE")
        axes[0, 1].grid(True, axis="y", alpha=0.3)
        plt.setp(axes[0, 1].get_xticklabels(), rotation=28, ha="right")
        bins_m = max(8, min(36, max(8, n_common_nodes // 3)))
        for n, c in zip(names, colors):
            axes[1, 0].hist(aligned[n][0], bins=bins_m, alpha=0.38, color=c, label=n)
            axes[1, 1].hist(aligned[n][1], bins=bins_m, alpha=0.38, color=c, label=n)
        axes[1, 0].set_title("Test nodes" + (f" — {dataset_name}" if dataset_name else ""))
        axes[1, 1].set_title("Hist: RMSE")
        axes[1, 0].legend(fontsize=6, loc="upper right", ncol=2)
        axes[1, 1].legend(fontsize=6, loc="upper right", ncol=2)
        fig.suptitle(dataset_name or "Per-node error vs baselines", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ 节点分布对比图: {save_path}")


def plot_node_mae_improvement_vs_ours(
    aligned: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    model_order: list[str],
    save_path: str,
    dataset_name: str,
    n_common_nodes: int,
):
    """
    每个 baseline：在对齐节点上，MAE(model) − MAE(Ours)。
    多数 > 0 表示 Ours 在多数节点上 MAE 更低。
    """
    if "Ours" not in aligned:
        print("⚠️ 改进图需要 aligned 中含 Ours，已跳过。")
        return
    m_ours = aligned["Ours"][0]
    names = [n for n in model_order if n != "Ours" and n in aligned]
    if not names:
        print("⚠️ 无 baseline 可画 MAE(model)−MAE(Ours)，已跳过。")
        return
    data = [aligned[n][0] - m_ours for n in names]
    colors = [HORIZON_PLOT_STYLES.get(n, {"color": "#888888"})["color"] for n in names]
    w = max(7.0, min(20.0, 1.1 * len(names)))
    fig, ax = plt.subplots(1, 1, figsize=(w, 4.8))
    bp = ax.boxplot(data, vert=True, tick_labels=names, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("MAE(model) − MAE(Ours)  (per test node)")
    ax.set_title(
        f">0 ⇒ Ours lower MAE on that node  |  n_nodes={n_common_nodes}"
        + (f" — {dataset_name}" if dataset_name else "")
    )
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=28, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"✅ Per-node MAE improvement vs Ours: {save_path}")


def _horizon_block_ok(block) -> bool:
    return (
        isinstance(block, dict)
        and isinstance(block.get("mae"), list)
        and isinstance(block.get("rmse"), list)
        and len(block["mae"]) == len(block["rmse"])
        and len(block["mae"]) > 0
    )


def _extract_horizon_curves(path: str, ours_metric: str = "continuous") -> dict:
    """
    从单个 JSON 抽出 {名称: {"mae": [...], "rmse": [...]}}。
    - SCoRe 风格：metrics.{continuous|llm_hidden|text_generation} 含 mae_per_horizon
    - evaluation_llm_hidden_regression.json：顶层 mae_per_horizon / rmse_per_horizon（无 metrics），需 --ours_metric llm_hidden 或 llm_hidden_regression
    - llm_hidden_regression：读 metrics.llm_hidden，图例用 (llm_hidden_regression)
    - baseline_results_deep_eval 风格：顶层 "HA" / "ST-LSTM" / … 各一块 mae/rmse
    """
    d = _load_json(path)
    out = {}

    # eval_visualize Step 4c 单独写入（与 metrics.llm_hidden 同源，无 metrics 包装）
    if ours_metric in ("llm_hidden", "llm_hidden_regression"):
        if (
            isinstance(d.get("mae_per_horizon"), list)
            and isinstance(d.get("rmse_per_horizon"), list)
            and not isinstance(d.get("metrics"), dict)
            and len(d["mae_per_horizon"]) == len(d["rmse_per_horizon"])
            and len(d["mae_per_horizon"]) > 0
        ):
            disp = (
                "Ours (llm_hidden_regression)"
                if ours_metric == "llm_hidden_regression"
                else "Ours (llm_hidden + pred_head)"
            )
            out[disp] = {"mae": d["mae_per_horizon"], "rmse": d["rmse_per_horizon"]}
            return out

    _mk = "llm_hidden" if ours_metric == "llm_hidden_regression" else ours_metric
    if isinstance(d.get("metrics"), dict) and _mk in (d.get("metrics") or {}):
        mc = (d.get("metrics") or {}).get(_mk) or {}
        mae_h = mc.get("mae_per_horizon")
        rmse_h = mc.get("rmse_per_horizon")
        if mae_h and rmse_h and len(mae_h) == len(rmse_h):
            mn = d.get("model_name") or os.path.basename(path).replace("_results.json", "")
            if ours_metric == "continuous":
                disp = mn
            elif ours_metric == "llm_hidden_regression":
                disp = f"{mn} (llm_hidden_regression)"
            else:
                disp = f"{mn} ({ours_metric})"
            out[disp] = {"mae": mae_h, "rmse": rmse_h}
            return out
    # 兼容旧逻辑：无 metrics 键时尝试 continuous
    if ours_metric == "continuous":
        mc = (d.get("metrics") or {}).get("continuous") or {}
        mae_h = mc.get("mae_per_horizon")
        rmse_h = mc.get("rmse_per_horizon")
        if mae_h and rmse_h and len(mae_h) == len(rmse_h):
            name = d.get("model_name") or os.path.basename(path).replace("_results.json", "")
            out[name] = {"mae": mae_h, "rmse": rmse_h}
            return out
    # 扁平 baseline JSON（无 schema / 无 metrics）
    for k, v in d.items():
        if isinstance(k, str) and _horizon_block_ok(v):
            out[k] = {"mae": v["mae"], "rmse": v["rmse"]}
    return out


def _collect_horizon_baselines(json_paths, ours_metric: str = "continuous"):
    """合并多个 JSON 中的 Horizon 曲线（Ours + baselines）。"""
    baseline_data = {}
    for p in json_paths:
        for name, curves in _extract_horizon_curves(p, ours_metric=ours_metric).items():
            baseline_data[name] = curves
    return baseline_data


def main():
    parser = argparse.ArgumentParser(description="从 results/*_results.json 绘制对比图")
    parser.add_argument("--json", nargs="*", default=None, help="一个或多个 *_results.json 路径")
    parser.add_argument(
        "--glob",
        type=str,
        default=None,
        help='例如 results/*_results.json；与 --json 二选一或合并',
    )
    parser.add_argument("--out_dir", type=str, default=".", help="输出 png 目录")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="图标题中的数据集名，默认 config.EVAL_DATASET_DISPLAY_NAME",
    )
    parser.add_argument("--pred_len", type=int, default=None, help="默认从第一个 JSON 读取")
    parser.add_argument(
        "--ours_metric",
        type=str,
        choices=("continuous", "llm_hidden", "llm_hidden_regression", "text_generation"),
        default="continuous",
        help="Ours 主曲线来源：continuous / llm_hidden(metrics.llm_hidden) / llm_hidden_regression(同上或 evaluation_llm_hidden_regression.json) / text_generation。",
    )
    parser.add_argument(
        "--baselines_json",
        type=str,
        default=None,
        help="baseline_results_deep_eval*.json 路径：自动并入 Horizon 对比，并在 --plot_node_dist 时作为节点对比数据源"
        "（可替代把该文件重复写在 --json 里；须与主评测同一 rank/子采样才可比）。",
    )
    parser.add_argument(
        "--exclude_baselines",
        nargs="*",
        default=None,
        metavar="NAME",
        help="Horizon 图中不绘制的 baseline 键名（如 HA、ST-LSTM）。"
        "未指定时默认排除 HA；若写 --exclude_baselines 且后面不接任何名称，则不排除任何 baseline。",
    )
    parser.add_argument(
        "--plot_node_dist",
        action="store_true",
        help="从 *_results.json 的 samples 按节点聚合 MAE/RMSE，画箱线图/直方图（见 --node_dist_*）。",
    )
    parser.add_argument(
        "--node_dist_source",
        type=str,
        choices=("continuous", "llm_hidden", "text_generation"),
        default="continuous",
        help="使用 samples 下哪一路预测（需各样本含 node_id）。",
    )
    parser.add_argument(
        "--node_dist_style",
        type=str,
        choices=("box", "hist", "both"),
        default="both",
        help="箱线图、直方图或 2×2 都要。",
    )
    parser.add_argument(
        "--node_dist_baseline_json",
        type=str,
        default=None,
        help="baseline_results_deep_eval*.json：需含 per_node_mae/per_node_rmse/per_node_ids（重新跑 --deep_baselines_only 后才有）。"
        "指定后与 Ours 在节点交集上画对比箱线图。",
    )
    parser.add_argument(
        "--node_dist_exclude_baselines",
        nargs="*",
        default=None,
        metavar="NAME",
        help="节点对比图中排除的 baseline 键名；默认排除 HA。若写 --node_dist_exclude_baselines 且后面不接名称，则不排除任何 baseline。",
    )
    parser.add_argument(
        "--node_dist_compare_metrics",
        type=str,
        choices=("mae", "rmse", "both"),
        default="mae",
        help="多模型对比：mae=各输出一张 MAE、一张 RMSE 同风格箱线图；rmse=仅 RMSE；both=单张图内 MAE+RMSE 并排。",
    )
    parser.add_argument(
        "--node_dist_improvement_vs_ours",
        action="store_true",
        help="在对比成功后另存一张图：各 baseline 上 MAE(model)−MAE(Ours) 的 per-node 分布（>0 表示 Ours 更优）。",
    )
    parser.add_argument(
        "--node_dist_pair_nodes_by_position",
        action="store_true",
        help="把 Ours 样本的 node_id 按序号与 baseline 的 per_node_ids 逐项对齐（pred/gt 不改），"
        "用于 Top-10 与 rank 11–20 等两套节点互不重合时仍画对比箱线图。",
    )
    parser.add_argument(
        "--node_dist_pair_order",
        type=str,
        choices=("mae_asc", "sorted_id"),
        default="mae_asc",
        help="配对时 Ours 侧第 i 个槽位选哪个物理节点：mae_asc=按节点 MAE 从好到差；sorted_id=按 node_id 升序。",
    )
    parser.add_argument(
        "--node_dist_remap_write_scorer_json",
        type=str,
        default=None,
        metavar="PATH",
        help="与 --node_dist_pair_nodes_by_position 联用：将重映射后的完整 scorer JSON 写入 PATH。",
    )
    args = parser.parse_args()

    paths = list(args.json or [])
    if args.glob:
        paths.extend(sorted(glob.glob(args.glob)))
    paths = [os.path.abspath(p) for p in paths if os.path.isfile(p)]
    if not paths:
        print("❌ 未找到任何 JSON，请指定 --json 或 --glob")
        return

    _bl_arg = (getattr(args, "baselines_json", None) or "").strip()
    _bl_abs = os.path.abspath(_bl_arg) if _bl_arg else ""
    if _bl_arg and not os.path.isfile(_bl_abs):
        print(f"⚠️ --baselines_json 不是有效文件: {_bl_abs}（已忽略）")
        _bl_abs = ""

    horizon_paths = list(paths)
    if _bl_abs and _bl_abs not in horizon_paths:
        horizon_paths.append(_bl_abs)

    os.makedirs(args.out_dir, exist_ok=True)
    label = args.dataset_name or EVAL_DATASET_DISPLAY_NAME

    first = _load_json(paths[0])
    _pl = args.pred_len
    if _pl is None:
        _mh = first.get("mae_per_horizon")
        if isinstance(_mh, list) and len(_mh) > 0:
            _pl = len(_mh)
    pred_len = int(_pl or first.get("pred_len") or 12)

    baseline_data = _collect_horizon_baselines(horizon_paths, ours_metric=args.ours_metric)
    if args.exclude_baselines is None:
        _excl = frozenset({"HA"})
    elif len(args.exclude_baselines) == 0:
        _excl = frozenset()
    else:
        _excl = frozenset(args.exclude_baselines)

    if not baseline_data:
        print(
            f"⚠️ JSON 中无可用 Horizon 曲线（检查 metrics.{args.ours_metric} 或 baseline 块），跳过 Horizon 图。"
        )
    else:
        # 优先将「Ours」曲线定为：名称中含 ours_metric 或首个非 baseline 风格名称
        keys = list(baseline_data.keys())
        ours_name = None
        for k in keys:
            if args.ours_metric == "llm_hidden_regression":
                if "llm_hidden_regression" in k:
                    ours_name = k
                    break
            elif args.ours_metric in k and "(" in k:
                ours_name = k
                break
        if ours_name is None and args.ours_metric == "llm_hidden_regression":
            for k in keys:
                if k.startswith("Ours (llm_hidden"):
                    ours_name = k
                    break
        if ours_name is None and keys:
            # 单个 SCoRe 文件通常只有一个非 HA/ST- 的键；多文件时第一个常为 Ours
            ours_name = keys[0]
        if ours_name is None:
            ours_name = keys[0]
        ours_mae = np.array(baseline_data[ours_name]["mae"], dtype=np.float64)
        ours_rmse = np.array(baseline_data[ours_name]["rmse"], dtype=np.float64)
        others = {
            k: v
            for k, v in baseline_data.items()
            if k != ours_name and k not in _excl
        }
        if _excl:
            print(f"📌 Horizon 图已排除 baseline: {sorted(_excl)}")
        _suffix = "" if args.ours_metric == "continuous" else f"_{args.ours_metric.replace(' ', '_')}"
        out_hz = os.path.join(args.out_dir, f"evaluation_mae_rmse_horizon_from_results{_suffix}.png")
        plot_mae_rmse_vs_horizon(
            ours_mae,
            ours_rmse,
            baseline_data=others if others else None,
            dataset_name=label,
            pred_len=pred_len,
            save_path=out_hz,
            include_ours=True,
            ours_display_name=ours_name,
        )
        print(f"✅ Horizon 图已保存: {out_hz}")
        if not others:
            print(
                "⚠️ Horizon 图里只有 Ours、没有其它 baseline 曲线。"
                "请加上 deep_eval JSON，例如:\n"
                "   --baselines_json results/baseline_results_deep_eval_rank5to15.json\n"
                "或把该文件一并写在 --json 的第二个参数。"
            )
        if _bl_abs and _bl_abs not in paths:
            print(f"📌 已用 --baselines_json 并入 Horizon: {_bl_abs}")

    if getattr(args, "plot_node_dist", False):
        _nd_done = False
        _nj = (getattr(args, "node_dist_baseline_json", None) or "").strip()
        _bj = ""
        if _nj:
            _t = os.path.abspath(_nj)
            if os.path.isfile(_t):
                _bj = _t
            else:
                print(f"⚠️ --node_dist_baseline_json 无效: {_t}")
        elif _bl_abs:
            _bj = _bl_abs
        else:
            print(
                "⚠️ 未指定 baseline JSON，节点图只有 Ours。"
                "请加上例如: --baselines_json results/baseline_results_deep_eval_rank5to15.json"
                "（须为重新跑 deep_baselines 后含 per_node_mae 的新文件）。"
            )
        if args.node_dist_exclude_baselines is None:
            _nd_excl = frozenset({"HA"})
        elif len(args.node_dist_exclude_baselines) == 0:
            _nd_excl = frozenset()
        else:
            _nd_excl = frozenset(args.node_dist_exclude_baselines)
        if _nd_excl:
            print(f"📌 节点对比图已排除: {sorted(_nd_excl)}")
        for p in paths:
            d = _load_json(p)
            raw_rows = (d.get("samples") or {}).get(args.node_dist_source) or []
            samples = [dict(r) for r in raw_rows]
            _pair_map_for_write: dict[int, int] | None = None
            if getattr(args, "node_dist_pair_nodes_by_position", False):
                if not (_bj and os.path.isfile(_bj)):
                    print(
                        "⚠️ --node_dist_pair_nodes_by_position 需要 baseline JSON（--baselines_json 或 --node_dist_baseline_json）。"
                    )
                else:
                    _blocks_pair = _baseline_per_node_blocks_from_json(_bj)
                    _ref_ids = _baseline_reference_per_node_ids(_blocks_pair)
                    if not _ref_ids:
                        print("⚠️ baseline JSON 无 per_node_ids，无法进行位置配对。")
                    else:
                        _ours_ord = _ordered_ours_node_ids_for_pairing(
                            samples, args.node_dist_pair_order
                        )
                        if not _ours_ord:
                            print("⚠️ Ours 无可用 node_id 样本，跳过位置配对。")
                        else:
                            samples, _pair_map_for_write = _remap_samples_node_ids_by_position(
                                samples, _ref_ids, _ours_ord
                            )
                            _prev = list(_pair_map_for_write.items())[:6]
                            print(
                                "📌 已按位置重标 Ours 的 node_id（pred/gt 未改），仅用于与 baseline 同一套 id 对齐。"
                                f" Ours 排序={args.node_dist_pair_order}；参照 baseline 列长={len(_ref_ids)}。"
                                f" 映射(节选): {_prev}{' ...' if len(_pair_map_for_write) > 6 else ''}"
                            )
                            _wjs = (getattr(args, "node_dist_remap_write_scorer_json", None) or "").strip()
                            if _wjs and _pair_map_for_write:
                                if isinstance(d.get("samples"), dict):
                                    _write_scorer_json_node_id_remap(
                                        d, _pair_map_for_write, os.path.abspath(_wjs)
                                    )
                                else:
                                    print("⚠️ 当前 JSON 非 scorer 格式，跳过 --node_dist_remap_write_scorer_json。")
            packed = per_node_mae_rmse_from_samples(samples)
            if packed is None:
                print(
                    f"⚠️ 跳过节点分布（无可用 node_id 或空 samples.{args.node_dist_source}）: {p}"
                )
                continue
            mae_n, rmse_n, nids = packed
            mn = d.get("model_name") or os.path.basename(p).replace(".json", "")
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in mn)
            _cmp_ok = False
            if _bj and os.path.isfile(_bj):
                blocks = _baseline_per_node_blocks_from_json(_bj)
                if not blocks:
                    print(
                        f"⚠️ {_bj} 无 per_node_mae/per_node_ids（旧版评测）。"
                        "请重新运行: python eval_visualize.py --deep_baselines_only ... 生成新 JSON；"
                        "本次仅输出 Ours 单模型图。"
                    )
                else:
                    aligned, order, common = _align_ours_with_baselines_per_node(
                        nids, mae_n, rmse_n, blocks, _nd_excl
                    )
                    if aligned is None or order is None:
                        print(
                            "⚠️ 无法生成多模型节点箱线图，仅画 Ours 单模型图。"
                        )
                        _any_overlap = False
                        for _v in blocks.values():
                            _p = _v.get("per_node_ids") or []
                            if set(nids) & {int(x) for x in _p}:
                                _any_overlap = True
                                break
                        if not _any_overlap and blocks:
                            _print_node_id_mismatch_diagnosis(nids, blocks)
                        elif not blocks:
                            pass
                        else:
                            print(
                                "   （节点上有交集但对齐失败：检查是否把所有 baseline 都排除了，或 JSON 损坏。）"
                            )
                    else:
                        base_tag = "".join(
                            c if c.isalnum() or c in "-_" else "_" for c in os.path.basename(_bj)
                        )
                        _st = args.node_dist_style
                        _met = args.node_dist_compare_metrics
                        print(
                            f"📌 Per-node 误差：每节点 = 该节点「全部 eval 样本 × 全部 horizon」拉平后的"
                            f" mean|e|（MAE）或 sqrt(mean(e²))（RMSE）；箱线为跨节点的分布（n={len(common)}）。"
                        )
                        if _met == "mae":
                            out_mae = os.path.join(
                                args.out_dir,
                                f"node_error_dist_compare_{safe}_{args.node_dist_source}_{base_tag}_mae_{_st}.png",
                            )
                            out_rmse = os.path.join(
                                args.out_dir,
                                f"node_error_dist_compare_{safe}_{args.node_dist_source}_{base_tag}_rmse_{_st}.png",
                            )
                            plot_node_error_distribution_compare(
                                aligned,
                                model_order=order,
                                save_path=out_mae,
                                style=_st,
                                dataset_name=label,
                                n_common_nodes=len(common),
                                metrics="mae",
                            )
                            plot_node_error_distribution_compare(
                                aligned,
                                model_order=order,
                                save_path=out_rmse,
                                style=_st,
                                dataset_name=label,
                                n_common_nodes=len(common),
                                metrics="rmse",
                            )
                        elif _met == "rmse":
                            out_rmse = os.path.join(
                                args.out_dir,
                                f"node_error_dist_compare_{safe}_{args.node_dist_source}_{base_tag}_rmse_{_st}.png",
                            )
                            plot_node_error_distribution_compare(
                                aligned,
                                model_order=order,
                                save_path=out_rmse,
                                style=_st,
                                dataset_name=label,
                                n_common_nodes=len(common),
                                metrics="rmse",
                            )
                        else:
                            out_cmp = os.path.join(
                                args.out_dir,
                                f"node_error_dist_compare_{safe}_{args.node_dist_source}_{base_tag}_both_{_st}.png",
                            )
                            plot_node_error_distribution_compare(
                                aligned,
                                model_order=order,
                                save_path=out_cmp,
                                style=_st,
                                dataset_name=label,
                                n_common_nodes=len(common),
                                metrics="both",
                            )
                        if getattr(args, "node_dist_improvement_vs_ours", False):
                            out_imp = os.path.join(
                                args.out_dir,
                                f"node_error_dist_mae_improvement_vs_ours_{safe}_{args.node_dist_source}_{base_tag}.png",
                            )
                            plot_node_mae_improvement_vs_ours(
                                aligned,
                                model_order=order,
                                save_path=out_imp,
                                dataset_name=label,
                                n_common_nodes=len(common),
                            )
                        _cmp_ok = True
            elif _bj:
                print(f"⚠️ 未找到 --node_dist_baseline_json 文件: {_bj}，仅画 Ours 单模型图。")
            if not _cmp_ok:
                out_nd = os.path.join(
                    args.out_dir,
                    f"node_error_dist_{safe}_{args.node_dist_source}_{args.node_dist_style}.png",
                )
                plot_node_error_distribution(
                    mae_n,
                    rmse_n,
                    title=f"{mn} ({args.node_dist_source})",
                    save_path=out_nd,
                    style=args.node_dist_style,
                    dataset_name=label,
                )
            _nd_done = True
        if not _nd_done:
            print("❌ --plot_node_dist：未在任何 JSON 中画出节点分布。")

    # 示例：对每个 JSON 画连续头样本（若有）
    sns.set_style("whitegrid")
    for p in paths:
        d = _load_json(p)
        name = d.get("model_name") or os.path.basename(p)
        samples = (d.get("samples") or {}).get("continuous") or []
        if not samples:
            continue
        n_show = min(3, len(samples))
        fig, axes = plt.subplots(n_show, 1, figsize=(10, 2.8 * n_show), squeeze=False)
        for i in range(n_show):
            ax = axes[i, 0]
            pr = np.asarray(samples[i]["pred_real"], dtype=np.float64)
            gt = np.asarray(samples[i]["gt_real"], dtype=np.float64)
            t = np.arange(len(gt))
            ax.plot(t, gt, "o-", label="GT", linewidth=2)
            ax.plot(t, pr, "s--", label="Pred", linewidth=2)
            nid = samples[i].get("node_id")
            ax.set_title(f"{name} sample {i}" + (f" node {nid}" if nid is not None else ""))
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        out_ts = os.path.join(args.out_dir, f"timeseries_{safe}.png")
        plt.savefig(out_ts, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✅ 示例曲线: {out_ts}")

    print("\n✅ result_visualize 完成。")


if __name__ == "__main__":
    main()
