"""由 eval_visualize 拆分：原完整评估+可视化流程（向后兼容）。"""
# 不在本模块修改 CUDA_VISIBLE_DEVICES；用哪几张卡、暴露几张卡，一律在运行命令前由 shell 自行 export。

from eval_common import *
from eval_common import (  # import * 不导入以下划线开头的名称
    _model_path,
    _load_icl_train_matrix_normalized,
    _continuous_head_forward_collect,
    _build_rag_pool_spatial_train_nodes,
    _build_rag_pool_for_eval,
    _report_continuous_top_nodes_by_column,
    _pick_representative_indices_per_node,
    _pick_best_sample_index_per_node,
    _compute_per_horizon_metrics,
)

def main():
    print("=" * 60)
    print("🚀 Traffic RL Model Evaluation & Visualization")
    print("=" * 60)
    print(f"🖥️  device={DEVICE}（默认强制 CUDA；无 GPU 调试请 export TRAFFIC_RL_ALLOW_CPU=1）")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--checkpoint_path", type=str, default=CHECKPOINT_PATH)
    parser.add_argument(
        "--deep_baselines_root",
        type=str,
        default=None,
        help="已训练深度学习 baseline 根目录，内含 st_lstm/stgcn/agcrn/best.pt；与当前 test 数据一起评估并画入 Horizon 图",
    )
    parser.add_argument(
        "--deep_baselines_only",
        action="store_true",
        help="仅评估上述三个 baseline，不加载 LLM（无需 checkpoint_path）",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Horizon 图标题/Y 轴中的数据集名；默认 config.EVAL_DATASET_DISPLAY_NAME（Milan），或环境变量 TRAFFIC_RL_EVAL_DATASET_NAME",
    )
    parser.add_argument(
        "--continuous_only",
        action="store_true",
        help="仅评估连续头 predict_continuous，跳过 LLM 文本生成（Step 4）；"
        "加载模型时不拉 Qwen 基座（continuous_head_rank_only），显存≈encoder+头，避免 OOM。"
        "若同时指定 --eval_llm_hidden_regression 则仍加载完整 LLM。",
    )
    parser.add_argument(
        "--top_nodes_report",
        type=int,
        default=-1,
        help="连续头评估后，按节点聚合 MAE 输出最优 Top-N 节点 ID 及象限/ZS 区域统计，并写入 evaluation_top_nodes_continuous.json；"
        "默认取环境变量 TRAFFIC_RL_EVAL_TOP_NODES 或 100；0=关闭；-1=用默认",
    )
    parser.add_argument(
        "--continuous_eval_top_nodes_only",
        type=int,
        default=-1,
        help="连续头 MAE/RMSE、连续头图、Horizon 中 Ours 仅使用「全 test 上按节点 MAE 从优到劣」的前 N 个节点上的全部窗口；"
        "0=不筛选（全 test）；-1=用环境变量 TRAFFIC_RL_EVAL_CONTINUOUS_TOP_ONLY 或 0。输出文件名带 _top{N}nodes 后缀以免覆盖。",
    )
    parser.add_argument(
        "--continuous_eval_node_rank_start",
        type=int,
        default=None,
        help="与 --continuous_eval_node_rank_end 联用：按连续头节点 MAE 从优到劣排名（1=最优），"
        "仅取第 rank_start～rank_end 名（含端点）上的样本评连续头并对齐 baseline；"
        "与 --continuous_eval_top_nodes_only 同时设时以本区间为准。",
    )
    parser.add_argument(
        "--continuous_eval_node_rank_end",
        type=int,
        default=None,
        help="见 --continuous_eval_node_rank_start。",
    )
    parser.add_argument(
        "--icl_num_demos",
        type=int,
        default=0,
        help="真·ICL：在 prompt 中从 train 区（空间 ZS 时仅 train_nodes 列）检索 top-K 形状相似窗口，写入 History+Future 示例；0=关闭。",
    )
    parser.add_argument(
        "--icl_max_candidates",
        type=int,
        default=5000,
        help="ICL 检索时随机候选 (t,节点) 数量上限（越大越准、越慢）。",
    )
    parser.add_argument(
        "--rag_use_full_grid",
        action="store_true",
        help="空间 ZS 下仍用全图节点建 RAG Ref（默认仅用 train_nodes 列，与训练一致、避免形态泄漏）。",
    )
    parser.add_argument(
        "--eval_llm_hidden_regression",
        action="store_true",
        help="额外评估：traffic+文本 embed → LLM 单次前向 → 末 token hidden → pred_head（与 predict_continuous 对比）。",
    )
    parser.add_argument(
        "--llm_hidden_max_batches",
        type=int,
        default=0,
        help="LLM hidden 评估最多跑多少个 batch（0=全量 test_loader）。",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="SCoRe",
        help="保存结果 JSON 时的模型简称，文件名为 {model_name}_results.json。",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="评测输出根目录：{model}_results.json、baseline_results_deep_eval.json、evaluation_*.json/png 等均写入此处（默认 results）。",
    )
    parser.add_argument(
        "--eval_test_node_fraction",
        type=float,
        default=None,
        help="空间 ZS 时在未见 test 节点池中再随机保留的比例（如 0.1）；默认用 config.EVAL_TEST_NODE_FRACTION 或环境变量。",
    )
    parser.add_argument(
        "--eval_test_node_seed",
        type=int,
        default=913,
        help="test 节点子采样随机种子（与 spatial_split 无关，仅影响 eval 子集）。",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="不生成任何 matplotlib 图（eval_SCoRe 推荐）。",
    )
    parser.add_argument(
        "--deep_eval_json_name",
        type=str,
        default=None,
        help="仅 --deep_baselines_only（或写 baseline JSON 时）：汇总 JSON 的文件名（可含或不含 .json），"
        "写入 results_dir；不设则默认 baseline_results_deep_eval.json（或带 rank 后缀）。",
    )
    parser.add_argument(
        "--save_results_json",
        action="store_true",
        help="将指标与样本级 pred/gt 写入 results/{model_name}_results.json。",
    )
    args, _ = parser.parse_known_args()
    _rd = getattr(args, "results_dir", None) or "results"
    results_dir = os.path.abspath(_rd)
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 评测输出目录: {results_dir}")

    from config import EVAL_TEST_NODE_FRACTION as _EVAL_TNF

    _etf = getattr(args, "eval_test_node_fraction", None)
    if _etf is None:
        _etf = _EVAL_TNF

    # 禁止在 main() 内再 `import matplotlib.pyplot as plt`：否则 plt 会变成局部变量，
    # 当 skip_plots=False 时未赋值却全程使用 → UnboundLocalError。
    # plt 来自 eval_common（from eval_common import *）。
    if getattr(args, "skip_plots", False):
        plt.savefig = lambda *a, **k: None  # noqa: ARG005
        plt.show = lambda *a, **k: None  # noqa: ARG005

    eval_dataset_label = args.dataset_name if args.dataset_name else EVAL_DATASET_DISPLAY_NAME
    if getattr(args, "top_nodes_report", -1) == -1:
        top_nodes_report_n = EVAL_TOP_NODES_REPORT
    else:
        top_nodes_report_n = int(args.top_nodes_report)
    if getattr(args, "continuous_eval_top_nodes_only", -1) == -1:
        eval_top_only_n = EVAL_CONTINUOUS_TOP_ONLY
    else:
        eval_top_only_n = int(args.continuous_eval_top_nodes_only)

    _rs = getattr(args, "continuous_eval_node_rank_start", None)
    _re = getattr(args, "continuous_eval_node_rank_end", None)
    rank_range_ok = (
        _rs is not None
        and _re is not None
        and int(_rs) >= 1
        and int(_re) >= int(_rs)
    )
    rank_start_1 = int(_rs) if rank_range_ok else None
    rank_end_1 = int(_re) if rank_range_ok else None

    def _deep_eval_output_basename():
        n = getattr(args, "deep_eval_json_name", None)
        if n:
            s = str(n).strip()
            return s if s.endswith(".json") else f"{s}.json"
        if rank_range_ok:
            return f"baseline_results_deep_eval_rank{rank_start_1}to{rank_end_1}.json"
        return "baseline_results_deep_eval.json"

    continuous_eval_suffix = ""
    continuous_top_nodes_order = None  # Top-N 模式下按节点 MAE 从优到劣的 node_id 列表（用于连续头曲线图）

    # ---------- 仅评估深度学习 baseline（ST-LSTM / STGCN / AGCRN），不加载 LLM ----------
    if getattr(args, "deep_baselines_only", False):
        if not args.deep_baselines_root or not os.path.isdir(args.deep_baselines_root):
            print("❌ --deep_baselines_only 需同时指定有效的 --deep_baselines_root（含各 model/best.pt）")
            return
        eval_seq_len = SEQ_LEN
        eval_pred_len = PRED_LEN
        eval_steps_per_day = STEPS_PER_DAY
        print("\n📦 Loading Dataset (deep baselines only, same pipeline as full eval)...")
        mp = _model_path()
        tokenizer = AutoTokenizer.from_pretrained(
            mp,
            trust_remote_code=True,
            local_files_only=os.path.isdir(mp),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        _, _, test_loader, scaler, num_nodes, spatial_meta = get_dataloaders(
            data_path=DATA_PATH,
            tokenizer=tokenizer,
            batch_size=BATCH_SIZE,
            seq_len=eval_seq_len,
            pred_len=eval_pred_len,
            steps_per_day=eval_steps_per_day,
            num_workers=0,
            eval_test_node_fraction=_etf,
            eval_test_node_seed=int(getattr(args, "eval_test_node_seed", 913)),
        )
        print(f"✅ Test dataset loaded: {len(test_loader.dataset)} samples, num_nodes: {num_nodes}")
        if spatial_meta and spatial_meta.get("spatial_zero_shot"):
            print(
                "🛰️ 空间 Zero-shot：HA + deep_baselines_root 下各 torch 模型（如 ST-LSTM / STGCN / AGCRN / DCRNN / "
                "Graph WaveNet / ASTGCN / STTN，见 run_baseline.py --spatial_zero_shot）。"
            )
        node_ids_filter = None
        if rank_range_ok:
            if not spatial_meta or not spatial_meta.get("spatial_zero_shot"):
                print("❌ 节点排名区间仅支持空间 Zero-shot（TRAFFIC_RL_SPATIAL_ZERO_SHOT=1）。")
                return
            _ckpt_rank = getattr(args, "checkpoint_path", None) or CHECKPOINT_PATH
            if not _ckpt_rank or not os.path.isdir(_ckpt_rank):
                print(
                    "❌ 排名区间模式需 --checkpoint_path 指向含 pred_head 的目录："
                    "仅用于按连续头逐节点 MAE 排序，不保存/不输出我们的方法主结果。"
                )
                return
            print(
                "\n🔮 仅加载 checkpoint 跑一次连续头前向以确定节点 MAE 排名（不写 SCoRe 结果、不跑文本生成）..."
            )
            rank_model = load_model(
                _ckpt_rank,
                DEVICE,
                num_nodes=num_nodes,
                seq_len=eval_seq_len,
                steps_per_day=eval_steps_per_day,
                continuous_head_rank_only=True,
            )
            _ph_ok = getattr(rank_model, "pred_head", None) is not None or (
                getattr(rank_model, "_use_flatten_head", False)
                and getattr(rank_model, "pred_head_flatten", None) is not None
            )
            if not _ph_ok:
                print("❌ checkpoint 中无 pred_head / pred_head_flatten，无法按连续头排名。")
                return
            rank_pack = _continuous_head_forward_collect(rank_model, test_loader, scaler, DEVICE)
            try:
                del rank_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            npc = rank_pack["node_pred_chunks"]
            if not npc:
                print("❌ 连续头未产生预测，无法排名。")
                return
            per_node_mae_bn = {}
            for _nid in npc:
                p = np.concatenate(npc[_nid])
                g = np.concatenate(rank_pack["node_gt_chunks"][_nid])
                per_node_mae_bn[_nid] = float(np.mean(np.abs(p - g)))
            ranked_bn = sorted(per_node_mae_bn.keys(), key=lambda x: per_node_mae_bn[x])
            if rank_start_1 > len(ranked_bn):
                print(f"❌ rank_start={rank_start_1} 超过可评节点数 {len(ranked_bn)}")
                return
            slice_nodes = ranked_bn[rank_start_1 - 1 : rank_end_1]
            if not slice_nodes:
                print(f"❌ 排名区间 {rank_start_1}～{rank_end_1} 为空。")
                return
            node_ids_filter = set(slice_nodes)
            print(
                f"📌 Baseline 仅聚合：连续头节点 MAE 第 {rank_start_1}～{min(rank_end_1, len(ranked_bn))} 名，"
                f"共 {len(node_ids_filter)} 个节点。"
            )
        deep_data = run_deep_baselines_eval(
            args.deep_baselines_root,
            test_loader,
            scaler,
            num_nodes,
            DEVICE,
            eval_node_id=0,
            spatial_meta=spatial_meta,
            node_ids_filter=node_ids_filter,
        )
        if not deep_data:
            print("❌ 未找到任何有效 baseline 权重，请检查 --deep_baselines_root")
            return
        out_json = os.path.join(results_dir, _deep_eval_output_basename())
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(deep_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已保存 {out_json}")
        if not getattr(args, "skip_plots", False):
            plot_mae_rmse_vs_horizon(
                None,
                None,
                baseline_data=deep_data,
                dataset_name=eval_dataset_label,
                pred_len=eval_pred_len,
                save_path=os.path.join(results_dir, "evaluation_mae_rmse_horizon.png"),
                include_ours=False,
            )
        print("\n" + "=" * 60)
        print("✅ Deep baselines evaluation complete (no LLM).")
        print("=" * 60)
        return

    # 1. 查找最佳 checkpoint
    print("\n📂 Step 1: Finding Best Checkpoint...")
    if args.checkpoint_path:
        if not os.path.isdir(args.checkpoint_path):
            print(f"❌ --checkpoint_path 不是有效目录: {args.checkpoint_path}")
            print("   请改为实际路径（内含 adapter / pred_head 等），例如:")
            print("     --checkpoint_path ./checkpoints_score_telecom/best_mae")
            print("   或设置环境变量 TRAFFIC_RL_EVAL_CHECKPOINT_PATH 指向该目录。")
            print("   若仅评估 HA + baseline、不加载主模型与连续头，请使用:")
            print("     --deep_baselines_only --deep_baselines_root ./baseline_checkpoints/<run_tag>")
            return
        checkpoint_path = args.checkpoint_path
        print(f"✅ Using explicit checkpoint_path: {checkpoint_path}")
    else:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir)
    meta = load_checkpoint_meta(checkpoint_path)
    eval_seq_len = SEQ_LEN
    eval_pred_len = PRED_LEN
    eval_steps_per_day = STEPS_PER_DAY
    if meta:
        eval_seq_len = meta.get("seq_len", SEQ_LEN)
        eval_pred_len = meta.get("pred_len", PRED_LEN)
        eval_steps_per_day = meta.get("steps_per_day", STEPS_PER_DAY)
        print(f"✅ Using checkpoint meta: seq_len={eval_seq_len}, pred_len={eval_pred_len}, steps_per_day={eval_steps_per_day}")
    
    # 2. 加载数据
    print("\n📦 Step 2: Loading Dataset...")
    model_path = _model_path()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=os.path.isdir(model_path),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    _, _, test_loader, scaler, num_nodes, spatial_meta = get_dataloaders(
        data_path=DATA_PATH,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        seq_len=eval_seq_len,
        pred_len=eval_pred_len,
        steps_per_day=eval_steps_per_day,
        num_workers=0,
        eval_test_node_fraction=_etf,
        eval_test_node_seed=int(getattr(args, "eval_test_node_seed", 913)),
    )
    
    print(f"✅ Test dataset loaded: {len(test_loader.dataset)} samples, num_nodes: {num_nodes}")
    print(f"✅ Scaler loaded - Mean shape: {np.array(scaler.mean).shape}, Std shape: {np.array(scaler.std).shape}")

    icl_tr_norm = None
    if getattr(args, "icl_num_demos", 0) > 0:
        try:
            tn_icl = None
            if spatial_meta and spatial_meta.get("spatial_zero_shot"):
                tn_icl = spatial_meta.get("train_nodes")
            icl_pack = _load_icl_train_matrix_normalized(
                DATA_PATH,
                train_nodes=tn_icl,
                seq_len=eval_seq_len,
                pred_len=eval_pred_len,
            )
            icl_tr_norm = icl_pack["tr_norm"]
            print(
                f"✅ ICL train matrix (train time slice only): shape={icl_tr_norm.shape}, "
                f"columns={'train_nodes' if tn_icl else 'full grid'}"
            )
        except Exception as e:
            print(f"⚠️ ICL matrix load failed: {e}")

    if spatial_meta and spatial_meta.get("spatial_zero_shot"):
        print(
            "🛰️ 当前为空间 Zero-shot：test 仅含未见节点列。"
            "若提供 --deep_baselines_root，将加载 spatial 版 baseline（HA + torch，见 run_baseline.py）。"
        )

    # 3. 加载模型（需在「Top-N 节点与 baseline 对齐」时先于 baseline 跑连续头）
    print("\n🤖 Step 3: Loading Model...")
    _continuous_rank_only = bool(getattr(args, "continuous_only", False)) and not bool(
        getattr(args, "eval_llm_hidden_regression", False)
    )
    if _continuous_rank_only:
        print("📌 continuous_only：以 continuous_head_rank_only 加载（不加载 Qwen），便于低显存评连续头。")
    model = load_model(
        checkpoint_path,
        DEVICE,
        num_nodes=num_nodes,
        seq_len=eval_seq_len,
        steps_per_day=eval_steps_per_day,
        continuous_head_rank_only=_continuous_rank_only,
    )
    print("✅ Model loaded successfully")

    has_continuous_head = getattr(model, "pred_head", None) is not None or (
        getattr(model, "_use_flatten_head", False)
        and getattr(model, "pred_head_flatten", None) is not None
    )

    precomputed_continuous_pack = None
    deep_baseline_data = None
    deep_baselines_aligned_to_topn = False
    if args.deep_baselines_root and os.path.isdir(args.deep_baselines_root):
        align_baseline_to_ours_topn = (
            getattr(args, "continuous_only", False)
            and (eval_top_only_n > 0 or rank_range_ok)
            and has_continuous_head
            and spatial_meta
            and spatial_meta.get("spatial_zero_shot")
        )
        if align_baseline_to_ours_topn:
            if rank_range_ok:
                print(
                    f"\n🔮 预跑连续头以按节点 MAE 排名，取第 {rank_start_1}～{rank_end_1} 名节点，"
                    "HA / torch baseline 将仅在相同 node_id 子集上聚合。"
                )
            else:
                print(
                    f"\n🔮 预跑连续头以确定 Ours 的 Top-{eval_top_only_n} 节点，"
                    "HA / torch baseline 将仅在相同 node_id 子集上聚合（与 Ours 可比）。"
                )
            precomputed_continuous_pack = _continuous_head_forward_collect(model, test_loader, scaler, DEVICE)
            npc = precomputed_continuous_pack["node_pred_chunks"]
            if not npc:
                print("❌ 连续头未产生预测，无法对齐 baseline。")
                return
            per_node_mae_bn = {}
            for _nid in npc:
                p = np.concatenate(npc[_nid])
                g = np.concatenate(precomputed_continuous_pack["node_gt_chunks"][_nid])
                per_node_mae_bn[_nid] = float(np.mean(np.abs(p - g)))
            ranked_bn = sorted(per_node_mae_bn.keys(), key=lambda x: per_node_mae_bn[x])
            if rank_range_ok:
                if rank_start_1 > len(ranked_bn):
                    print(f"❌ rank_start={rank_start_1} 超过可评节点数 {len(ranked_bn)}")
                    return
                # 第 rank_start～rank_end 名（含）：下标 [rank_start-1 : rank_end]
                slice_nodes = ranked_bn[rank_start_1 - 1 : rank_end_1]
                if not slice_nodes:
                    print(
                        f"❌ 排名区间 {rank_start_1}～{rank_end_1} 在 {len(ranked_bn)} 个节点下为空，请放宽区间。"
                    )
                    return
                keep_nodes_for_baseline = set(slice_nodes)
                print(
                    f"📌 连续头节点 MAE 排名：取第 {rank_start_1}～{min(rank_end_1, len(ranked_bn))} 名，"
                    f"共 {len(keep_nodes_for_baseline)} 个节点。"
                )
            else:
                n_keep_bn = min(eval_top_only_n, len(ranked_bn))
                keep_nodes_for_baseline = set(ranked_bn[:n_keep_bn])
            deep_baseline_data = run_deep_baselines_eval(
                args.deep_baselines_root,
                test_loader,
                scaler,
                num_nodes,
                DEVICE,
                eval_node_id=0,
                spatial_meta=spatial_meta,
                node_ids_filter=keep_nodes_for_baseline,
            )
            deep_baselines_aligned_to_topn = True
        else:
            deep_baseline_data = run_deep_baselines_eval(
                args.deep_baselines_root,
                test_loader,
                scaler,
                num_nodes,
                DEVICE,
                eval_node_id=0,
                spatial_meta=spatial_meta,
                node_ids_filter=None,
            )
            if (
                (eval_top_only_n > 0 or rank_range_ok)
                and spatial_meta
                and spatial_meta.get("spatial_zero_shot")
                and args.deep_baselines_root
            ):
                print(
                    "⚠️ 当前未同时满足 continuous_only + 连续头 +（--continuous_eval_top_nodes_only 或 排名区间）："
                    "baseline 仍为全 test；若 Ours 使用子集，Horizon 与 baseline 不可直接比。"
                )
        if deep_baseline_data:
            _bp = os.path.join(results_dir, _deep_eval_output_basename())
            with open(_bp, "w", encoding="utf-8") as f:
                json.dump(deep_baseline_data, f, indent=2, ensure_ascii=False)
            print(f"✅ {_bp} 已写入（将与 Ours 一并画入 Horizon 图）")
    if getattr(args, "continuous_only", False):
        if not has_continuous_head:
            print("❌ --continuous_only 需要 checkpoint 中存在 pred_head.pt 或 pred_head_flatten.pt")
            return
        print("\n⏭️  --continuous_only：跳过 LLM 文本生成与 RAG，直接进入连续头 Step 4b")
    
    # 4. 推理（prompt 与训练对齐时可加载 RAG；生成可配置采样）
    print("\n🔮 Step 4: Running Inference...")
    rag_ctx = None
    if EVAL_USE_RAG_PROMPT and not getattr(args, "continuous_only", False):
        try:
            if (
                spatial_meta
                and spatial_meta.get("spatial_zero_shot")
                and not getattr(args, "rag_use_full_grid", False)
            ):
                tn_rag = spatial_meta.get("train_nodes")
                if tn_rag:
                    rag_ctx = _build_rag_pool_spatial_train_nodes(
                        DATA_PATH, tn_rag, steps_per_day=eval_steps_per_day
                    )
                    print(
                        f"✅ RAG pool (spatial train cols): T={rag_ctx['pool_T']}, "
                        f"N={rag_ctx['pool_N']}, train_days={rag_ctx['train_days']}"
                    )
                else:
                    rag_ctx = _build_rag_pool_for_eval(DATA_PATH, steps_per_day=eval_steps_per_day)
                    print(
                        f"✅ RAG pool for eval: T={rag_ctx['pool_T']}, N={rag_ctx['pool_N']}, "
                        f"train_days={rag_ctx['train_days']}"
                    )
            else:
                rag_ctx = _build_rag_pool_for_eval(DATA_PATH, steps_per_day=eval_steps_per_day)
                print(
                    f"✅ RAG pool for eval: T={rag_ctx['pool_T']}, N={rag_ctx['pool_N']}, "
                    f"train_days={rag_ctx['train_days']}"
                )
        except Exception as e:
            print(f"⚠️ RAG pool build failed, fallback to no-Ref prompt: {e}")
    all_preds_norm = []  # 归一化的预测值
    all_gts_norm = []    # 归一化的真实值
    all_preds_real = []  # 真实尺度的预测值
    all_gts_real = []    # 真实尺度的真实值

    if not getattr(args, "continuous_only", False):
        if EVAL_DO_SAMPLE:
            print(f"✅ Generation: do_sample=True, temperature={EVAL_TEMPERATURE}, top_p={EVAL_TOP_P}")
        else:
            print("✅ Generation: do_sample=False (greedy, 与 validate 一致)")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inferencing")):
                history = batch['history'].to(DEVICE)
                input_ids_raw = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                gt_norm = batch['target'].to(DEVICE)  # (B, 12, Nodes)
                time_idx = batch['time_idx'].to(DEVICE)
                node_idx = batch['node_idx'].to(DEVICE)

                # 构造 prompt（EVAL_USE_RAG_PROMPT=True 时与 train_SCoRe 一致含 Similar Pattern (Ref)）
                prompts = enrich_prompt_with_urban_context(
                    tokenizer,
                    input_ids_raw,
                    time_idx,
                    history,
                    pred_len=eval_pred_len,
                    rag_ctx=rag_ctx,
                    batch_idx=batch_idx,
                    use_rag=EVAL_USE_RAG_PROMPT and rag_ctx is not None,
                    steps_per_day=eval_steps_per_day,
                    icl_num_demos=getattr(args, "icl_num_demos", 0),
                    icl_tr_norm=icl_tr_norm,
                    seq_len=eval_seq_len,
                    icl_max_candidates=getattr(args, "icl_max_candidates", 5000),
                    node_idx=node_idx,
                    spatial_zero_shot=bool(
                        spatial_meta and spatial_meta.get("spatial_zero_shot")
                    ),
                )
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)

                # 准备 history（只取第一个节点）
                history_input = history[:, :, 0:1]  # (B, 12, 1)

                # 生成（传入 attention_mask 以正确屏蔽 padding，避免左填充被参与计算）
                try:
                    gen_kwargs = dict(
                        history=history_input,
                        time_idx=time_idx,
                        node_idx=node_idx,
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=MAX_NEW_TOKENS,
                        min_new_tokens=10,
                        pad_token_id=tokenizer.pad_token_id,
                        do_sample=EVAL_DO_SAMPLE,
                    )
                    if EVAL_DO_SAMPLE:
                        gen_kwargs["temperature"] = EVAL_TEMPERATURE
                        gen_kwargs["top_p"] = EVAL_TOP_P
                    output_ids = model.generate(**gen_kwargs)

                    # 解码生成的文本
                    generated_texts = tokenizer.batch_decode(
                        output_ids[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )

                    # 处理每个样本
                    B = gt_norm.shape[0]
                    for i in range(B):
                        # 解析预测值
                        pred_norm = parse_prediction_text(generated_texts[i], pred_len=eval_pred_len)

                        if pred_norm is None:
                            # 解析失败，跳过
                            continue

                        # 获取真实值（归一化，取第一个节点）
                        gt_norm_single = gt_norm[i, :, 0].detach().cpu().numpy()  # (12,)
                        nid = int(node_idx[i, 0].item())
                        pred_real = scaler.inverse_transform(np.array(pred_norm).reshape(1, -1), node_id=nid).flatten()
                        gt_real = scaler.inverse_transform(gt_norm_single.reshape(1, -1), node_id=nid).flatten()

                        # 与 train_SCoRe validate 一致：排除 MAE>1e6 的异常样本，避免解析噪声主导指标
                        mae_i = np.mean(np.abs(pred_real - gt_real))
                        if mae_i >= 1e6:
                            continue

                        # 收集结果
                        all_preds_norm.append(pred_norm)
                        all_gts_norm.append(gt_norm_single)
                        all_preds_real.append(pred_real)
                        all_gts_real.append(gt_real)

                except Exception as e:
                    print(f"⚠️ Error in batch {batch_idx}: {e}")
                    continue
    else:
        print("⏭️  已跳过 Step 4（LLM 文本生成），节省大量时间")

    # 4b. 连续预测头（若 checkpoint 含 pred_head.pt）：直接回归 12 维，通常更贴趋势、MAE 可更低
    mae_h_ours_cont, rmse_h_ours_cont = None, None
    all_preds_real_cont = []
    all_gts_real_cont = []
    all_node_ids_cont = []
    cont_head_ok = getattr(model, "pred_head", None) is not None or (
        getattr(model, "_use_flatten_head", False) and getattr(model, "pred_head_flatten", None) is not None
    )
    if cont_head_ok:
        if precomputed_continuous_pack is not None:
            print("\n🔮 Step 4b: 使用预跑连续头结果（与 HA/torch baseline 子集一致，不重复前向）")
            all_preds_real_cont = list(precomputed_continuous_pack["all_preds_real_cont"])
            all_gts_real_cont = list(precomputed_continuous_pack["all_gts_real_cont"])
            all_node_ids_cont = list(precomputed_continuous_pack["all_node_ids_cont"])
            node_pred_chunks = precomputed_continuous_pack["node_pred_chunks"]
            node_gt_chunks = precomputed_continuous_pack["node_gt_chunks"]
        else:
            print("\n🔮 Step 4b: Continuous head inference (pred_head)...")
            node_pred_chunks = defaultdict(list)
            node_gt_chunks = defaultdict(list)
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Continuous"):
                    history = batch["history"].to(DEVICE)
                    gt_norm = batch["target"].to(DEVICE)
                    time_idx = batch["time_idx"].to(DEVICE)
                    node_idx = batch["node_idx"].to(DEVICE)
                    history_input = history[:, :, 0:1]
                    hj = batch.get("history_joint")
                    if hj is not None:
                        hj = hj.to(DEVICE)
                    pred_norm_t = model.predict_continuous(
                        history_input, time_idx, node_idx, history_joint=hj
                    )
                    if pred_norm_t is None:
                        continue
                    pred_norm_t = pred_norm_t.detach().cpu().float().numpy()
                    B = pred_norm_t.shape[0]
                    for i in range(B):
                        pred_norm = pred_norm_t[i]
                        gt_norm_single = gt_norm[i, :, 0].detach().cpu().numpy()
                        nid = int(node_idx[i, 0].item())
                        pred_real = scaler.inverse_transform(
                            np.array(pred_norm).reshape(1, -1), node_id=nid
                        ).flatten()
                        gt_real = scaler.inverse_transform(
                            gt_norm_single.reshape(1, -1), node_id=nid
                        ).flatten()
                        all_preds_real_cont.append(pred_real)
                        all_gts_real_cont.append(gt_real)
                        all_node_ids_cont.append(nid)
                        node_pred_chunks[nid].append(pred_real)
                        node_gt_chunks[nid].append(gt_real)
        if (eval_top_only_n > 0 or rank_range_ok) and len(all_preds_real_cont) > 0:
            per_node_mae = {}
            for nid in node_pred_chunks:
                p = np.concatenate(node_pred_chunks[nid])
                g = np.concatenate(node_gt_chunks[nid])
                per_node_mae[nid] = float(np.mean(np.abs(p - g)))
            ranked = sorted(per_node_mae.keys(), key=lambda x: per_node_mae[x])
            if rank_range_ok:
                if rank_start_1 > len(ranked):
                    print(f"❌ rank_start={rank_start_1} 超过可评节点数 {len(ranked)}")
                    return
                slice_ranked = ranked[rank_start_1 - 1 : rank_end_1]
                if not slice_ranked:
                    print(f"❌ 排名区间 {rank_start_1}～{rank_end_1} 无节点，请检查 --continuous_eval_node_rank_*。")
                    return
                keep_nodes = set(slice_ranked)
                continuous_top_nodes_order = list(slice_ranked)
                n_keep = len(keep_nodes)
                continuous_eval_suffix = f"_rank{rank_start_1}to{rank_end_1}"
                print(
                    f"📌 连续头输出已限制为：节点 MAE 第 {rank_start_1}～{min(rank_end_1, len(ranked))} 名上的全部窗口 "
                    f"（共 {len(keep_nodes)} 个节点，待过滤样本数见下）。"
                )
            else:
                n_keep = min(eval_top_only_n, len(ranked))
                keep_nodes = set(ranked[:n_keep])
                continuous_top_nodes_order = ranked[:n_keep]
                continuous_eval_suffix = f"_top{n_keep}nodes"
                print(
                    f"📌 连续头输出已限制为：按节点 MAE 最优的前 {n_keep} 个节点上的全部窗口 "
                    f"（共 {len(all_preds_real_cont)} 条样本）。"
                )
            nf_preds = []
            nf_gts = []
            nf_nids = []
            nf_p = defaultdict(list)
            nf_g = defaultdict(list)
            for p, g, nid in zip(all_preds_real_cont, all_gts_real_cont, all_node_ids_cont):
                if nid in keep_nodes:
                    nf_preds.append(p)
                    nf_gts.append(g)
                    nf_nids.append(nid)
                    nf_p[nid].append(p)
                    nf_g[nid].append(g)
            all_preds_real_cont = nf_preds
            all_gts_real_cont = nf_gts
            all_node_ids_cont = nf_nids
            node_pred_chunks = nf_p
            node_gt_chunks = nf_g
            if rank_range_ok:
                print(
                    f"📌 连续头过滤后样本数：{len(all_preds_real_cont)}（排名区间 {rank_start_1}～{rank_end_1}）。"
                )
            if deep_baseline_data and not deep_baselines_aligned_to_topn:
                print(
                    "⚠️ Horizon 图中 Ours 为上述子集；deep baseline 仍为全 test 上结果，横向对比仅作参考。"
                )
            if len(all_preds_real_cont) == 0:
                print("❌ 筛选后无连续头样本，请检查 --continuous_eval_top_nodes_only 与数据。")
                return

        if len(all_preds_real_cont) > 0:
            flat_pc = np.concatenate(all_preds_real_cont)
            flat_gc = np.concatenate(all_gts_real_cont)
            mae_c = np.mean(np.abs(flat_pc - flat_gc))
            rmse_c = np.sqrt(np.mean((flat_pc - flat_gc) ** 2))
            _tag = (
                "（节点 MAE 排名区间子集）"
                if rank_range_ok and continuous_eval_suffix
                else ("（仅最优 N 节点子集）" if continuous_eval_suffix else "")
            )
            print(
                f"✅ Continuous head — MAE: {mae_c:.2f}, RMSE: {rmse_c:.2f}{_tag} "
                f"(Horizon 图与 baseline 对比时以连续头为准)"
            )
            _report_continuous_top_nodes_by_column(
                DATA_PATH,
                spatial_meta,
                node_pred_chunks,
                node_gt_chunks,
                top_nodes_report_n,
                out_json=os.path.join(
                    results_dir, f"evaluation_top_nodes_continuous{continuous_eval_suffix}.json"
                ),
            )
            # 保存连续头专用对比图
            sns.set_style("whitegrid")
            # 勿与 Step 6 的 rng 同名：后者赋值会使 rng 变为整函数局部变量，导致此处 else 分支 UnboundLocalError
            rng_cont = np.random.default_rng(43)
            if continuous_top_nodes_order is not None and len(continuous_top_nodes_order) > 0:
                idxs = _pick_representative_indices_per_node(
                    all_node_ids_cont,
                    all_preds_real_cont,
                    all_gts_real_cont,
                    continuous_top_nodes_order,
                )
                used_per_node_windows = len(idxs) > 0
                if len(idxs) == 0:
                    n_show = min(5, len(all_preds_real_cont))
                    idxs = list(rng_cont.choice(len(all_preds_real_cont), n_show, replace=False))
                _w = 12 if len(idxs) > 5 else 10
                _h = 6 if len(idxs) > 5 else 5
                fig_c, ax_c = plt.subplots(1, 1, figsize=(_w, _h))
                if used_per_node_windows:
                    colors = plt.cm.tab10(np.linspace(0, 1, max(len(idxs), 1)))
                    for k, idx in enumerate(idxs):
                        nid = int(all_node_ids_cont[idx])
                        c = colors[k % 10]
                        ax_c.plot(
                            all_gts_real_cont[idx],
                            color=c,
                            label=f"GT node {nid}",
                            alpha=0.9,
                            linestyle="-",
                            linewidth=1.6,
                        )
                        ax_c.plot(
                            all_preds_real_cont[idx],
                            color=c,
                            label=f"Pred node {nid}",
                            alpha=0.9,
                            linestyle="--",
                            linewidth=1.6,
                        )
                    _sub_title = (
                        f" — one window per Top-{len(idxs)} node (median-MAE window per node)"
                    )
                else:
                    for k, idx in enumerate(idxs):
                        ax_c.plot(
                            all_gts_real_cont[idx],
                            label=f"GT #{idx}",
                            alpha=0.8,
                            linestyle="-",
                        )
                        ax_c.plot(
                            all_preds_real_cont[idx],
                            label=f"Pred #{idx}",
                            alpha=0.8,
                            linestyle="--",
                        )
                    _sub_title = " — random windows (per-node pick failed)"
            else:
                fig_c, ax_c = plt.subplots(1, 1, figsize=(10, 5))
                n_show = min(5, len(all_preds_real_cont))
                idxs = rng_cont.choice(len(all_preds_real_cont), n_show, replace=False)
                for k, idx in enumerate(idxs):
                    ax_c.plot(
                        all_gts_real_cont[idx],
                        label=f"GT #{idx}",
                        alpha=0.8,
                        linestyle="-",
                    )
                    ax_c.plot(
                        all_preds_real_cont[idx],
                        label=f"Pred #{idx}",
                        alpha=0.8,
                        linestyle="--",
                    )
                _sub_title = ""
            ax_c.set_title(
                f"Continuous head (pred_head) — MAE={mae_c:.2f}, RMSE={rmse_c:.2f}"
                + (f" [{continuous_eval_suffix.strip('_')}]" if continuous_eval_suffix else "")
                + _sub_title
            )
            ax_c.set_xlabel("Time Steps")
            ax_c.set_ylabel("Traffic Volume (Real Scale)")
            ax_c.legend(loc="best", fontsize=8)
            ax_c.grid(True, alpha=0.3)
            plt.tight_layout()
            _fn_cont = os.path.join(
                results_dir, f"evaluation_result_continuous{continuous_eval_suffix}.png"
            )
            plt.savefig(_fn_cont, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Saved {_fn_cont}")

            # Top-N 模式：每个选中节点各画「该节点内」样本 MAE 最小的窗口；否则：全局样本 MAE 最小的 K 条
            if continuous_top_nodes_order is not None and len(continuous_top_nodes_order) > 0:
                best_indices = _pick_best_sample_index_per_node(
                    all_node_ids_cont,
                    all_preds_real_cont,
                    all_gts_real_cont,
                    continuous_top_nodes_order,
                )
                K = len(best_indices)
                per_sample_mae = []
                for idx in range(len(all_preds_real_cont)):
                    p = np.asarray(all_preds_real_cont[idx], dtype=np.float64)
                    g = np.asarray(all_gts_real_cont[idx], dtype=np.float64)
                    per_sample_mae.append(np.mean(np.abs(p - g)))
                per_sample_mae = np.array(per_sample_mae)
            else:
                K = min(NUM_CONTINUOUS_BEST_TO_PLOT, len(all_preds_real_cont))
                per_sample_mae = []
                for idx in range(len(all_preds_real_cont)):
                    p = np.asarray(all_preds_real_cont[idx], dtype=np.float64)
                    g = np.asarray(all_gts_real_cont[idx], dtype=np.float64)
                    per_sample_mae.append(np.mean(np.abs(p - g)))
                per_sample_mae = np.array(per_sample_mae)
                best_indices = list(np.argsort(per_sample_mae)[:K]) if K >= 1 else []

            if K >= 1 and len(best_indices) > 0:
                fig_best, axes = plt.subplots(
                    K, 1, figsize=(10, max(3.2, min(4, 36 / max(K, 1))) * K), squeeze=False
                )
                for rank, idx in enumerate(best_indices):
                    ax_b = axes[rank, 0]
                    gt = np.asarray(all_gts_real_cont[idx])
                    pr = np.asarray(all_preds_real_cont[idx])
                    steps = np.arange(len(gt))
                    ax_b.plot(steps, gt, "o-", label="Ground Truth", color="#1f77b4", linewidth=2, markersize=6)
                    ax_b.plot(steps, pr, "s--", label="Prediction", color="#ff7f0e", linewidth=2, markersize=5)
                    mae_i = per_sample_mae[idx]
                    rmse_i = np.sqrt(np.mean((pr - gt) ** 2))
                    if continuous_top_nodes_order is not None:
                        _nid = int(all_node_ids_cont[idx])
                        ax_b.set_title(
                            f"node {_nid} — best window in this node  MAE={mae_i:.2f}, RMSE={rmse_i:.2f}"
                        )
                    else:
                        ax_b.set_title(
                            f"Best #{rank+1} (sample index {idx})  MAE={mae_i:.2f}, RMSE={rmse_i:.2f}"
                        )
                    ax_b.set_xlabel("Time Steps")
                    ax_b.set_ylabel("Traffic Volume (Real Scale)")
                    ax_b.legend(loc="best")
                    ax_b.grid(True, alpha=0.3)
                _sub = (
                    f"{continuous_eval_suffix.replace('_', '')}"
                    if continuous_eval_suffix
                    else "full test"
                )
                if continuous_top_nodes_order is not None:
                    _st = (
                        f"Continuous head — best window per Top-{K} node "
                        f"(min sample MAE within each node) ({_sub}: MAE={mae_c:.2f}, RMSE={rmse_c:.2f})"
                    )
                else:
                    _st = (
                        f"Continuous head — top {K} samples by per-sample MAE "
                        f"({_sub}: MAE={mae_c:.2f}, RMSE={rmse_c:.2f})"
                    )
                fig_best.suptitle(_st, fontsize=11, y=1.02)
                plt.tight_layout()
                out_best = os.path.join(
                    results_dir, f"evaluation_result_continuous_best{continuous_eval_suffix}.png"
                )
                plt.savefig(out_best, dpi=300, bbox_inches="tight")
                plt.close()
                if continuous_top_nodes_order is not None:
                    print(f"✅ Saved {out_best} (每节点一条：节点内样本 MAE 最小)")
                else:
                    print(f"✅ Saved {out_best} (top {K} by sample MAE)")
            mae_h_ours_cont, rmse_h_ours_cont = _compute_per_horizon_metrics(
                all_preds_real_cont, all_gts_real_cont, eval_pred_len
            )
            print("📌 Horizon 图中「Ours」= 连续头 predict_continuous（与 baseline 对比，非文本生成）。")

    n_llm_hidden_eval = 0
    mae_ph_llm_hidden = None
    rmse_ph_llm_hidden = None
    all_pr_h = []
    all_gt_h = []
    all_nid_h = []
    if getattr(args, "eval_llm_hidden_regression", False):
        if not cont_head_ok:
            print("⚠️ --eval_llm_hidden_regression 需要 checkpoint 中的 pred_head / pred_head_flatten，已跳过。")
        else:
            print("\n🔮 Step 4c: LLM last-hidden + pred_head（单次前向，与 predict_continuous 对比）...")
            rag_ctx_llm = rag_ctx
            if rag_ctx_llm is None and EVAL_USE_RAG_PROMPT:
                try:
                    if (
                        spatial_meta
                        and spatial_meta.get("spatial_zero_shot")
                        and not getattr(args, "rag_use_full_grid", False)
                    ):
                        tn_rag = spatial_meta.get("train_nodes")
                        if tn_rag:
                            rag_ctx_llm = _build_rag_pool_spatial_train_nodes(
                                DATA_PATH, tn_rag, steps_per_day=eval_steps_per_day
                            )
                        else:
                            rag_ctx_llm = _build_rag_pool_for_eval(
                                DATA_PATH, steps_per_day=eval_steps_per_day
                            )
                    else:
                        rag_ctx_llm = _build_rag_pool_for_eval(
                            DATA_PATH, steps_per_day=eval_steps_per_day
                        )
                    print("✅ Step 4c: 已补建 RAG ctx（用于与 Step 4 一致的 prompt）")
                except Exception as e:
                    print(f"⚠️ Step 4c RAG 构建失败，prompt 无 Ref 行: {e}")
            all_pr_h.clear()
            all_gt_h.clear()
            all_nid_h.clear()
            max_b = int(getattr(args, "llm_hidden_max_batches", 0) or 0)
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_loader, desc="LLM hidden+head")):
                    if max_b > 0 and batch_idx >= max_b:
                        break
                    history = batch["history"].to(DEVICE)
                    input_ids_raw = batch["input_ids"].to(DEVICE)
                    gt_norm = batch["target"].to(DEVICE)
                    time_idx = batch["time_idx"].to(DEVICE)
                    node_idx = batch["node_idx"].to(DEVICE)
                    prompts = enrich_prompt_with_urban_context(
                        tokenizer,
                        input_ids_raw,
                        time_idx,
                        history,
                        pred_len=eval_pred_len,
                        rag_ctx=rag_ctx_llm,
                        batch_idx=batch_idx,
                        use_rag=EVAL_USE_RAG_PROMPT and rag_ctx_llm is not None,
                        steps_per_day=eval_steps_per_day,
                        icl_num_demos=getattr(args, "icl_num_demos", 0),
                        icl_tr_norm=icl_tr_norm,
                        seq_len=eval_seq_len,
                        icl_max_candidates=getattr(args, "icl_max_candidates", 5000),
                        node_idx=node_idx,
                        spatial_zero_shot=bool(
                            spatial_meta and spatial_meta.get("spatial_zero_shot")
                        ),
                    )
                    inputs = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(DEVICE)
                    history_input = history[:, :, 0:1]
                    pred_t = model.predict_continuous_llm_hidden(
                        history_input,
                        time_idx,
                        node_idx,
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                    )
                    if pred_t is None:
                        continue
                    pred_norm_t = pred_t.detach().cpu().float().numpy()
                    B = pred_norm_t.shape[0]
                    for i in range(B):
                        pred_norm = pred_norm_t[i]
                        gt_norm_single = gt_norm[i, :, 0].detach().cpu().numpy()
                        nid = int(node_idx[i, 0].item())
                        pred_real = scaler.inverse_transform(
                            np.array(pred_norm).reshape(1, -1), node_id=nid
                        ).flatten()
                        gt_real = scaler.inverse_transform(
                            gt_norm_single.reshape(1, -1), node_id=nid
                        ).flatten()
                        all_pr_h.append(pred_real)
                        all_gt_h.append(gt_real)
                        all_nid_h.append(nid)
            if (eval_top_only_n > 0 or rank_range_ok) and len(all_pr_h) > 0:
                if continuous_top_nodes_order is not None and len(continuous_top_nodes_order) > 0:
                    keep_h = set(int(x) for x in continuous_top_nodes_order)
                    nf_p, nf_g, nf_n = [], [], []
                    for p, g, nid in zip(all_pr_h, all_gt_h, all_nid_h):
                        if nid in keep_h:
                            nf_p.append(p)
                            nf_g.append(g)
                            nf_n.append(nid)
                    all_pr_h, all_gt_h, all_nid_h = nf_p, nf_g, nf_n
                    _rk = (
                        f"排名 {rank_start_1}～{rank_end_1}"
                        if rank_range_ok
                        else f"Top-{len(keep_h)}"
                    )
                    print(f"📌 Step 4c 已与连续头对齐：{_rk} 节点（{len(all_pr_h)} 条样本）。")
                else:
                    node_pred_h = defaultdict(list)
                    node_gt_h = defaultdict(list)
                    for p, g, nid in zip(all_pr_h, all_gt_h, all_nid_h):
                        node_pred_h[nid].append(p)
                        node_gt_h[nid].append(g)
                    per_node_mae_h = {}
                    for nid in node_pred_h:
                        p = np.concatenate(node_pred_h[nid])
                        g = np.concatenate(node_gt_h[nid])
                        per_node_mae_h[nid] = float(np.mean(np.abs(p - g)))
                    ranked_h = sorted(per_node_mae_h.keys(), key=lambda x: per_node_mae_h[x])
                    n_keep_h = min(eval_top_only_n, len(ranked_h))
                    keep_h = set(ranked_h[:n_keep_h])
                    nf_p, nf_g, nf_n = [], [], []
                    for p, g, nid in zip(all_pr_h, all_gt_h, all_nid_h):
                        if nid in keep_h:
                            nf_p.append(p)
                            nf_g.append(g)
                            nf_n.append(nid)
                    all_pr_h, all_gt_h, all_nid_h = nf_p, nf_g, nf_n
                    print(
                        f"📌 Step 4c 按 LLM hidden 节点 MAE 取 Top-{n_keep_h}（{len(all_pr_h)} 条样本）。"
                    )
            n_llm_hidden_eval = len(all_pr_h)
            if n_llm_hidden_eval > 0:
                flat_ph = np.concatenate(all_pr_h)
                flat_gh = np.concatenate(all_gt_h)
                mae_llm_h = float(np.mean(np.abs(flat_ph - flat_gh)))
                rmse_llm_h = float(np.sqrt(np.mean((flat_ph - flat_gh) ** 2)))
                mae_ph_llm_hidden, rmse_ph_llm_hidden = _compute_per_horizon_metrics(
                    all_pr_h, all_gt_h, eval_pred_len
                )
                _cmp = ""
                if len(all_preds_real_cont) > 0:
                    _mae_c = float(
                        np.mean(
                            np.abs(
                                np.concatenate(all_preds_real_cont)
                                - np.concatenate(all_gts_real_cont)
                            )
                        )
                    )
                    _cmp = f" | 连续头 MAE={_mae_c:.2f}（同筛选策略下可比）"
                print(
                    f"✅ LLM hidden + pred_head — MAE: {mae_llm_h:.2f}, RMSE: {rmse_llm_h:.2f}{_cmp}"
                )
                _llm_json = os.path.join(results_dir, "evaluation_llm_hidden_regression.json")
                with open(_llm_json, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "mae": mae_llm_h,
                            "rmse": rmse_llm_h,
                            "mae_per_horizon": mae_ph_llm_hidden.tolist(),
                            "rmse_per_horizon": rmse_ph_llm_hidden.tolist(),
                            "num_samples": n_llm_hidden_eval,
                            "max_batches": max_b,
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"✅ 已写入 {_llm_json}")
            else:
                print("⚠️ Step 4c 无有效样本（检查 pred_head / 显存 / max_batches）。")

    if getattr(args, "save_results_json", False):
        _mn = getattr(args, "model_name", "SCoRe")
        _out_path = os.path.join(results_dir, f"{_mn}_results.json")
        save_scorer_results_json(
            _out_path,
            model_name=_mn,
            data_path=DATA_PATH,
            checkpoint_path=checkpoint_path,
            eval_seq_len=eval_seq_len,
            eval_pred_len=eval_pred_len,
            eval_steps_per_day=eval_steps_per_day,
            spatial_meta=spatial_meta,
            eval_test_node_fraction=_etf,
            all_preds_real=all_preds_real,
            all_gts_real=all_gts_real,
            all_preds_real_cont=all_preds_real_cont,
            all_gts_real_cont=all_gts_real_cont,
            all_node_ids_cont=all_node_ids_cont,
            all_preds_llm_h=all_pr_h,
            all_gts_llm_h=all_gt_h,
            all_node_ids_llm_h=all_nid_h if n_llm_hidden_eval > 0 else None,
            n_llm_hidden_eval=n_llm_hidden_eval,
            mae_ph_llm_hidden=mae_ph_llm_hidden,
            rmse_ph_llm_hidden=rmse_ph_llm_hidden,
        )
        print(f"✅ 结果已保存: {_out_path}")

    if getattr(args, "skip_plots", False):
        print("\n⏭️  --skip_plots：跳过剩余可视化与 Horizon 图。")
        return

    if len(all_preds_real) == 0 and len(all_preds_real_cont) == 0 and n_llm_hidden_eval == 0:
        print("❌ No valid predictions generated!")
        return
    if len(all_preds_real) == 0 and len(all_preds_real_cont) > 0:
        print("📌 文本生成无有效样本；仍输出 Horizon 图，「Ours」= 连续头。")
        baseline_data = None
        if BASELINE_RESULTS_PATH and os.path.isfile(BASELINE_RESULTS_PATH):
            with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
        if deep_baseline_data:
            if baseline_data is None:
                baseline_data = {}
            for _k, _v in deep_baseline_data.items():
                baseline_data[_k] = _v
        _hz_path = os.path.join(results_dir, f"evaluation_mae_rmse_horizon{continuous_eval_suffix}.png")
        plot_mae_rmse_vs_horizon(
            mae_h_ours_cont,
            rmse_h_ours_cont,
            baseline_data=baseline_data,
            dataset_name=eval_dataset_label,
            pred_len=eval_pred_len,
            save_path=_hz_path,
            include_ours=True,
            ours_display_name="Ours (continuous head)",
        )
        print("\n" + "=" * 60)
        print(f"✅ {_hz_path} saved (Ours = continuous head + baselines).")
        print("=" * 60)
        return
    
    # 5. 计算指标（均在「全量 test 样本 × 12 步」上计算）
    print("\n📊 Step 5: Computing Metrics...")
    flat_preds_real = np.concatenate(all_preds_real)  # shape (N*12,)
    flat_gts_real = np.concatenate(all_gts_real)

    mae = np.mean(np.abs(flat_preds_real - flat_gts_real))
    rmse = np.sqrt(np.mean((flat_preds_real - flat_gts_real) ** 2))
    # SMAPE = 100 * mean( 2|pred-gt| / (|pred|+|gt|+eps) )
    denom = np.abs(flat_preds_real) + np.abs(flat_gts_real) + 1e-8
    smape = np.mean(2.0 * np.abs(flat_preds_real - flat_gts_real) / denom) * 100

    # 按 Horizon (1~12) 计算 MAE/RMSE，用于「MAE/RMSE vs Horizon」图
    mae_per_horizon, rmse_per_horizon = _compute_per_horizon_metrics(
        all_preds_real, all_gts_real, eval_pred_len
    )
    
    print(f"\n✅ Evaluation Results:")
    print(f"   - MAE (Mean Absolute Error): {mae:.2f}")
    print(f"   - RMSE (Root Mean Square Error): {rmse:.2f}")
    print(f"   - SMAPE (Symmetric Mean Absolute Percentage Error): {smape:.2f}%")
    print(f"   - Valid Samples: {len(all_preds_real)}")
    
    # 6. 可视化（指标与图一致：MAE/RMSE/SMAPE 均为全量 test 样本的 73×12 个点计算）
    print("\n🎨 Step 6: Generating Visualizations...")
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(16, 10))
    
    # 图 1: 时间序列对比图（展示部分样本，标题为全量指标）
    ax1 = plt.subplot(2, 1, 1)
    rng_text_viz = np.random.default_rng(42)
    num_samples = min(NUM_SAMPLES_TO_VISUALIZE, len(all_preds_real))
    indices = rng_text_viz.choice(len(all_preds_real), num_samples, replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))
    
    for i, idx in enumerate(indices):
        pred = all_preds_real[idx]
        gt = all_gts_real[idx]
        x = range(len(gt))
        
        ax1.plot(x, gt, label=f'Sample {idx} GT', color=colors[i], alpha=0.7, linestyle='-', linewidth=2)
        ax1.plot(x, pred, label=f'Sample {idx} Pred', color=colors[i], linestyle='--', linewidth=2, marker='o', markersize=4)
    
    ax1.set_title(f"Time Series Comparison: Ground Truth (Solid) vs Prediction (Dashed)\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, SMAPE: {smape:.1f}% (full test set)", 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time Steps", fontsize=12)
    ax1.set_ylabel("Traffic Volume (Real Scale, Erlang)", fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 图 2: 效果最好的一组——按样本 MAE 选出一个最佳样本，只画该组的真实值 vs 预测值折线对比
    ax2 = plt.subplot(2, 1, 2)
    per_sample_mae = np.array([np.mean(np.abs(np.array(p) - np.array(g))) for p, g in zip(all_preds_real, all_gts_real)])
    best_idx = int(np.argmin(per_sample_mae))
    best_gt = all_gts_real[best_idx]
    best_pred = all_preds_real[best_idx]
    best_mae_single = per_sample_mae[best_idx]
    x_steps = np.arange(len(best_gt))
    ax2.plot(x_steps, best_gt, label='Ground Truth', color='C0', linestyle='-', linewidth=2.5, marker='o', markersize=6)
    ax2.plot(x_steps, best_pred, label='Prediction', color='C1', linestyle='--', linewidth=2.5, marker='s', markersize=6)
    ax2.set_title(f"Best Sample (Sample #{best_idx}, MAE={best_mae_single:.2f}): Ground Truth vs Prediction", 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel("Time Steps", fontsize=12)
    ax2.set_ylabel("Traffic Volume (Real Scale, Erlang)", fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    textstr = f'Full Test: MAE {mae:.2f}, RMSE {rmse:.2f}, SMAPE {smape:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.02, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(results_dir, "evaluation_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")

    # 7. 绘制「MAE / RMSE vs Horizon」双图（与论文 CHI-taxi 风格一致）
    print("\n📈 Step 7: Generating MAE/RMSE vs Horizon comparison...")
    baseline_data = None
    if BASELINE_RESULTS_PATH and os.path.isfile(BASELINE_RESULTS_PATH):
        with open(BASELINE_RESULTS_PATH, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    if deep_baseline_data:
        if baseline_data is None:
            baseline_data = {}
        for k, v in deep_baseline_data.items():
            baseline_data[k] = v
    if mae_h_ours_cont is not None:
        mae_ph = mae_h_ours_cont
        rmse_ph = rmse_h_ours_cont
        ours_nm = "Ours (continuous head)"
    else:
        mae_ph = mae_per_horizon
        rmse_ph = rmse_per_horizon
        ours_nm = "Ours (text gen)"
    _hz_path2 = os.path.join(results_dir, f"evaluation_mae_rmse_horizon{continuous_eval_suffix}.png")
    plot_mae_rmse_vs_horizon(
        mae_ph,
        rmse_ph,
        baseline_data=baseline_data,
        dataset_name=eval_dataset_label,
        pred_len=eval_pred_len,
        save_path=_hz_path2,
        include_ours=True,
        ours_display_name=ours_nm,
    )
    
    # 显示图片（仅保留原图，horizon 图已保存）
    plt.show()
    
    print("\n" + "=" * 60)
    print("✅ Evaluation Complete!")
    print("=" * 60)
