#!/usr/bin/env bash
# =============================================================================
# Traffic_RL1 — 五模块消融编排（GPU 上长跑；支持顺序 train/eval 或 train_parallel/eval_parallel）
#
# M1 ST-encoder: TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER=1  → 训练 run: m1_freeze_st_encoder
# M2 LLM(LoRA): TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA=1     → 训练 run: m2_freeze_llm_lora
# M3 RL:        TRAFFIC_RL_ENABLE_RL=0                  → 训练 run: m3_no_rl
# M4 continuous head: train_pred_head 多配置（非 train_SCoRe）→ 子命令 m4 / heads
# M5 Context:   TRAFFIC_RL_ABLATE_NO_CONTEXT_AUG=1      → 训练 run: m5_no_context_aug
#
# M1–M3、M5 + 连续指标：若仅 eval 骨干 best_mae（无 train_pred_head），continuous micro 常完全 tied。
# 需在各路 best_mae 上再训一轮与 M4「full_small_data_best」同协议的 train_pred_head，再 eval →
#   JSON tag 形如 m1_freeze_st_encoder_pred；子命令 score_heads / score_heads_parallel
#
# 默认将所有「结果」JSON、汇总 CSV、训练 checkpoint 写到本仓库 Traffic_RL1 根下：
#   ablation_results/
#   ├── ablation_*.json / ablation_summary.csv
#   └── checkpoints/{00_full,m1_*,m2_*,...}/best_mae
# 不在此脚本里改写主训练的默认 OUTPUT_DIR（config.py checkpoints_score_*）；每条 train/eval/M4 仅在子进程的
# env 内设置 TRAFFIC_RL_OUTPUT_DIR → ablation_results/checkpoints/...
# 可覆盖：
#   export TRAFFIC_RL_ABLATION_OUTPUT_ROOT=/path/to/custom_root
#   （checkpoint 默认 $ROOT/checkpoints，JSON 进该目录；或再设 TRAFFIC_RL_ABLATION_CKPT_ROOT / TRAFFIC_RL_ABLATION_RESULTS_DIR）
#
# 用法（在 Traffic_RL1 根目录）：
#   ./scripts/run_ablations.sh train    # M1+M2+M3+M5 + Full（五条 train_SCoRe，顺序单进程）
#   ./scripts/run_ablations.sh train_parallel # 同上，按 GPU 列表并行波次（见下）
#   ./scripts/run_ablations.sh eval     # 对上述 best_mae 评测
#   ./scripts/run_ablations.sh eval_parallel
#   ./scripts/run_ablations.sh m4       # M4：连续头消融（串行四条 train_pred_head + eval）
#   ./scripts/run_ablations.sh m4_parallel # M4：四条各占 TRAFFIC_RL_ABLATION_GPUS 中的一卡并行
#   ./scripts/run_ablations.sh score_heads_parallel  # M1/M2/M3/M5：各骨干上 train_pred_head + eval（与 M4 full 对齐）
#   ./scripts/run_ablations.sh collect  # 汇总 CSV
#   ./scripts/run_ablations.sh all
#   ./scripts/run_ablations.sh all_parallel  # train_parallel → eval_parallel → m4_parallel → score_heads_parallel → collect
#
# 并行训练/评测/M4（一卡一路；勿让 SCORE_LAUNCH / HEAD_LAUNCH 再启本实验 multi_gpu）：
#   export TRAFFIC_RL_ABLATION_GPUS="2,3,4,6"
#
# M4（train_pred_head）过长时每 epoch 可达 ~1h：主要为 small_data_best 默认 max_windows_train≈20000。
# 加速可调低每 epoch 步数（精度可能下降，消融对比仍可看相对次序）：
#   export TRAFFIC_RL_M4_PRED_HEAD_EXTRA_ARGS="--max_windows_train 4000 --batch_size 16"
# 可选同时缩短上限轮数： "... --epochs 80 --early_stop_patience 15"
#
# 多卡同一实验（单条消融内 DDP，与本脚本并行互斥）：一般不要用 train_parallel，
#   改用：export SCORE_LAUNCH="accelerate launch --multi_gpu --num_processes=4 ./train_SCoRe.py"
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCORE_LAUNCH="${SCORE_LAUNCH:-python train_SCoRe.py}"
HEAD_LAUNCH="${HEAD_LAUNCH:-python train_pred_head.py}"

_DEFAULT_ABL_ROOT="${ROOT}/ablation_results"
ABL_OUTPUT_ROOT="${TRAFFIC_RL_ABLATION_OUTPUT_ROOT:-${_DEFAULT_ABL_ROOT}}"
ABL_PREFIX="${TRAFFIC_RL_ABLATION_CKPT_ROOT:-${ABL_OUTPUT_ROOT}/checkpoints}"
RESULTS_JSON_DIR="${TRAFFIC_RL_ABLATION_RESULTS_DIR:-${ABL_OUTPUT_ROOT}}"

FULL_TAG="00_full"

# --- 并行用：每条子 shell 独占一张物理卡 ---
_parse_abl_gpu_list() {
  local s="${TRAFFIC_RL_ABLATION_GPUS:-2,3,4,6}"
  IFS=',' read -ra _ABL_GPU_ARR <<< "$s"
  if [[ "${#_ABL_GPU_ARR[@]}" -eq 0 ]]; then
    _die "TRAFFIC_RL_ABLATION_GPUS 为空"
  fi
  local i
  for i in "${!_ABL_GPU_ARR[@]}"; do
    _ABL_GPU_ARR[$i]="${_ABL_GPU_ARR[$i]//[[:space:]]/}"
  done
}

SCORE_TAGS=(
  "${FULL_TAG}"
  m1_freeze_st_encoder
  m2_freeze_llm_lora
  m3_no_rl
  m5_no_context_aug
)

_die() { echo "❌ $*" >&2; exit 1; }

_export_common() {
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
  export TRAFFIC_RL_SEED="${TRAFFIC_RL_SEED:-42}"
  # 供 collect_ablation_results.py 默认扫描（未传 --scan_dir 时）
  export TRAFFIC_RL_ABLATION_RESULTS_DIR="${RESULTS_JSON_DIR}"
}

_ensure_dirs() {
  mkdir -p "${RESULTS_JSON_DIR}" "${ABL_PREFIX}"
}

_clear_ablate_flags() {
  unset TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER \
        TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA \
        TRAFFIC_RL_ABLATE_NO_CONTEXT_AUG \
        TRAFFIC_RL_ENABLE_RL TRAFFIC_RL_SKIP_RL || true
}

_run_score_train() {
  local tag="$1"
  shift
  _clear_ablate_flags
  export TRAFFIC_RL_OUTPUT_DIR="${ABL_PREFIX}/${tag}"
  mkdir -p "${TRAFFIC_RL_OUTPUT_DIR}"
  echo ""
  echo "========== [train_SCoRe] ${tag} → ${TRAFFIC_RL_OUTPUT_DIR} =========="
  env "$@" ${SCORE_LAUNCH}
}

# 并行专用：物理 GPU id + tag + env 覆盖（子 shell 内绑定单卡）
_run_score_train_on_gpu() {
  local phys_gpu="$1" tag="$2"
  shift 2
  (
    cd "$ROOT" || exit 1
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    export TRAFFIC_RL_SEED="${TRAFFIC_RL_SEED:-42}"
    export CUDA_VISIBLE_DEVICES="${phys_gpu}"
    export TRAFFIC_RL_CUDA_DEVICE="${TRAFFIC_RL_CUDA_DEVICE:-0}"
    _clear_ablate_flags
    export TRAFFIC_RL_OUTPUT_DIR="${ABL_PREFIX}/${tag}"
    mkdir -p "${TRAFFIC_RL_OUTPUT_DIR}"
    echo ""
    echo "========== [train_SCoRe] GPU ${phys_gpu} ${tag} → ${TRAFFIC_RL_OUTPUT_DIR} =========="
    env "$@" ${SCORE_LAUNCH}
  )
}

_run_train_parallel() {
  _parse_abl_gpu_list
  local ng="${#_ABL_GPU_ARR[@]}"
  local i=0 j phys tag fail=0
  echo "🚀 train_parallel：每波最多 ${ng} 路，GPU 列表：${_ABL_GPU_ARR[*]}"
  while [[ "$i" -lt "${#SCORE_TAGS[@]}" ]]; do
    local pids=()
    for ((j = 0; j < ng && i < ${#SCORE_TAGS[@]}; j++, i++)); do
      phys="${_ABL_GPU_ARR[$j]}"
      tag="${SCORE_TAGS[$i]}"
      case "$tag" in
        "${FULL_TAG}")
          _run_score_train_on_gpu "$phys" "$tag" TRAFFIC_RL_ENABLE_RL=1 &
          ;;
        m1_freeze_st_encoder)
          _run_score_train_on_gpu "$phys" "$tag" TRAFFIC_RL_ENABLE_RL=1 TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER=1 &
          ;;
        m2_freeze_llm_lora)
          _run_score_train_on_gpu "$phys" "$tag" TRAFFIC_RL_ENABLE_RL=1 TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA=1 &
          ;;
        m3_no_rl)
          _run_score_train_on_gpu "$phys" "$tag" TRAFFIC_RL_ENABLE_RL=0 &
          ;;
        m5_no_context_aug)
          _run_score_train_on_gpu "$phys" "$tag" TRAFFIC_RL_ENABLE_RL=1 TRAFFIC_RL_ABLATE_NO_CONTEXT_AUG=1 &
          ;;
        *)
          _die "未知 train tag: $tag"
          ;;
      esac
      pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
  done
  [[ "$fail" -eq 0 ]] || _die "train_parallel 中至少一路失败"
  echo "✅ train_parallel 完成。checkpoint: ${ABL_PREFIX}"
}

_eval_one() {
  local tag="$1"
  local ck="${ABL_PREFIX}/${tag}/best_mae"
  [[ -d "$ck" ]] || _die "缺少 checkpoint: $ck （请先完成 train）"
  echo ""
  echo "========== [eval] ${tag} → JSON 写入 ${RESULTS_JSON_DIR} =========="
  python eval_SCoRe.py \
    --checkpoint_path "$ck" \
    --model_name "ablation_${tag}" \
    --results_dir "$RESULTS_JSON_DIR" \
    --eval_test_node_fraction 1.0 \
    --continuous_only \
    --skip_plots
}

_eval_one_on_gpu() {
  local phys_gpu="$1" tag="$2"
  local ck="${ABL_PREFIX}/${tag}/best_mae"
  [[ -d "$ck" ]] || _die "缺少 checkpoint: $ck （请先完成 train）tag=${tag}"
  (
    cd "$ROOT" || exit 1
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${phys_gpu}"
    export TRAFFIC_RL_CUDA_DEVICE="${TRAFFIC_RL_CUDA_DEVICE:-0}"
    echo ""
    echo "========== [eval] GPU ${phys_gpu} ${tag} → ${RESULTS_JSON_DIR} =========="
    python eval_SCoRe.py \
      --checkpoint_path "$ck" \
      --model_name "ablation_${tag}" \
      --results_dir "$RESULTS_JSON_DIR" \
      --eval_test_node_fraction 1.0 \
      --continuous_only \
      --skip_plots
  )
}

_run_eval_parallel() {
  _parse_abl_gpu_list
  local ng="${#_ABL_GPU_ARR[@]}"
  local i=0 j phys tag fail=0
  echo "🚀 eval_parallel：每波最多 ${ng} 路，GPU：${_ABL_GPU_ARR[*]}"
  while [[ "$i" -lt "${#SCORE_TAGS[@]}" ]]; do
    local pids=()
    for ((j = 0; j < ng && i < ${#SCORE_TAGS[@]}; j++, i++)); do
      phys="${_ABL_GPU_ARR[$j]}"
      tag="${SCORE_TAGS[$i]}"
      _eval_one_on_gpu "$phys" "$tag" &
      pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
  done
  [[ "$fail" -eq 0 ]] || _die "eval_parallel 中至少一路失败"
  echo "✅ eval_parallel 完成。JSON：${RESULTS_JSON_DIR}"
}

_full_tag_best_mae() {
  echo "${ABL_PREFIX}/${FULL_TAG}/best_mae"
}

# M4 train_pred_head 额外 CLI（空格分隔）；用于加速：如 --max_windows_train 4000 --batch_size 16
_m4_pred_extra_args() {
  if [[ -n "${TRAFFIC_RL_M4_PRED_HEAD_EXTRA_ARGS:-}" ]]; then
    read -ra _M4_PRED_EXTRA <<< "${TRAFFIC_RL_M4_PRED_HEAD_EXTRA_ARGS}"
  else
    _M4_PRED_EXTRA=()
  fi
}

_head_one() {
  local name="$1"
  shift
  local base="$(_full_tag_best_mae)"
  [[ -d "$base" ]] || _die "M4 需要 Full 的 best_mae: $base（先跑 train 中 ${FULL_TAG}）"
  local out="${ABL_PREFIX}/m4_heads/${name}"
  mkdir -p "$out"
  _m4_pred_extra_args
  echo ""
  echo "========== [M4 train_pred_head] ${name} → ${out} =========="
  ${HEAD_LAUNCH} "${_M4_PRED_EXTRA[@]}" \
    --checkpoint_dir "$base" \
    --save_dir "$out" \
    "$@"
  echo ""
  echo "========== [M4 eval] ${name} =========="
  python eval_SCoRe.py \
    --checkpoint_path "$out" \
    --model_name "ablation_m4_${name}" \
    --results_dir "$RESULTS_JSON_DIR" \
    --eval_test_node_fraction 1.0 \
    --continuous_only \
    --skip_plots
}

# M4 并行专用：同一物理 GPU 上先 train_pred_head 再 eval（与其他路互不干扰）
_head_one_on_gpu() {
  local phys_gpu="$1" name="$2"
  shift 2
  local base="$(_full_tag_best_mae)"
  [[ -d "$base" ]] || _die "M4 需要 Full 的 best_mae: $base（先跑 train 中 ${FULL_TAG}）"
  local out="${ABL_PREFIX}/m4_heads/${name}"
  mkdir -p "$out"
  (
    cd "$ROOT" || exit 1
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${phys_gpu}"
    export TRAFFIC_RL_CUDA_DEVICE="${TRAFFIC_RL_CUDA_DEVICE:-0}"
    if [[ -n "${TRAFFIC_RL_M4_PRED_HEAD_EXTRA_ARGS:-}" ]]; then
      read -ra _M4_PRED_EXTRA <<< "${TRAFFIC_RL_M4_PRED_HEAD_EXTRA_ARGS}"
    else
      _M4_PRED_EXTRA=()
    fi
    echo ""
    echo "========== [M4 train_pred_head] GPU ${phys_gpu} ${name} → ${out} =========="
    ${HEAD_LAUNCH} "${_M4_PRED_EXTRA[@]}" \
      --checkpoint_dir "$base" \
      --save_dir "$out" \
      "$@"
    echo ""
    echo "========== [M4 eval] GPU ${phys_gpu} ${name} =========="
    python eval_SCoRe.py \
      --checkpoint_path "$out" \
      --model_name "ablation_m4_${name}" \
      --results_dir "$RESULTS_JSON_DIR" \
      --eval_test_node_fraction 1.0 \
      --continuous_only \
      --skip_plots
  ) &
}

M4_PARALLEL_NAMES=(
  full_small_data_best
  no_fft
  no_skip
  no_joint
)

# 需在 train 完成后于各 tag 的 best_mae 上训连续头（与 _head_one full_small_data_best 同 CLI）
SCORE_PRED_HEAD_TAGS=(
  m1_freeze_st_encoder
  m2_freeze_llm_lora
  m3_no_rl
  m5_no_context_aug
)

_run_m4_suite() {
  _ensure_dirs
  # Full 模型上的连续头结构消融（与 backbone 五条独立）
  _head_one "full_small_data_best" --small_data_best
  _head_one "no_fft" --small_data_best --no_fft
  _head_one "no_skip" --small_data_best --no_skip
  _head_one "no_joint" --small_data_best --no_joint_context
}

_run_m4_parallel() {
  _ensure_dirs
  _parse_abl_gpu_list
  local ng="${#_ABL_GPU_ARR[@]}"
  local nj="${#M4_PARALLEL_NAMES[@]}"
  local i=0 j phys name fail=0
  echo "🚀 m4_parallel：共 ${nj} 路，GPU 列表（波次轮转）：${_ABL_GPU_ARR[*]}"
  while [[ "$i" -lt "$nj" ]]; do
    local pids=()
    for ((j = 0; j < ng && i < nj; j++, i++)); do
      phys="${_ABL_GPU_ARR[$j]}"
      name="${M4_PARALLEL_NAMES[$i]}"
      case "$name" in
        full_small_data_best)
          _head_one_on_gpu "$phys" "$name" --small_data_best &
          ;;
        no_fft)
          _head_one_on_gpu "$phys" "$name" --small_data_best --no_fft &
          ;;
        no_skip)
          _head_one_on_gpu "$phys" "$name" --small_data_best --no_skip &
          ;;
        no_joint)
          _head_one_on_gpu "$phys" "$name" --small_data_best --no_joint_context &
          ;;
        *)
          _die "未知 M4 变体: $name"
          ;;
      esac
      pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
  done
  [[ "$fail" -eq 0 ]] || _die "m4_parallel 中至少一路失败"
  echo "✅ m4_parallel 完成。m4_heads: ${ABL_PREFIX}/m4_heads/"
}

# 在给定骨干 best_mae 上训「与 Ours(M4 full_small_data_best)相同」的连续头并 eval → ablation_<tag>_pred_results.json
_score_pred_head_one() {
  local tag="$1"
  local base="${ABL_PREFIX}/${tag}/best_mae"
  local out="${ABL_PREFIX}/score_pred_heads/${tag}/full_small_data_best"
  [[ -d "$base" ]] || _die "score_heads 需要骨干 checkpoint: $base（请先完成 train ${tag}）"
  mkdir -p "$out"
  _m4_pred_extra_args
  echo ""
  echo "========== [score_heads train_pred_head] ${tag}  base=${base} → ${out} =========="
  ${HEAD_LAUNCH} "${_M4_PRED_EXTRA[@]}" \
    --checkpoint_dir "$base" \
    --save_dir "$out" \
    --small_data_best
  echo ""
  echo "========== [score_heads eval] ablation_${tag}_pred =========="
  python eval_SCoRe.py \
    --checkpoint_path "$out" \
    --model_name "ablation_${tag}_pred" \
    --results_dir "$RESULTS_JSON_DIR" \
    --eval_test_node_fraction 1.0 \
    --continuous_only \
    --skip_plots
}

_score_pred_head_one_on_gpu() {
  local phys_gpu="$1" tag="$2"
  local base="${ABL_PREFIX}/${tag}/best_mae"
  local out="${ABL_PREFIX}/score_pred_heads/${tag}/full_small_data_best"
  [[ -d "$base" ]] || _die "score_heads 需要骨干 checkpoint: $base（请先完成 train ${tag}）"
  (
    cd "$ROOT" || exit 1
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    export CUDA_VISIBLE_DEVICES="${phys_gpu}"
    export TRAFFIC_RL_CUDA_DEVICE="${TRAFFIC_RL_CUDA_DEVICE:-0}"
    mkdir -p "$out"
    if [[ -n "${TRAFFIC_RL_M4_PRED_HEAD_EXTRA_ARGS:-}" ]]; then
      read -ra _M4_PRED_EXTRA <<< "${TRAFFIC_RL_M4_PRED_HEAD_EXTRA_ARGS}"
    else
      _M4_PRED_EXTRA=()
    fi
    echo ""
    echo "========== [score_heads GPU ${phys_gpu}] ${tag} base=${base} → ${out} =========="
    ${HEAD_LAUNCH} "${_M4_PRED_EXTRA[@]}" \
      --checkpoint_dir "$base" \
      --save_dir "$out" \
      --small_data_best
    echo ""
    echo "========== [score_heads eval GPU ${phys_gpu}] ablation_${tag}_pred =========="
    python eval_SCoRe.py \
      --checkpoint_path "$out" \
      --model_name "ablation_${tag}_pred" \
      --results_dir "$RESULTS_JSON_DIR" \
      --eval_test_node_fraction 1.0 \
      --continuous_only \
      --skip_plots
  ) &
}

_run_score_heads_suite() {
  _ensure_dirs
  echo "🔗 score_heads：在 M1/M2/M3/M5 的 best_mae 上各训一条 full_small_data_best 连续头（与 M4/Ours 协议一致）。"
  local t fail=0
  for t in "${SCORE_PRED_HEAD_TAGS[@]}"; do
    _score_pred_head_one "$t" || fail=1
  done
  [[ "$fail" -eq 0 ]] || _die "score_heads 中至少一路失败"
  echo "✅ score_heads 完成。输出目录: ${ABL_PREFIX}/score_pred_heads/"
}

_run_score_heads_parallel() {
  _ensure_dirs
  _parse_abl_gpu_list
  local ng="${#_ABL_GPU_ARR[@]}"
  local nj="${#SCORE_PRED_HEAD_TAGS[@]}"
  local i=0 j phys tag fail=0
  echo "🚀 score_heads_parallel：${nj} 路，GPU 轮转：${_ABL_GPU_ARR[*]}"
  while [[ "$i" -lt "$nj" ]]; do
    local pids=()
    for ((j = 0; j < ng && i < nj; j++, i++)); do
      phys="${_ABL_GPU_ARR[$j]}"
      tag="${SCORE_PRED_HEAD_TAGS[$i]}"
      _score_pred_head_one_on_gpu "$phys" "$tag"
      pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
  done
  [[ "$fail" -eq 0 ]] || _die "score_heads_parallel 至少一路失败"
  echo "✅ score_heads_parallel 完成。"
}

_run_collect() {
  python scripts/collect_ablation_results.py \
    --scan_dir "${RESULTS_JSON_DIR}" \
    --out_csv "${RESULTS_JSON_DIR}/ablation_summary.csv"
}

cmd="${1:-all}"
_export_common
_ensure_dirs

case "$cmd" in
  train)
    echo "📂 消融输出根目录: ${ABL_OUTPUT_ROOT}"
    echo "📂 Checkpoints:     ${ABL_PREFIX}"
    echo "📂 评测 JSON/CSV:   ${RESULTS_JSON_DIR}"
    # M2：m2_freeze_llm_lora（freeze LoRA）；M4 不在此阶段，见 m4 / heads
    _run_score_train "${FULL_TAG}" TRAFFIC_RL_ENABLE_RL=1
    _run_score_train "m1_freeze_st_encoder" TRAFFIC_RL_ENABLE_RL=1 TRAFFIC_RL_ABLATE_FREEZE_ST_ENCODER=1
    _run_score_train "m2_freeze_llm_lora" TRAFFIC_RL_ENABLE_RL=1 TRAFFIC_RL_ABLATE_FREEZE_LLM_LORA=1
    _run_score_train "m3_no_rl" TRAFFIC_RL_ENABLE_RL=0
    _run_score_train "m5_no_context_aug" TRAFFIC_RL_ENABLE_RL=1 TRAFFIC_RL_ABLATE_NO_CONTEXT_AUG=1
    echo "✅ train 完成。checkpoint: ${ABL_PREFIX}"
    ;;
  train_parallel)
    echo "📂 消融输出根目录: ${ABL_OUTPUT_ROOT}"
    echo "📂 Checkpoints:     ${ABL_PREFIX}"
    echo "📂 评测 JSON/CSV:   ${RESULTS_JSON_DIR}"
    _run_train_parallel
    ;;
  eval)
    for t in "${FULL_TAG}" m1_freeze_st_encoder m2_freeze_llm_lora m3_no_rl m5_no_context_aug; do
      _eval_one "$t"
    done
    ;;
  eval_parallel)
    _run_eval_parallel
    ;;
  heads|m4)
    _run_m4_suite
    ;;
  m4_parallel|heads_parallel)
    _run_m4_parallel
    ;;
  collect)
    _run_collect
    ;;
  score_heads)
    _run_score_heads_suite
    ;;
  score_heads_parallel)
    _run_score_heads_parallel
    ;;
  all)
    "$0" train
    "$0" eval
    "$0" m4
    "$0" score_heads
    "$0" collect
    ;;
  all_parallel)
    "$0" train_parallel
    "$0" eval_parallel
    "$0" m4_parallel
    "$0" score_heads_parallel
    "$0" collect
    ;;
  *)
    echo "用法: $0 train|train_parallel|eval|eval_parallel|m4|m4_parallel|heads|heads_parallel|score_heads|score_heads_parallel|collect|all|all_parallel" >&2
    echo "说明: score_heads* 需在对应骨干 best_mae 就绪后运行（可先 train 再 score_heads_parallel + collect）。M4=m4_parallel。" >&2
    exit 1
    ;;
esac
