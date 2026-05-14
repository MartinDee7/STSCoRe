# -*- coding: utf-8 -*-
"""
连续头回归损失：
- regression_loss_mse_only：仅 MSE（或可选 SmoothL1），与 Time-LLM 等一致，简单稳定。
- regression_loss_urban_style：复合损失（趋势/峰值等），暂保留供需要时调用。
"""
import torch
import torch.nn as nn


def regression_loss_mse_only(pred, target, use_smoothl1=False, smoothl1_beta=1.0):
    """
    pred, target: (B, pred_len)，与 batch['target'] 同空间。
    仅 MSE 或 SmoothL1（Huber），无 slope/curve/peak 等项。
    返回 (loss, logs)；logs 含 mse、mae 便于打印。
    """
    if use_smoothl1:
        loss = nn.functional.smooth_l1_loss(pred, target, beta=smoothl1_beta)
        logs = {"mse": nn.functional.mse_loss(pred, target).detach()}
    else:
        loss = nn.functional.mse_loss(pred, target)
        logs = {"mse": loss.detach()}
    logs["mae"] = (pred - target).abs().mean().detach()
    logs["total"] = loss.detach()
    return loss, logs


def pearson_corr_per_sample(pred, target, eps=1e-8):
    """pred, target: (B, T)；返回标量 mean corr"""
    B, T = pred.shape
    corrs = []
    for b in range(B):
        p, g = pred[b], target[b]
        denom = p.std() * g.std() + eps
        if denom < eps * 10:
            continue
        p = p - p.mean()
        g = g - g.mean()
        c = (p * g).mean() / denom
        c = torch.clamp(c, -1.0, 1.0)
        corrs.append(c)
    if not corrs:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return torch.stack(corrs).mean()


def regression_loss_urban_style(
    pred,
    target,
    # 主损失：SmoothL1 对尖峰更稳，MAE 直接压 reported MAE
    use_smoothl1=True,
    smoothl1_beta=0.05,
    lambda_mae=0.25,
    # 趋势：一阶 + 二阶差分
    lambda_slope=0.1,
    lambda_curve=0.08,
    # 形状：相关
    lambda_corr=0.05,
    # 峰值/幅值：max/min 对齐 + 高值加权 L1
    lambda_range=0.12,
    lambda_peak_weight=0.15,
    peak_quantile=0.75,
    # 显式 MSE：直接拉低 RMSE
    lambda_mse=0.0,
    # 逐步涨跌方向一致：sign(Δpred)==sign(Δgt)，趋势更准
    lambda_direction=0.0,
    # 后期步加权：缓解「前几步还行、后面拉不下去/贴均值」
    lambda_horizon=0.0,
    horizon_ramp=1.5,
):
    """
    pred, target: (B, pred_len)，与 batch['target'] 同空间（如归一化）。
    返回 (loss, logs_dict)。
    """
    logs = {}
    # --- 主项：SmoothL1 或 MSE + 显式 MAE ---
    if use_smoothl1:
        main = nn.functional.smooth_l1_loss(pred, target, beta=smoothl1_beta)
        logs["smoothl1"] = main.detach()
    else:
        main = nn.functional.mse_loss(pred, target)
        logs["mse"] = main.detach()
    mae = (pred - target).abs().mean()
    logs["mae"] = mae.detach()
    loss = main + lambda_mae * mae

    # --- Horizon 加权 L1：越往后的预测步权重越大（默认线性 ramp）---
    if lambda_horizon > 0 and pred.shape[1] > 0:
        T = pred.shape[1]
        # w_t = 1 + ramp * t/(T-1)，后期步更贵
        t = torch.arange(T, device=pred.device, dtype=pred.dtype)
        w = 1.0 + horizon_ramp * t / max(T - 1, 1)
        w = w.view(1, -1)
        horiz_loss = ((pred - target).abs() * w).sum() / w.sum()
        loss = loss + lambda_horizon * horiz_loss
        logs["horizon_w"] = horiz_loss.detach()

    # --- 显式 MSE（RMSE 与 MSE 单调相关，加大权重可压 RMSE）---
    if lambda_mse > 0:
        mse = nn.functional.mse_loss(pred, target)
        loss = loss + lambda_mse * mse
        logs["mse_aux"] = mse.detach()

    # --- 一阶差分（趋势方向）---
    if pred.shape[1] > 1:
        d_pred = pred[:, 1:] - pred[:, :-1]
        d_tgt = target[:, 1:] - target[:, :-1]
        slope_loss = (d_pred - d_tgt).abs().mean()
        loss = loss + lambda_slope * slope_loss
        logs["slope"] = slope_loss.detach()
        # 方向一致：每步涨跌符号相同则不计罚，否则罚 |d_pred|+|d_tgt| 量级，避免只追求斜率幅值
        if lambda_direction > 0:
            sign_p = torch.sign(d_pred)
            sign_g = torch.sign(d_tgt)
            # 忽略接近 0 的步（无方向）
            mask = (sign_g.abs() > 1e-6).float()
            mismatch = (sign_p != sign_g).float() * mask
            dir_loss = (mismatch * (d_pred.abs() + d_tgt.abs())).sum() / (mask.sum() + 1e-8)
            loss = loss + lambda_direction * dir_loss
            logs["direction"] = dir_loss.detach()
    else:
        slope_loss = torch.tensor(0.0, device=pred.device)

    # --- 二阶差分（曲率/峰谷形状，抑制「直线」）---
    if pred.shape[1] > 2:
        d2_pred = d_pred[:, 1:] - d_pred[:, :-1]
        d2_tgt = d_tgt[:, 1:] - d_tgt[:, :-1]
        curve_loss = (d2_pred - d2_tgt).abs().mean()
        loss = loss + lambda_curve * curve_loss
        logs["curve"] = curve_loss.detach()

    # --- 相关 ---
    corr = pearson_corr_per_sample(pred, target)
    corr_loss = 1.0 - corr
    loss = loss + lambda_corr * corr_loss
    logs["corr"] = corr.detach()

    # --- 幅值对齐：整条预测曲线的 max/min 与真值接近（峰值高度、谷底）---
    if lambda_range > 0 and pred.shape[1] >= 1:
        max_p, max_t = pred.max(dim=1)[0], target.max(dim=1)[0]
        min_p, min_t = pred.min(dim=1)[0], target.min(dim=1)[0]
        range_loss = (max_p - max_t).abs().mean() + (min_p - min_t).abs().mean()
        loss = loss + lambda_range * range_loss
        logs["range"] = range_loss.detach()

    # --- 峰值区域加权：真值高分位以上误差权重大，避免预测整体压扁 ---
    if lambda_peak_weight > 0 and pred.shape[1] > 1:
        t = target
        # 每行分位阈值（偏高段加重）
        try:
            thresh = torch.quantile(t.float(), peak_quantile, dim=1, keepdim=True)
        except Exception:
            k = max(1, int(t.shape[1] * peak_quantile))
            thresh, _ = t.kthvalue(k, dim=1, keepdim=True)
        high_mask = (t >= thresh).float()
        # 高值处 under-prediction 更重罚（峰值抬不起来）
        under = nn.functional.relu(target - pred)
        peak_loss = (high_mask * under).mean() + (high_mask * (pred - target).abs()).mean()
        loss = loss + lambda_peak_weight * peak_loss
        logs["peak"] = peak_loss.detach()

    logs["total"] = loss.detach()
    return loss, logs
