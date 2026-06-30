import torch
import torch.nn as nn

from .config import ModelConfig
from .vocab import FEATURE_NAMES


def _ts_delay(x, d=1):
    if d <= 0:
        return x
    pad = torch.zeros((x.shape[0], d), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-d]], dim=1)


def _causal_windows(x, window):
    if window <= 1:
        return x.unsqueeze(-1)
    pad = torch.full(
        (x.shape[0], window - 1),
        float("nan"),
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([pad, x], dim=1).unfold(1, window, 1)


def rolling_mean(x, window):
    clean = torch.where(torch.isfinite(x), x, torch.nan)
    windows = _causal_windows(clean, window)
    mean = torch.nanmean(windows, dim=-1)
    return torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)


def rolling_sum(x, window):
    clean = torch.where(torch.isfinite(x), x, torch.nan)
    windows = _causal_windows(clean, window)
    valid = torch.isfinite(windows)
    values = torch.where(valid, windows, torch.zeros_like(windows))
    return values.sum(dim=-1)


def rolling_max(x, window):
    clean = torch.where(torch.isfinite(x), x, torch.nan)
    windows = _causal_windows(clean, window)
    values = torch.nan_to_num(windows, nan=float("-inf"))
    out = values.max(dim=-1)[0]
    return torch.nan_to_num(out, nan=0.0, neginf=0.0, posinf=0.0)


def rolling_min(x, window):
    clean = torch.where(torch.isfinite(x), x, torch.nan)
    windows = _causal_windows(clean, window)
    values = torch.nan_to_num(windows, nan=float("inf"))
    out = values.min(dim=-1)[0]
    return torch.nan_to_num(out, nan=0.0, neginf=0.0, posinf=0.0)


def rolling_robust_norm(x, window=None, clip=5.0):
    window = window or ModelConfig.ROLLING_NORM_WINDOW
    clean = torch.where(torch.isfinite(x), x, torch.nan)
    windows = _causal_windows(clean, window)
    median = torch.nanmedian(windows, dim=-1)[0]
    abs_dev = torch.abs(windows - median.unsqueeze(-1))
    mad = torch.nanmedian(abs_dev, dim=-1)[0] + 1e-6
    norm = (clean - median) / mad
    norm = torch.clamp(norm, -clip, clip)
    return torch.nan_to_num(norm, nan=0.0, posinf=clip, neginf=-clip)


def safe_log_return(close):
    prev = _ts_delay(close, 1)
    ret = torch.log(torch.clamp(close, min=1e-9) / torch.clamp(prev, min=1e-9))
    ret[:, 0] = 0.0
    return torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MemeIndicators:
    @staticmethod
    def liquidity_health(liquidity, fdv):
        ratio = liquidity / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = _ts_delay(volume, 1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        vol_chg[:, 0] = 0.0
        acc = vol_chg - _ts_delay(vol_chg, 1)
        acc[:, 0] = 0.0
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        ma = rolling_mean(close, window)
        dev = (close - ma) / (ma + 1e-9)
        return torch.nan_to_num(dev, nan=0.0, posinf=5.0, neginf=-5.0)

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = safe_log_return(close)
        ret_sq = ret ** 2
        vol_ma = rolling_mean(ret_sq, window)
        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = safe_log_return(close)
        mom = rolling_sum(ret, window)
        
        # Detect reversals
        mom_prev = _ts_delay(mom, 1)
        reversal = (mom * mom_prev < 0).float()
        reversal[:, 0] = 0.0
        
        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - _ts_delay(close, 1)
        ret[:, 0] = 0.0
        
        gains = torch.relu(ret)
        losses = torch.relu(-ret)
        
        avg_gain = rolling_mean(gains, window)
        avg_loss = rolling_mean(losses, window)
        
        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return (rsi - 50) / 50  # Normalize


class AdvancedFactorEngineer:
    """Compatibility wrapper for the expanded main feature set."""
    def __init__(self):
        self.rms_norm = RMSNormFactor(1)
    
    def robust_norm(self, t):
        """Causal rolling normalization using median absolute deviation."""
        return rolling_robust_norm(t)
    
    def compute_advanced_features(self, raw_dict):
        """Compute the same expanded feature space used by the main pipeline."""
        return FeatureEngineer.compute_features(raw_dict)


class FeatureEngineer:
    INPUT_DIM = len(FEATURE_NAMES)

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict['close']
        o = raw_dict['open']
        h = raw_dict['high']
        l = raw_dict['low']
        v = raw_dict['volume']
        liq = raw_dict['liquidity']
        fdv = raw_dict['fdv']
        
        ret = safe_log_return(c)
        ret_5 = rolling_sum(ret, 5)
        ret_15 = rolling_sum(ret, 15)
        liq_score = MemeIndicators.liquidity_health(liq, fdv)
        liq_prev = _ts_delay(liq, 1)
        fdv_prev = _ts_delay(fdv, 1)
        liq_chg = (liq - liq_prev) / (liq_prev + 1.0)
        fdv_chg = (fdv - fdv_prev) / (fdv_prev + 1.0)
        liq_chg[:, 0] = 0.0
        fdv_chg[:, 0] = 0.0

        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)
        fomo = MemeIndicators.fomo_acceleration(v)
        dev_20 = MemeIndicators.pump_deviation(c, 20)
        dev_60 = MemeIndicators.pump_deviation(c, 60)
        log_vol = torch.log1p(v)
        vol_ma20 = rolling_mean(v, 20)
        vol_shock = v / (vol_ma20 + 1.0) - 1.0

        vol_prev = _ts_delay(v, 1)
        vol_trend = (v - vol_prev) / (vol_prev + 1.0)
        vol_trend[:, 0] = 0.0

        vol_cluster = MemeIndicators.volatility_clustering(c)
        momentum_rev = MemeIndicators.momentum_reversal(c)
        rel_strength = MemeIndicators.relative_strength(c, h, l)
        hl_range = (h - l) / (c + 1e-9)
        close_pos = (c - l) / (h - l + 1e-9)
        liq_usage = v / (liq + 1.0)

        high_20 = rolling_max(h, 20)
        low_20 = rolling_min(l, 20)
        drawup_20 = (c - low_20) / (low_20 + 1e-9)
        drawdown_20 = (c - high_20) / (high_20 + 1e-9)

        features = torch.stack([
            rolling_robust_norm(ret),
            rolling_robust_norm(ret_5),
            rolling_robust_norm(ret_15),
            liq_score,
            rolling_robust_norm(liq_chg),
            rolling_robust_norm(fdv_chg),
            pressure,
            rolling_robust_norm(fomo),
            rolling_robust_norm(dev_20),
            rolling_robust_norm(dev_60),
            rolling_robust_norm(log_vol),
            rolling_robust_norm(vol_shock),
            rolling_robust_norm(vol_trend),
            rolling_robust_norm(vol_cluster),
            momentum_rev,
            rolling_robust_norm(rel_strength),
            rolling_robust_norm(hl_range),
            close_pos,
            rolling_robust_norm(liq_usage),
            rolling_robust_norm(drawup_20),
            rolling_robust_norm(drawdown_20),
        ], dim=1)
        
        return features
