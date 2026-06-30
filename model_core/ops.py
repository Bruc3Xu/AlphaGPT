import torch

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)


def _causal_windows(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x.unsqueeze(-1)
    pad = torch.full((x.shape[0], window - 1), float("nan"), device=x.device, dtype=x.dtype)
    return torch.cat([pad, x], dim=1).unfold(1, window, 1)


def _rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    windows = _causal_windows(x, window)
    valid = torch.isfinite(windows)
    values = torch.where(valid, windows, torch.zeros_like(windows))
    count = valid.sum(dim=-1).clamp_min(1)
    return values.sum(dim=-1) / count


def _rolling_std(x: torch.Tensor, window: int) -> torch.Tensor:
    windows = _causal_windows(x, window)
    valid = torch.isfinite(windows)
    mean = _rolling_mean(x, window)
    centered = torch.where(valid, windows - mean.unsqueeze(-1), torch.zeros_like(windows))
    count = valid.sum(dim=-1).clamp_min(1)
    return torch.sqrt((centered ** 2).sum(dim=-1) / count + 1e-6)


def _rolling_max(x: torch.Tensor, window: int) -> torch.Tensor:
    windows = _causal_windows(x, window)
    values = torch.nan_to_num(windows, nan=float("-inf"))
    return torch.nan_to_num(values.max(dim=-1)[0], nan=0.0, posinf=0.0, neginf=0.0)


def _rolling_min(x: torch.Tensor, window: int) -> torch.Tensor:
    windows = _causal_windows(x, window)
    values = torch.nan_to_num(windows, nan=float("inf"))
    return torch.nan_to_num(values.min(dim=-1)[0], nan=0.0, posinf=0.0, neginf=0.0)


def _ts_delta(x: torch.Tensor, d: int) -> torch.Tensor:
    return x - _ts_delay(x, d)


def _ts_rank(x: torch.Tensor, window: int) -> torch.Tensor:
    windows = _causal_windows(x, window)
    valid = torch.isfinite(windows)
    latest = x.unsqueeze(-1)
    count = valid.sum(dim=-1).clamp_min(1)
    rank = ((windows <= latest) & valid).float().sum(dim=-1) / count
    return torch.nan_to_num(rank * 2.0 - 1.0, nan=0.0, posinf=0.0, neginf=0.0)


def _cs_rank(x: torch.Tensor) -> torch.Tensor:
    clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clean.shape[0] <= 1:
        return torch.zeros_like(clean)
    ranks = clean.argsort(dim=0).argsort(dim=0).to(clean.dtype)
    return ranks / (clean.shape[0] - 1) * 2.0 - 1.0


def _cs_zscore(x: torch.Tensor) -> torch.Tensor:
    clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = clean.mean(dim=0, keepdim=True)
    std = clean.std(dim=0, keepdim=True, unbiased=False) + 1e-6
    return torch.clamp((clean - mean) / std, -5.0, 5.0)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y


def _rolling_zscore(x: torch.Tensor, window: int = 120) -> torch.Tensor:
    if window <= 1:
        return torch.zeros_like(x)
    pad = torch.full((x.shape[0], window - 1), float("nan"), device=x.device, dtype=x.dtype)
    windows = torch.cat([pad, x], dim=1).unfold(1, window, 1)
    valid = torch.isfinite(windows)
    values = torch.where(valid, windows, torch.zeros_like(windows))
    count = valid.sum(dim=-1).clamp_min(1)
    mean = values.sum(dim=-1) / count
    centered = torch.where(valid, windows - mean.unsqueeze(-1), torch.zeros_like(windows))
    var = (centered ** 2).sum(dim=-1) / count
    z = (x - mean) / torch.sqrt(var + 1e-6)
    return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _op_jump(x: torch.Tensor) -> torch.Tensor:
    z = _rolling_zscore(x)
    return torch.relu(z - 3.0)


def _safe_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    eps = torch.full_like(y, 1e-6)
    signed_eps = torch.where(y < 0, -eps, eps)
    denom = torch.where(torch.abs(y) < 1e-6, signed_eps, y)
    return x / denom


def _clip(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -5.0, 5.0)


def _sqrt_abs(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(x) + 1e-9)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', _safe_div, 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('DELAY3', lambda x: _ts_delay(x, 3), 1),
    ('DELTA1', lambda x: _ts_delta(x, 1), 1),
    ('DELTA5', lambda x: _ts_delta(x, 5), 1),
    ('MA5', lambda x: _rolling_mean(x, 5), 1),
    ('MA20', lambda x: _rolling_mean(x, 20), 1),
    ('STD20', lambda x: _rolling_std(x, 20), 1),
    ('ZSCORE20', lambda x: _rolling_zscore(x, 20), 1),
    ('TS_RANK20', lambda x: _ts_rank(x, 20), 1),
    ('TS_MAX20', lambda x: _rolling_max(x, 20), 1),
    ('TS_MIN20', lambda x: _rolling_min(x, 20), 1),
    ('CS_RANK', _cs_rank, 1),
    ('CS_ZSCORE', _cs_zscore, 1),
    ('CLIP', _clip, 1),
    ('TANH', torch.tanh, 1),
    ('SQRT_ABS', _sqrt_abs, 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1)
]
