from dataclasses import dataclass

from .ops import OPS_CONFIG


FEATURE_NAMES = (
    "RET",
    "RET_5",
    "RET_15",
    "LIQ_SCORE",
    "LIQ_CHG",
    "FDV_CHG",
    "PRESSURE",
    "FOMO",
    "DEV_20",
    "DEV_60",
    "LOG_VOL",
    "VOL_SHOCK",
    "VOL_TREND",
    "VOL_CLUSTER",
    "MOM_REV",
    "REL_STRENGTH",
    "HL_RANGE",
    "CLOSE_POS",
    "LIQ_USAGE",
    "DRAWUP_20",
    "DRAWDOWN_20",
)


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
