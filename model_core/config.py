import torch
import os
from .vocab import FORMULA_VOCAB

class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"
    BATCH_SIZE = 8192
    TRAIN_STEPS = 1000
    LEARNING_RATE = 3e-4
    MAX_FORMULA_LEN = 12
    ROLLING_NORM_WINDOW = 120
    WALK_FORWARD_FOLDS = 3
    WALK_FORWARD_TRAIN_RATIO = 0.5
    VALIDATION_REWARD_WEIGHT = 0.5
    VALUE_LOSS_COEF = 0.25
    ENTROPY_COEF = 0.01
    GRAD_CLIP_NORM = 1.0
    TRADE_SIZE_USD = 1000.0
    MIN_LIQUIDITY = 5000.0 # 低于此流动性视为归零/无法交易
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    INPUT_DIM = FORMULA_VOCAB.feature_count
