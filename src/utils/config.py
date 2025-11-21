"""
配置文件 - src/utils/config.py
集中管理所有超参数和路径配置
[!! 已统一去噪接口 !!]
"""

import os
from pathlib import Path
import yaml
import torch

HPO_SEARCH_SPACE = {
    'WPD_WAVELET': ['db4', 'sym5', 'coif3'],  # 尝试不同的小波基
    'WPD_LEVEL': [3, 4, 5],                   # 尝试不同的分解层数
    'WPD_THRESHOLD_SCALE': [0.8, 1.0, 1.5, 2.0] # 尝试不同的阈值缩放
}

class Config:
    """全局配置类"""

    # ==================== 路径配置 ====================
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    FEATURES_DIR = DATA_DIR / "features"

    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    LOGS_DIR = PROJECT_ROOT / "logs"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # 创建必要的目录
    for dir_path in [PROCESSED_DATA_DIR, FEATURES_DIR, CHECKPOINTS_DIR,
                     LOGS_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # ==================== 数据配置 ====================
    DATA_FILE = "DAS_data.csv"
    RANDOM_SEED = 42
    INPUT_FEATURES = 101  # 空间测量点的数量

    # 数据集划分比例
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    # 类别映射
    LABEL_MAPPING = {
        'Y': 1,  # 异常事件
        'N': 0,  # 正常
        'NA': -1  # 未标注(处理时删除)
    }

    # ==================== 去噪与预处理策略 (统一接口) ====================
    # [!! 修改 !!] 核心控制开关：选择去噪方法
    # 选项: 'wpd' (小波包), 'spectral' (频谱减法), 'none' (不去噪)
    DENOISE_METHOD = 'wpd'

    # --- 方法 1: WPD 去噪参数 (仅当 DENOISE_METHOD='wpd' 时生效) ---
    WPD_WAVELET = 'db4'        # 小波基 ('db4', 'sym5', 'coif3')
    # 跑 CNN-2D 时建议用: coif3
    WPD_LEVEL = 4              # 分解层数 (通常 4 层)
    WPD_THRESHOLD_SCALE = 1.5  # 阈值力度 (越大去噪越强, 建议 0.5-2.0)
    WPD_THRESHOLD_METHOD = 'soft'  # 'soft' (保留细节) or 'hard' (强去噪)

    # --- 方法 2: 频谱减法参数 (仅当 DENOISE_METHOD='spectral' 时生效) ---
    NOISE_PERCENTILE = 10      # 噪声能量百分位数

    # --- 通用预处理参数 ---
    NORMALIZATION_METHOD = 'zscore'  # 'zscore', 'minmax'
    HIGH_FREQ_THRESHOLD = 100  # Hz
    HIGH_FREQ_PERCENTILE = 0.8

    # 缺失值处理
    INTERPOLATION_METHOD = 'linear'
    MAX_MISSING_RATIO = 0.3

    # ==================== 特征提取配置 ====================
    # 注意：WPD特征提取是独立的，即使预处理选了'none'，这里也可以开启
    ENABLE_WPD_FEATURES = True  # 是否提取WPD能量/熵特征
    WPD_FEATURE_LEVEL = 3       # 特征提取用的层数 (通常比去噪用的浅)
    WPD_FEATURE_WAVELET = 'db4'

    # 时域特征
    TIME_FEATURES = [
        'energy', 'max_amplitude', 'std', 'zero_crossing_rate',
        'kurtosis', 'skewness', 'rms'
    ]

    # 频域特征
    NUM_FREQ_BANDS = 8
    SAMPLE_RATE = 5000  # Hz

    # 空间特征
    SPATIAL_FEATURES = [
        'spatial_gradient', 'spatial_variance',
        'max_position', 'energy_centroid'
    ]

    # ==================== 时间窗口配置 ====================
    WINDOW_SIZE = 20
    WINDOW_STRIDE = 1

    # ==================== 经典ML配置 ====================
    # SVM
    SVM_CONFIG = {
        'C': 10,
        'break_ties': False,
        'cache_size': 200,
        'class_weight': 'balanced',
        'coef0': 0.0,
        'decision_function_shape': 'ovr',
        'degree': 3,
        'gamma': 'scale',
        'kernel': 'linear',
        'max_iter': -1,
        'probability': True,
        'random_state': 42,
        'shrinking': True,
        'tol': 0.001,
        'verbose': False
    }

    # Random Forest
    RF_CONFIG = {
        'n_estimators': 478,
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    # XGBoost
    XGB_CONFIG = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,
        'learning_rate': 0.15099,
        'n_estimators': 184,
        'subsample': 0.8036,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    # ==================== 深度学习配置 ====================
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    WEIGHT_DECAY = 1e-5

    # 早停
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_DELTA = 1e-4

    DEFAULT_PENALTY_WEIGHT_1 = 15.0

    # 学习率调度
    LR_SCHEDULER = 'ReduceLROnPlateau'
    LR_PATIENCE = 5
    LR_FACTOR = 0.5

    # LSTM-CNN配置
    LSTM_CNN_CONFIG = {
        'learning_rate': 0.000025753,
        'cnn_filters': [256, 128, 64],
        'cnn_kernel_sizes': [5, 3, 3],
        'lstm_hidden_size': 128,
        'lstm_num_layers': 1,
        'fc_hidden_sizes': [256, 64],
        'dropout': 0.52988,
        'num_classes': 2
    }

    # LSTM AE
    LSTM_AE_CONFIG = {
        'learning_rate': 0.001,
        'encoder_hidden_sizes': [64, 32],
        'decoder_hidden_sizes': [32, 64],
        'latent_dim': 16,
        'dropout': 0.3
    }

    # 1D-CNN配置
    CNN_1D_CONFIG = {
        'learning_rate': 0.001,
        'conv_channels': [128, 128, 64],
        'kernel_sizes': [5, 3, 3],
        'pool_sizes': [2, 2, 2],
        'fc_hidden_sizes': [256, 64],
        'dropout': 0.5,
        'num_classes': 2
    }

    # 2D-CNN配置
    CNN_2D_CONFIG = {
        'learning_rate': 0.00001,
        'conv_channels': [32, 64, 64],
        'fc_hidden': [128],
        'dropout': 0.5,
        'num_classes': 2
    }

    # ==================== 评估配置 ====================
    CV_FOLDS = 5
    CV_STRATEGY = 'stratified'

    METRICS = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'confusion_matrix'
    ]

    # DAS特定指标
    TDR_THRESHOLD = 0.8
    FAR_THRESHOLD = 0.1

    # ==================== 可视化配置 ====================
    FIG_SIZE = (12, 6)
    DPI = 300
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'

    # ==================== 日志配置 ====================
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    @classmethod
    def save_config(cls, filepath):
        """保存配置到YAML文件"""
        config_dict = {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

        # 转换Path对象为字符串
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(filepath, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        print(f"配置已保存到: {filepath}")

    @classmethod
    def load_config(cls, filepath):
        """从YAML文件加载配置"""
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        for key, value in config_dict.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

        print(f"配置已从文件加载: {filepath}")

    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 60)
        print("当前配置:")
        print("=" * 60)

        sections = [
            ('路径配置', ['PROJECT_ROOT', 'DATA_DIR', 'CHECKPOINTS_DIR']),
            ('数据配置', ['RANDOM_SEED', 'TRAIN_RATIO', 'VAL_RATIO', 'TEST_RATIO']),
            ('去噪与预处理', ['DENOISE_METHOD', 'WPD_WAVELET', 'NORMALIZATION_METHOD']),
            ('深度学习配置', ['DEVICE', 'BATCH_SIZE', 'NUM_EPOCHS'])
        ]

        for section_name, keys in sections:
            print(f"\n{section_name}:")
            print("-" * 60)
            for key in keys:
                if hasattr(cls, key):
                    value = getattr(cls, key)
                    print(f"  {key}: {value}")

        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()