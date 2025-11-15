"""
配置文件 - src/utils/config.py
集中管理所有超参数和路径配置
"""

import os
from pathlib import Path
import yaml
import torch


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
    INPUT_FEATURES = 101  # 空间测量点的数量 (来自 X_train.shape[2])

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

    # ==================== 预处理配置 ====================
    # 归一化方法: 'zscore', 'minmax', 'high_freq_energy'
    NORMALIZATION_METHOD = 'high_freq_energy'
    HIGH_FREQ_THRESHOLD = 100  # Hz, 高频阈值
    HIGH_FREQ_PERCENTILE = 0.8  # 使用后20%作为高频

    # 缺失值处理
    INTERPOLATION_METHOD = 'linear'  # 'linear', 'cubic', 'nearest'
    MAX_MISSING_RATIO = 0.3  # 超过30%缺失的样本将被删除

    # ==================== 特征工程配置 ====================
    # 时域特征
    TIME_FEATURES = [
        'energy', 'max_amplitude', 'std', 'zero_crossing_rate',
        'kurtosis', 'skewness', 'rms'
    ]

    # 频域特征
    NUM_FREQ_BANDS = 8  # 频带数量
    SAMPLE_RATE = 5000  # Hz, 采样率(根据实际情况调整)

    # 空间特征
    SPATIAL_FEATURES = [
        'spatial_gradient', 'spatial_variance',
        'max_position', 'energy_centroid'
    ]

    # ==================== 时间窗口配置 ====================
    WINDOW_SIZE = 5  # 时间窗口大小(帧数)
    WINDOW_STRIDE = 1  # 窗口滑动步长

    # ==================== 经典ML配置 ====================
    # SVM
    SVM_CONFIG = {
        'kernel': 'linear',
        'C': 1,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': RANDOM_SEED
    }

    # Random Forest
    RF_CONFIG = {
        'n_estimators': 300,
        'max_depth': 30,
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
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'n_jobs': -1
    }

    # ==================== 深度学习配置 ====================
    # 通用训练配置
    DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # 早停
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_DELTA = 1e-4

    # 学习率调度
    LR_SCHEDULER = 'ReduceLROnPlateau'
    LR_PATIENCE = 5
    LR_FACTOR = 0.5

    # LSTM-CNN配置
    LSTM_CNN_CONFIG = {
        'cnn_filters': [256, 128, 64],
        'cnn_kernel_sizes': [5, 3, 3],
        'lstm_hidden_size': 64,
        'lstm_num_layers': 1,
        'fc_hidden_sizes': [256, 64],
        'dropout': 0.6,
        'num_classes': 2
    }

    # LSTM Autoencoder配置
    LSTM_AE_CONFIG = {
        'encoder_hidden_sizes': [64, 32],
        'decoder_hidden_sizes': [32, 64],
        'latent_dim': 16,
        'dropout': 0.3
    }

    # 1D-CNN配置
    CNN_1D_CONFIG = {
        'conv_channels': [128, 128, 64],
        'kernel_sizes': [5, 3, 3],
        'pool_sizes': [2, 2, 2],
        'fc_hidden_sizes': [256, 64],
        'dropout': 0.5,
        'num_classes': 2
    }

    # ==================== 评估配置 ====================
    # 交叉验证
    CV_FOLDS = 5
    CV_STRATEGY = 'stratified'  # 'stratified', 'kfold', 'timeseries'

    # 评估指标
    METRICS = [
        'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'confusion_matrix'
    ]

    # DAS特定指标
    TDR_THRESHOLD = 0.8  # True Detection Rate目标
    FAR_THRESHOLD = 0.1  # False Alarm Rate目标

    # ==================== 可视化配置 ====================
    FIG_SIZE = (12, 6)
    DPI = 300
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'

    # ==================== 日志配置 ====================
    LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
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
            ('预处理配置', ['NORMALIZATION_METHOD', 'INTERPOLATION_METHOD']),
            ('深度学习配置', ['DEVICE', 'BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE'])
        ]

        for section_name, keys in sections:
            print(f"\n{section_name}:")
            print("-" * 60)
            for key in keys:
                if hasattr(cls, key):
                    value = getattr(cls, key)
                    print(f"  {key}: {value}")

        print("=" * 60)


# 使用示例
if __name__ == "__main__":
    # 打印配置
    Config.print_config()

    # 保存配置
    # Config.save_config(Config.PROJECT_ROOT / "config.yaml")

    # 加载配置
    # Config.load_config(Config.PROJECT_ROOT / "config.yaml")