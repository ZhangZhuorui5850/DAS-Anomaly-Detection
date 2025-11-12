"""
数据预处理模块 - src/data/preprocessing.py
包含: 数据加载、清洗、归一化、特征提取
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

warnings.filterwarnings('ignore')


class DASDataLoader:
    """DAS数据加载器"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.spatial_cols = None

    def load_data(self) -> pd.DataFrame:
        """加载原始数据"""
        print(f"正在加载数据: {self.data_path}")

        self.df = pd.read_csv(self.data_path)

        # 识别空间测量列(X开头的列)
        self.spatial_cols = [col for col in self.df.columns if col.startswith('X')]

        print(f"数据形状: {self.df.shape}")
        print(f"空间测量点数量: {len(self.spatial_cols)}")
        print(f"标签分布:\n{self.df['status'].value_counts()}")

        return self.df

    def get_basic_stats(self) -> Dict:
        """获取基本统计信息"""
        stats = {
            'total_samples': len(self.df),
            'num_spatial_points': len(self.spatial_cols),
            'missing_ratio': self.df[self.spatial_cols].isnull().sum().sum() /
                           (len(self.df) * len(self.spatial_cols)),
            'label_distribution': self.df['status'].value_counts().to_dict()
        }
        return stats


class DASPreprocessor:
    """DAS数据预处理器"""

    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.spatial_cols = None

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        print("\n处理缺失值...")

        # 识别空间列
        self.spatial_cols = [col for col in df.columns if col.startswith('X')]

        # 删除完全缺失的行
        before_len = len(df)
        df = df.dropna(subset=self.spatial_cols, how='all')
        print(f"删除完全缺失行: {before_len - len(df)} 行")

        # 计算每行缺失比例
        missing_ratio = df[self.spatial_cols].isnull().sum(axis=1) / len(self.spatial_cols)
        valid_mask = missing_ratio < self.config.MAX_MISSING_RATIO
        df = df[valid_mask]
        print(f"删除缺失率>{self.config.MAX_MISSING_RATIO}的行: {(~valid_mask).sum()} 行")

        # 空间插值填充剩余缺失值
        if df[self.spatial_cols].isnull().sum().sum() > 0:
            print("使用空间插值填充剩余缺失值...")
            df[self.spatial_cols] = df[self.spatial_cols].interpolate(
                method=self.config.INTERPOLATION_METHOD,
                axis=1,
                limit_direction='both'
            )

            # 如果还有缺失,用前向填充
            df[self.spatial_cols] = df[self.spatial_cols].fillna(method='ffill', axis=1)
            df[self.spatial_cols] = df[self.spatial_cols].fillna(method='bfill', axis=1)


        print(f"处理后数据形状: {df.shape}")
        return df

    def handle_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理标签"""
        print("\n处理标签...")

        # 删除NA标签
        before_len = len(df)
        df = df[df['status'] != 'NA']
        print(f"删除NA标签: {before_len - len(df)} 行")

        # 映射标签
        df['label'] = df['status'].map(self.config.LABEL_MAPPING)
        df = df[df['label'].isin([0, 1])]  # 只保留有效标签

        print(f"标签分布:\n{df['label'].value_counts()}")

        return df

    def normalize_signals(self, df: pd.DataFrame,
                         method: str = 'high_freq_energy',
                         fit: bool = True) -> pd.DataFrame:
        """
        信号归一化
        method: 'zscore', 'minmax', 'high_freq_energy'
        """
        print(f"\n使用 {method} 方法归一化...")

        spatial_cols = [col for col in df.columns if col.startswith('X')]

        if method == 'zscore':
            if fit:
                self.scaler = StandardScaler()
                df[spatial_cols] = self.scaler.fit_transform(df[spatial_cols])
            else:
                df[spatial_cols] = self.scaler.transform(df[spatial_cols])

        elif method == 'minmax':
            if fit:
                self.scaler = MinMaxScaler(feature_range=(-1, 1))
                df[spatial_cols] = self.scaler.fit_transform(df[spatial_cols])
            else:
                df[spatial_cols] = self.scaler.transform(df[spatial_cols])

        elif method == 'high_freq_energy':
            # 基于高频能量的归一化(应对距离衰减)
            # 参考: Tejedor et al. (2016)
            def normalize_by_high_freq(row):
                signal_data = row.values
                # 使用后20%的点作为高频区域
                high_freq_idx = int(len(signal_data) * self.config.HIGH_FREQ_PERCENTILE)
                high_freq_energy = np.sum(np.abs(signal_data[high_freq_idx:]))

                if high_freq_energy > 1e-10:
                    return signal_data / high_freq_energy
                else:
                    return signal_data
            # ------------------------------------------------------------------
            # ↓↓↓ (修复) 替换下面这行代码 ↓↓↓
            # ------------------------------------------------------------------
            # 原代码 (会导致ValueError):
            # df[spatial_cols] = df[spatial_cols].apply(normalize_by_high_freq, axis=1)

            # 新代码:
            # .apply 返回一个 Series, 其中每个元素是一个 numpy 数组
            result_series = df[spatial_cols].apply(normalize_by_high_freq, axis=1)

            # 我们必须将这个 Series of arrays "解包" 回一个 DataFrame
            # 明确指定 index 和 columns
            df[spatial_cols] = pd.DataFrame(
                result_series.tolist(),
                index=df.index,
                columns=spatial_cols
            )
            # ------------------------------------------------------------------
            # ↑↑↑ 修复结束 ↑↑↑
            # ------------------------------------------------------------------
        return df

    def spectral_subtraction(self, df: pd.DataFrame,
                            noise_percentile: float = 10) -> pd.DataFrame:
        """
        频谱减法去噪
        参考: Xu et al. (2018)
        """
        print("\n应用频谱减法去噪...")

        spatial_cols = [col for col in df.columns if col.startswith('X')]

        # 估计噪声水平(使用低能量样本的平均值)
        signal_energy = df[spatial_cols].apply(lambda x: np.sum(x**2), axis=1)
        noise_threshold = np.percentile(signal_energy, noise_percentile)
        noise_signals = df[signal_energy < noise_threshold][spatial_cols]

        if len(noise_signals) > 0:
            # 计算平均噪声功率谱
            noise_fft = np.fft.fft(noise_signals.values, axis=1)
            noise_power = np.mean(np.abs(noise_fft)**2, axis=0)

            # 对每个信号进行频谱减法
            for idx in df.index:
                signal = df.loc[idx, spatial_cols].values
                #signal_fft = np.fft.fft(signal)

                try:
                    signal_fft = np.fft.fft(signal.astype(float))
                except Exception as e:
                    print(f"!! FFT 失败于 index {idx}: {e}")
                    print(f"   Signal data (first 5): {signal[:5]}")
                    continue  # 跳过这个有问题的信号



                signal_power = np.abs(signal_fft)**2

                # 减去噪声功率谱
                clean_power = np.maximum(signal_power - noise_power, 0)
                clean_fft = np.sqrt(clean_power) * np.exp(1j * np.angle(signal_fft))

                # 逆变换
                clean_signal = np.real(np.fft.ifft(clean_fft))
                df.loc[idx, spatial_cols] = clean_signal

        return df

    def preprocess_pipeline(self, df: pd.DataFrame,
                           fit: bool = True,
                           apply_denoising: bool = False) -> pd.DataFrame:
        """完整预处理流程"""
        print("="*60)
        print("开始数据预处理流程")
        print("="*60)

        # 1. 处理缺失值
        df = self.handle_missing_values(df)

        # 2. 处理标签
        df = self.handle_labels(df)

        # 3. 去噪(可选)
        if apply_denoising:
            df = self.spectral_subtraction(df)

        # 4. 归一化
        df = self.normalize_signals(
            df,
            method=self.config.NORMALIZATION_METHOD,
            fit=fit
        )

        print("\n预处理完成!")
        print("="*60)

        return df

    def save_scaler(self, filepath: str):
        """保存归一化器"""
        if self.scaler is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"归一化器已保存: {filepath}")

    def load_scaler(self, filepath: str):
        """加载归一化器"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"归一化器已加载: {filepath}")


def split_dataset(df: pd.DataFrame, config,
                 stratify: bool = True) -> Tuple[pd.DataFrame, ...]:
    """
    划分数据集为训练/验证/测试集
    """
    print("\n划分数据集...")

    X = df[[col for col in df.columns if col.startswith('X')]]
    y = df['label']

    # 先分出测试集
    stratify_param = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config.TEST_RATIO,
        stratify=stratify_param,
        random_state=config.RANDOM_SEED
    )

    # 再分出验证集
    val_ratio = config.VAL_RATIO / (1 - config.TEST_RATIO)
    stratify_param = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=stratify_param,
        random_state=config.RANDOM_SEED
    )

    print(f"训练集: {X_train.shape}, 标签分布: {y_train.value_counts().to_dict()}")
    print(f"验证集: {X_val.shape}, 标签分布: {y_val.value_counts().to_dict()}")
    print(f"测试集: {X_test.shape}, 标签分布: {y_test.value_counts().to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# 使用示例
if __name__ == "__main__":
    from src.utils.config import Config

    # 加载数据
    loader = DASDataLoader(Config.RAW_DATA_DIR / Config.DATA_FILE)
    df = loader.load_data()

    # 查看基本统计
    stats = loader.get_basic_stats()
    print("\n基本统计:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 预处理
    preprocessor = DASPreprocessor(Config)
    df_processed = preprocessor.preprocess_pipeline(df, fit=True, apply_denoising=True)

    # 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df_processed, Config, stratify=True
    )

    # 保存处理后的数据
    output_path = Config.PROCESSED_DATA_DIR / "processed_data.pkl"
    pd.to_pickle({
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }, output_path)
    print(f"\n处理后数据已保存: {output_path}")

    # 保存归一化器
    preprocessor.save_scaler(Config.PROCESSED_DATA_DIR / "scaler.pkl")