"""
数据预处理模块 - src/data/preprocessing.py
包含: 数据加载、清洗、归一化、特征提取
[!! 新增WPD去噪功能 !!]
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
import pywt  # [!! 新增 !!]

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
        self.noise_power = None

        # [!! 新增 !!] WPD去噪配置
        self.wpd_wavelet = config.WPD_WAVELET
        self.wpd_level = config.WPD_LEVEL
        self.wpd_threshold_method = config.WPD_THRESHOLD_METHOD
        self.wpd_threshold_scale = config.WPD_THRESHOLD_SCALE

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        print("\n处理缺失值...")

        # 识别空间列
        if self.spatial_cols is None:
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
        df = df[df['label'].isin([0, 1])]
        df['label'] = df['label'].astype(int)

        print(f"标签分布:\n{df['label'].value_counts()}")

        return df

    # ==================== [!! 新增 !!] WPD去噪方法 ====================
    def wpd_denoise(self, signal: np.ndarray,
                    threshold_scale: float = None) -> np.ndarray:
        """
        基于WPD的自适应阈值去噪

        参考论文:
        - Donoho & Johnstone (1994): Ideal spatial adaptation via wavelet shrinkage
        - 通用阈值: threshold = sigma * sqrt(2 * log(N))

        Args:
            signal: 原始信号
            threshold_scale: 阈值缩放因子(None则使用config配置)

        Returns:
            去噪后的信号
        """
        if threshold_scale is None:
            threshold_scale = self.wpd_threshold_scale

        try:
            # 1. WPD分解
            wp = pywt.WaveletPacket(
                data=signal,
                wavelet=self.wpd_wavelet,
                mode='symmetric',
                maxlevel=self.wpd_level
            )

            # 2. 获取所有叶节点
            nodes = wp.get_level(self.wpd_level, 'natural')

            # 3. 对每个子带应用阈值
            for node in nodes:
                coeffs = node.data

                # 使用MAD(中位数绝对偏差)估计噪声标准差
                # sigma = MAD / 0.6745 (假设高斯噪声)
                sigma = np.median(np.abs(coeffs)) / 0.6745

                # 通用阈值(Donoho & Johnstone, 1994)
                threshold = sigma * np.sqrt(2 * np.log(len(coeffs))) * threshold_scale

                # 应用阈值
                if self.wpd_threshold_method == 'soft':
                    # Soft thresholding (保留更多细节)
                    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='soft')
                else:
                    # Hard thresholding (更强的去噪)
                    coeffs_thresh = pywt.threshold(coeffs, threshold, mode='hard')

                node.data = coeffs_thresh

            # 4. 重构信号
            denoised = wp.reconstruct(update=True)

            # 5. 确保长度一致
            if len(denoised) > len(signal):
                denoised = denoised[:len(signal)]
            elif len(denoised) < len(signal):
                denoised = np.pad(denoised, (0, len(signal) - len(denoised)), mode='edge')

            return denoised

        except Exception as e:
            print(f"!! WPD去噪失败(index): {e}")
            return signal  # 失败则返回原信号

    def normalize_signals(self, df: pd.DataFrame,
                         method: str = 'zscore',
                         fit: bool = True) -> pd.DataFrame:
        """信号归一化"""
        print(f"\n使用 {method} 方法归一化 (Fit={fit})...")

        if self.spatial_cols is None:
             self.spatial_cols = [col for col in df.columns if col.startswith('X')]

        if method == 'zscore' or method == 'minmax':
            if fit:
                print("拟合 (Fit) 归一化器...")
                if method == 'zscore':
                    self.scaler = StandardScaler()
                elif method == 'minmax':
                    self.scaler = MinMaxScaler(feature_range=(-1, 1))

                df[self.spatial_cols] = self.scaler.fit_transform(df[self.spatial_cols])
            else:
                if self.scaler is None:
                    raise ValueError("归一化器未拟合。请先在训练集上 fit=True。")
                print("应用 (Transform) 归一化器...")
                df[self.spatial_cols] = self.scaler.transform(df[self.spatial_cols])

        elif method == 'high_freq_energy':
            def normalize_by_high_freq(row):
                signal_data = row.values
                high_freq_idx = int(len(signal_data) * self.config.HIGH_FREQ_PERCENTILE)
                high_freq_energy = np.sum(np.abs(signal_data[high_freq_idx:]))

                if high_freq_energy > 1e-10:
                    return signal_data / high_freq_energy
                else:
                    return signal_data

            result_series = df[self.spatial_cols].apply(normalize_by_high_freq, axis=1)
            df[self.spatial_cols] = pd.DataFrame(
                result_series.tolist(),
                index=df.index,
                columns=self.spatial_cols
            )
        return df

    def spectral_subtraction(self, df: pd.DataFrame,
                            noise_percentile: float = 10,
                            fit: bool = True) -> pd.DataFrame:
        """频谱减法去噪"""
        print(f"\n应用频谱减法去噪 (Fit={fit})...")

        if self.spatial_cols is None:
            self.spatial_cols = [col for col in df.columns if col.startswith('X')]

        if fit:
            print("估计噪声谱 (仅使用当前数据集)...")
            signal_energy = df[self.spatial_cols].apply(lambda x: np.sum(x**2), axis=1)
            noise_threshold = np.percentile(signal_energy, noise_percentile)
            noise_signals = df[signal_energy < noise_threshold][self.spatial_cols]

            if len(noise_signals) > 0:
                noise_fft = np.fft.fft(noise_signals.values, axis=1)
                self.noise_power = np.mean(np.abs(noise_fft)**2, axis=0)
            else:
                self.noise_power = np.zeros(len(self.spatial_cols))

        if self.noise_power is None:
             raise ValueError("噪声谱未估计。请先在训练集上 fit=True。")

        # 对每个信号进行频谱减法
        for idx in df.index:
            signal_data = df.loc[idx, self.spatial_cols].values
            try:
                signal_fft = np.fft.fft(signal_data.astype(float))
            except Exception as e:
                print(f"!! FFT 失败于 index {idx}: {e}")
                continue

            signal_power = np.abs(signal_fft)**2
            clean_power = np.maximum(signal_power - self.noise_power, 0)
            clean_fft = np.sqrt(clean_power) * np.exp(1j * np.angle(signal_fft))

            clean_signal = np.real(np.fft.ifft(clean_fft))
            df.loc[idx, self.spatial_cols] = clean_signal

        return df

    def preprocess_pipeline(self, df: pd.DataFrame,
                           fit: bool = True,
                           apply_denoising: bool = False,
                           apply_wpd_denoise: bool = False) -> pd.DataFrame:  # [!! 新增参数 !!]
        """
        完整预处理流程

        [!! 修改 !!] 新增WPD去噪选项
        """
        print("="*60)
        print(f"开始数据预处理流程 (Fit={fit})")
        if apply_wpd_denoise:
            print(f"  → WPD去噪: 启用 (小波={self.wpd_wavelet}, 层数={self.wpd_level})")
        elif apply_denoising:
            print(f"  → 频谱减法去噪: 启用")
        print("="*60)

        # 1. 处理缺失值
        df = self.handle_missing_values(df)

        # 2. 处理标签
        if 'status' in df.columns:
            df = self.handle_labels(df)

        # 3. [!! 新增 !!] WPD去噪(优先级高于频谱减法)
        if apply_wpd_denoise:
            print("\n应用WPD去噪...")
            from tqdm import tqdm

            for idx in tqdm(df.index, desc="WPD去噪进度"):
                signal = df.loc[idx, self.spatial_cols].values
                denoised = self.wpd_denoise(signal)
                df.loc[idx, self.spatial_cols] = denoised

            print("✓ WPD去噪完成")

        # 4. 频谱减法去噪(可选,与WPD二选一)
        elif apply_denoising:
            df = self.spectral_subtraction(df, fit=fit)

        # 5. 归一化
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


def temporal_split_dataset(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间顺序划分数据集"""
    print("\n按时间顺序划分数据集 (Temporal Split)...")

    n_samples = len(df)

    # 计算切分点
    train_end = int(n_samples * config.TRAIN_RATIO)
    val_end = int(n_samples * (config.TRAIN_RATIO + config.VAL_RATIO))

    # 切分数据
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    if 'label' in df.columns:
        print(f"训练集: {df_train.shape}, 标签分布: {df_train['label'].value_counts().to_dict()}")
        print(f"验证集: {df_val.shape}, 标签分布: {df_val['label'].value_counts().to_dict()}")
        print(f"测试集: {df_test.shape}, 标签分布: {df_test['label'].value_counts().to_dict()}")
    else:
        print(f"训练集: {df_train.shape}")
        print(f"验证集: {df_val.shape}")
        print(f"测试集: {df_test.shape}")

    return df_train, df_val, df_test


# 使用示例
if __name__ == "__main__":
    from src.utils.config import Config

    # 1. 加载数据
    loader = DASDataLoader(Config.RAW_DATA_DIR / Config.DATA_FILE)
    df_raw = loader.load_data()

    # 2. 初始化预处理器
    preprocessor = DASPreprocessor(Config)

    # 3. 处理标签
    df_labeled = preprocessor.handle_labels(df_raw)

    # 4. 划分数据集
    df_train, df_val, df_test = temporal_split_dataset(df_labeled, Config)

    # 5. 预处理(测试WPD去噪)
    print("\n" + "="*60)
    print("测试WPD去噪预处理")
    print("="*60)

    df_train_processed = preprocessor.preprocess_pipeline(
        df_train.head(100),  # 测试用小样本
        fit=True,
        apply_wpd_denoise=True  # [!! 测试WPD去噪 !!]
    )

    print(f"\n处理后数据形状: {df_train_processed.shape}")