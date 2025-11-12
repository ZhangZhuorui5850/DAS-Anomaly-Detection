"""
特征提取模块 - src/features/feature_extraction.py
提取时域、频域、时频域特征
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


class FeatureExtractor:
    """DAS信号特征提取器"""

    def __init__(self, config):
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.num_freq_bands = config.NUM_FREQ_BANDS

    def extract_time_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取时域特征
        参考: Tejedor et al. (2016)
        """
        features = {}

        # 基础统计特征
        features['energy'] = np.sum(signal ** 2)
        features['max_amplitude'] = np.max(np.abs(signal))
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['variance'] = np.var(signal)
        features['rms'] = np.sqrt(np.mean(signal ** 2))

        # 高阶统计特征
        features['skewness'] = stats.skew(signal)
        features['kurtosis'] = stats.kurtosis(signal)

        # 过零率
        zero_crossings = np.where(np.diff(np.sign(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)

        # 峰值相关
        features['peak_to_peak'] = np.ptp(signal)
        features['crest_factor'] = features['max_amplitude'] / features['rms'] if features['rms'] > 0 else 0

        # 能量集中度
        signal_abs = np.abs(signal)
        if signal_abs.sum() > 0:
            features['energy_entropy'] = -np.sum(
                (signal_abs / signal_abs.sum()) *
                np.log(signal_abs / signal_abs.sum() + 1e-10)
            )
        else:
            features['energy_entropy'] = 0

        return features

    def extract_frequency_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取频域特征
        参考: Xu et al. (2018) - Energy in Frequency Bands
        """
        features = {}

        # FFT变换
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / self.sample_rate)[:N // 2]
        power_spectrum = 2.0 / N * np.abs(yf[0:N // 2])

        # 总能量
        features['freq_total_energy'] = np.sum(power_spectrum ** 2)

        # 分频带能量
        freq_max = self.sample_rate / 2
        band_edges = np.linspace(0, freq_max, self.num_freq_bands + 1)

        for i in range(self.num_freq_bands):
            band_mask = (xf >= band_edges[i]) & (xf < band_edges[i + 1])
            band_energy = np.sum(power_spectrum[band_mask] ** 2)
            features[f'freq_band_{i}_energy'] = band_energy

            # 归一化频带能量
            if features['freq_total_energy'] > 0:
                features[f'freq_band_{i}_ratio'] = band_energy / features['freq_total_energy']
            else:
                features[f'freq_band_{i}_ratio'] = 0

        # 谱质心
        if power_spectrum.sum() > 0:
            features['spectral_centroid'] = np.sum(xf * power_spectrum) / power_spectrum.sum()
        else:
            features['spectral_centroid'] = 0

        # 谱扩散
        if power_spectrum.sum() > 0:
            features['spectral_spread'] = np.sqrt(
                np.sum(((xf - features['spectral_centroid']) ** 2) * power_spectrum) /
                power_spectrum.sum()
            )
        else:
            features['spectral_spread'] = 0

        # 谱熵
        psd_norm = power_spectrum / (power_spectrum.sum() + 1e-10)
        features['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm + 1e-10))

        # 主频率
        dominant_freq_idx = np.argmax(power_spectrum)
        features['dominant_frequency'] = xf[dominant_freq_idx]
        features['dominant_frequency_magnitude'] = power_spectrum[dominant_freq_idx]

        return features

    def extract_spatial_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取空间域特征
        针对DAS的空间分布特性
        """
        features = {}

        # 空间梯度
        spatial_gradient = np.abs(np.diff(signal))
        features['spatial_gradient_mean'] = np.mean(spatial_gradient)
        features['spatial_gradient_max'] = np.max(spatial_gradient)
        features['spatial_gradient_std'] = np.std(spatial_gradient)

        # 空间方差
        features['spatial_variance'] = np.var(signal)

        # 能量最大位置
        features['max_energy_position'] = np.argmax(np.abs(signal)) / len(signal)

        # 能量质心(空间)
        signal_abs = np.abs(signal)
        if signal_abs.sum() > 0:
            positions = np.arange(len(signal))
            features['energy_centroid'] = np.sum(positions * signal_abs) / signal_abs.sum()
            features['energy_centroid_normalized'] = features['energy_centroid'] / len(signal)
        else:
            features['energy_centroid'] = 0
            features['energy_centroid_normalized'] = 0

        # 能量扩散(空间)
        if signal_abs.sum() > 0:
            features['energy_spread'] = np.sqrt(
                np.sum(((positions - features['energy_centroid']) ** 2) * signal_abs) /
                signal_abs.sum()
            )
        else:
            features['energy_spread'] = 0

        # 局部能量峰值数量
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal_abs, height=np.mean(signal_abs))
        features['num_peaks'] = len(peaks)

        return features

    def extract_timefreq_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取时频域特征
        使用短时傅里叶变换(STFT)
        """
        features = {}

        # STFT
        nperseg = min(256, len(signal) // 4)
        f, t, Zxx = signal.stft(signal, fs=self.sample_rate, nperseg=nperseg)

        # 时频图的统计特征
        spectrogram = np.abs(Zxx)
        features['stft_mean'] = np.mean(spectrogram)
        features['stft_std'] = np.std(spectrogram)
        features['stft_max'] = np.max(spectrogram)

        # 时间-频率能量分布
        time_energy = np.sum(spectrogram, axis=0)
        freq_energy = np.sum(spectrogram, axis=1)

        features['time_energy_var'] = np.var(time_energy)
        features['freq_energy_var'] = np.var(freq_energy)

        return features

    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """提取所有特征"""
        features = {}

        # 时域特征
        time_features = self.extract_time_features(signal)
        features.update(time_features)

        # 频域特征
        freq_features = self.extract_frequency_features(signal)
        features.update(freq_features)

        # 空间特征
        spatial_features = self.extract_spatial_features(signal)
        features.update(spatial_features)

        # 时频特征
        # timefreq_features = self.extract_timefreq_features(signal)
        # features.update(timefreq_features)

        return features

    def extract_features_batch(self, X: pd.DataFrame,
                               show_progress: bool = True) -> pd.DataFrame:
        """批量提取特征"""
        from tqdm import tqdm

        feature_list = []

        iterator = tqdm(range(len(X))) if show_progress else range(len(X))

        for idx in iterator:
            signal = X.iloc[idx].values
            features = self.extract_all_features(signal)
            feature_list.append(features)

        feature_df = pd.DataFrame(feature_list, index=X.index)

        print(f"\n提取的特征数量: {feature_df.shape[1]}")
        print(f"特征名称: {list(feature_df.columns[:10])}... (显示前10个)")

        return feature_df


class SequenceGenerator:
    """
    为深度学习模型生成时间序列
    """

    def __init__(self, window_size: int = 5, stride: int = 1):
        self.window_size = window_size
        self.stride = stride

    def create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """
        创建滑动窗口序列

        Args:
            X: shape (n_samples, n_features)
            y: shape (n_samples,)

        Returns:
            X_seq: shape (n_sequences, window_size, n_features)
            y_seq: shape (n_sequences,) if y is provided
        """
        n_samples, n_features = X.shape
        n_sequences = (n_samples - self.window_size) // self.stride + 1

        X_seq = np.zeros((n_sequences, self.window_size, n_features))

        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            X_seq[i] = X[start_idx:end_idx]

        if y is not None:
            # 使用窗口最后一个样本的标签
            y_seq = np.array([
                y[i * self.stride + self.window_size - 1]
                for i in range(n_sequences)
            ])
            return X_seq, y_seq

        return X_seq

    def create_sequences_for_cnn(self, X: np.ndarray):
        """
        为1D-CNN准备数据

        Returns:
            X_cnn: shape (n_samples, n_features, 1)
        """
        return X.reshape(X.shape[0], X.shape[1], 1)


# 使用示例
if __name__ == "__main__":
    from src.utils.config import Config
    import pickle

    # 加载预处理后的数据
    data_path = Config.PROCESSED_DATA_DIR / "processed_data.pkl"
    data = pd.read_pickle(data_path)

    X_train = data['X_train']
    y_train = data['y_train']

    print("原始数据形状:", X_train.shape)

    # 1. 提取特征(用于经典ML)
    print("\n提取特征用于经典机器学习...")
    extractor = FeatureExtractor(Config)
    features_train = extractor.extract_features_batch(X_train, show_progress=True)

    print(f"特征矩阵形状: {features_train.shape}")
    print(f"\n特征列表:\n{list(features_train.columns)}")

    # 保存特征
    feature_output = Config.FEATURES_DIR / "features_train.pkl"
    pd.to_pickle({
        'features': features_train,
        'labels': y_train
    }, feature_output)
    print(f"\n特征已保存: {feature_output}")

    # 2. 生成序列(用于深度学习)
    print("\n生成序列用于深度学习...")
    seq_generator = SequenceGenerator(
        window_size=Config.WINDOW_SIZE,
        stride=Config.WINDOW_STRIDE
    )

    X_seq, y_seq = seq_generator.create_sequences(
        X_train.values,
        y_train.values
    )
    print(f"序列数据形状: {X_seq.shape}")
    print(f"序列标签形状: {y_seq.shape}")

    # 保存序列数据
    seq_output = Config.FEATURES_DIR / "sequences_train.pkl"
    pd.to_pickle({
        'X_seq': X_seq,
        'y_seq': y_seq
    }, seq_output)
    print(f"序列数据已保存: {seq_output}")