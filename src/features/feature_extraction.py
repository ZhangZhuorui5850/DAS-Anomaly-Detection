"""
特征提取模块 - src/features/feature_extraction.py
提取时域、频域、空间域特征
[!! 新增WPD特征提取器 !!]
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from typing import Dict, List
import warnings
import pywt  # [!! 新增 !!]
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ==================== [!! 新增 !!] WPD特征提取器 ====================
class WPDFeatureExtractor:
    """
    小波包分解(Wavelet Packet Decomposition)特征提取器

    参考论文:
    - Zhang et al. (2021): EMD + RF, 92.31%
    - Duan et al. (2023): CEEMDAN-Permutation Entropy + RBF, 88.15%
    """

    def __init__(self, wavelet='db4', level=3, config=None):
        """
        Args:
            wavelet: 小波基函数
                - 'db4': Daubechies 4 (平衡性好,论文常用)
                - 'sym5': Symlets 5 (对称性好)
                - 'coif3': Coiflets 3 (紧支撑)
            level: 分解层数
                - 3层 → 2^3 = 8个子带
                - 4层 → 2^4 = 16个子带
                - 5层 → 2^5 = 32个子带
            config: 配置对象
        """
        self.wavelet = wavelet
        self.level = level
        self.config = config

        # 计算子带数量
        self.num_subbands = 2 ** self.level

        print(f"初始化WPD特征提取器: 小波={wavelet}, 层数={level}, 子带数={self.num_subbands}")

    def extract_wpd_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        从单个信号中提取WPD特征

        Args:
            signal: 1D信号数组 (shape: [n_features,])

        Returns:
            包含以下特征的字典:
            1. 各子带能量 (wpd_energy_L_B)
            2. 各子带能量占比 (wpd_energy_ratio_L_B)
            3. 各子带熵 (wpd_entropy_L_B)
            4. 最优子带索引 (wpd_dominant_band)
            5. 能量集中度 (wpd_energy_concentration)
            6. 子带能量标准差 (wpd_energy_std)
        """
        features = {}

        try:
            # 1. 执行小波包分解
            wp = pywt.WaveletPacket(
                data=signal,
                wavelet=self.wavelet,
                mode='symmetric',  # 对称扩展(减少边界效应)
                maxlevel=self.level
            )

            # 2. 提取所有叶节点(终端节点)
            # 'natural'顺序: [aaa, aad, ada, add, daa, dad, dda, ddd] (3层)
            nodes = wp.get_level(self.level, 'natural')

            # 3. 计算各子带能量
            energies = []
            for i, node in enumerate(nodes):
                coeffs = node.data
                # 能量 = 系数平方和
                energy = np.sum(coeffs ** 2)
                energies.append(energy)
                features[f'wpd_energy_{self.level}_{i}'] = energy

            energies = np.array(energies)
            total_energy = np.sum(energies)

            # 4. 能量归一化(能量占比)
            if total_energy > 1e-10:
                for i, energy in enumerate(energies):
                    features[f'wpd_energy_ratio_{self.level}_{i}'] = energy / total_energy
            else:
                for i in range(len(energies)):
                    features[f'wpd_energy_ratio_{self.level}_{i}'] = 0.0

            # 5. 子带熵(Shannon熵)
            # 熵越大 → 能量越分散 → 可能是噪声
            # 熵越小 → 能量越集中 → 可能是有用信号
            for i, node in enumerate(nodes):
                coeffs = np.abs(node.data)
                if coeffs.sum() > 1e-10:
                    prob = coeffs / coeffs.sum()
                    ent = -np.sum(prob * np.log(prob + 1e-10))  # Shannon熵
                    features[f'wpd_entropy_{self.level}_{i}'] = ent
                else:
                    features[f'wpd_entropy_{self.level}_{i}'] = 0.0

            # 6. 全局统计特征
            # 最优子带(能量最大的子带索引)
            features['wpd_dominant_band'] = np.argmax(energies)

            # 能量集中度(最大能量/总能量)
            if total_energy > 1e-10:
                features['wpd_energy_concentration'] = np.max(energies) / total_energy
            else:
                features['wpd_energy_concentration'] = 0.0

            # 子带能量标准差(能量分布的均匀性)
            features['wpd_energy_std'] = np.std(energies)

            # 子带能量的变异系数(CV = std / mean)
            if np.mean(energies) > 1e-10:
                features['wpd_energy_cv'] = np.std(energies) / np.mean(energies)
            else:
                features['wpd_energy_cv'] = 0.0

            # 高频能量比(假设后半部分子带是高频)
            high_freq_start = len(energies) // 2
            high_freq_energy = np.sum(energies[high_freq_start:])
            if total_energy > 1e-10:
                features['wpd_high_freq_ratio'] = high_freq_energy / total_energy
            else:
                features['wpd_high_freq_ratio'] = 0.0

        except Exception as e:
            print(f"!! WPD特征提取失败: {e}")
            # 填充默认值
            for i in range(self.num_subbands):
                features[f'wpd_energy_{self.level}_{i}'] = 0.0
                features[f'wpd_energy_ratio_{self.level}_{i}'] = 0.0
                features[f'wpd_entropy_{self.level}_{i}'] = 0.0
            features['wpd_dominant_band'] = 0
            features['wpd_energy_concentration'] = 0.0
            features['wpd_energy_std'] = 0.0
            features['wpd_energy_cv'] = 0.0
            features['wpd_high_freq_ratio'] = 0.0

        return features

    def extract_features_batch(self, X: pd.DataFrame,
                               show_progress: bool = True) -> pd.DataFrame:
        """批量提取WPD特征"""
        feature_list = []
        iterator = tqdm(range(len(X)), desc="提取WPD特征") if show_progress else range(len(X))

        for idx in iterator:
            signal = X.iloc[idx].values
            wpd_features = self.extract_wpd_features(signal)
            feature_list.append(wpd_features)

        wpd_df = pd.DataFrame(feature_list, index=X.index)
        print(f"\n✓ 提取的WPD特征数量: {wpd_df.shape[1]}")
        return wpd_df


# ==================== 原有特征提取器(整合WPD) ====================
class FeatureExtractor:
    """DAS信号特征提取器"""

    def __init__(self, config):
        self.config = config
        self.sample_rate = config.SAMPLE_RATE
        self.num_freq_bands = config.NUM_FREQ_BANDS

        # [!! 新增 !!] WPD提取器
        if config.ENABLE_WPD_FEATURES:
            self.wpd_extractor = WPDFeatureExtractor(
                wavelet=config.WPD_FEATURE_WAVELET,
                level=config.WPD_FEATURE_LEVEL,
                config=config
            )
        else:
            self.wpd_extractor = None

    def extract_time_features(self, signal: np.ndarray) -> Dict[str, float]:
        """提取时域特征"""
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
        """提取频域特征"""
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
        """提取空间域特征"""
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

    def extract_all_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取所有特征(原有 + WPD)

        [!! 修改 !!] 整合WPD特征
        """
        features = {}

        # 原有特征
        time_features = self.extract_time_features(signal)
        features.update(time_features)

        freq_features = self.extract_frequency_features(signal)
        features.update(freq_features)

        spatial_features = self.extract_spatial_features(signal)
        features.update(spatial_features)

        # [!! 新增 !!] WPD特征
        if self.wpd_extractor is not None:
            wpd_features = self.wpd_extractor.extract_wpd_features(signal)
            features.update(wpd_features)

        return features

    def extract_features_batch(self, X: pd.DataFrame,
                               show_progress: bool = True) -> pd.DataFrame:
        """批量提取特征"""
        feature_list = []

        iterator = tqdm(range(len(X)), desc="提取所有特征") if show_progress else range(len(X))

        for idx in iterator:
            signal = X.iloc[idx].values
            features = self.extract_all_features(signal)
            feature_list.append(features)

        feature_df = pd.DataFrame(feature_list, index=X.index)

        print(f"\n✓ 提取的特征总数: {feature_df.shape[1]}")
        print(f"  - 时域特征: ~12个")
        print(f"  - 频域特征: ~{10 + 2*self.num_freq_bands}个")
        print(f"  - 空间特征: ~8个")
        if self.wpd_extractor is not None:
            num_wpd_features = 2 * self.wpd_extractor.num_subbands + self.wpd_extractor.num_subbands + 5
            print(f"  - WPD特征: ~{num_wpd_features}个")

        return feature_df


# ==================== 序列生成器(原有,不变) ====================
class SequenceGenerator:
    """为深度学习模型生成时间序列"""

    def __init__(self, window_size: int = 5, stride: int = 1):
        self.window_size = window_size
        self.stride = stride

    def create_sequences(self, X: np.ndarray, y: np.ndarray = None):
        """创建滑动窗口序列"""
        n_samples, n_features = X.shape
        n_sequences = (n_samples - self.window_size) // self.stride + 1

        X_seq = np.zeros((n_sequences, self.window_size, n_features))

        for i in range(n_sequences):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            X_seq[i] = X[start_idx:end_idx]

        if y is not None:
            y_seq = np.array([
                y[i * self.stride + self.window_size - 1]
                for i in range(n_sequences)
            ])
            return X_seq, y_seq

        return X_seq


# ==================== 使用示例 ====================
if __name__ == "__main__":
    from src.utils.config import Config

    # 加载预处理后的数据
    data_path = Config.PROCESSED_DATA_DIR / "processed_data.pkl"

    if not data_path.exists():
        print(f"错误: 数据文件不存在: {data_path}")
        print("请先运行: python main.py --mode preprocess")
    else:
        data = pd.read_pickle(data_path)
        X_train = data['X_train']
        y_train = data['y_train']

        print("原始数据形状:", X_train.shape)

        # 测试WPD特征提取器
        print("\n" + "="*60)
        print("测试WPD特征提取器")
        print("="*60)

        wpd_extractor = WPDFeatureExtractor(wavelet='db4', level=3, config=Config)
        wpd_features = wpd_extractor.extract_features_batch(X_train.head(10), show_progress=True)
        print(f"\nWPD特征矩阵形状: {wpd_features.shape}")
        print(f"WPD特征列表:\n{list(wpd_features.columns[:10])}...")

        # 测试完整特征提取
        print("\n" + "="*60)
        print("测试完整特征提取(原有 + WPD)")
        print("="*60)

        extractor = FeatureExtractor(Config)
        features_train = extractor.extract_features_batch(X_train.head(10), show_progress=True)

        print(f"\n完整特征矩阵形状: {features_train.shape}")
        print(f"\n特征列表示例:\n{list(features_train.columns[:15])}...")