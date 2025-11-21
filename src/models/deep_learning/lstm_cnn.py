"""
LSTM-CNN混合模型 - src/models/deep_learning/lstm_cnn.py
参考文献: Duraj et al. (2025), Xu et al. (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn_2d import CNN_2D_SpatioTemporal  # <--- 添加这一行


class LSTM_CNN(nn.Module):
    """
    LSTM-CNN混合网络
    先用CNN提取空间特征,再用LSTM捕获时序依赖
    """

    def __init__(self, config):
        super(LSTM_CNN, self).__init__()

        self.config = config
        lstm_cnn_cfg = config.LSTM_CNN_CONFIG

        # ------------------------------------------------------------------
        # (修复) 步骤 1: 检查并计算 CNN 池化后的输出长度
        # ------------------------------------------------------------------
        if not hasattr(config, 'INPUT_FEATURES'):
            raise ValueError("Config 中缺少 INPUT_FEATURES 属性! (请在 config.py 中设置 INPUT_FEATURES = 101)")

        cnn_output_len = config.INPUT_FEATURES  # 101

        # 模拟 CNN 的 MaxPool1d(2)
        # config 中有几个 cnn_filters (3个), 就池化几次
        for _ in lstm_cnn_cfg['cnn_filters']:
            cnn_output_len = cnn_output_len // 2

            # cnn_output_len 现在应该是 12 (101 // 2 -> 50 // 2 -> 25 // 2 -> 12)
        # ------------------------------------------------------------------

        # CNN分支 - 提取空间特征 (这部分不变)
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        in_channels = 1
        for filters, kernel_size in zip(lstm_cnn_cfg['cnn_filters'],
                                        lstm_cnn_cfg['cnn_kernel_sizes']):
            self.conv_layers.append(
                nn.Conv1d(in_channels, filters, kernel_size,
                          padding=kernel_size // 2)
            )
            self.bn_layers.append(nn.BatchNorm1d(filters))
            self.pool_layers.append(nn.MaxPool1d(2))
            in_channels = filters

        # ------------------------------------------------------------------
        # (修复) 步骤 2: 使用计算出的维度来定义 LSTM
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            # 原代码: input_size=lstm_cnn_cfg['cnn_filters'][-1], (这是 64)
            # 新代码: (展平后的大小, 12 * 64 = 768)
            input_size=cnn_output_len * lstm_cnn_cfg['cnn_filters'][-1],

            hidden_size=lstm_cnn_cfg['lstm_hidden_size'],
            num_layers=lstm_cnn_cfg['lstm_num_layers'],
            batch_first=True,
            dropout=lstm_cnn_cfg['dropout'] if lstm_cnn_cfg['lstm_num_layers'] > 1 else 0
        )
        # ------------------------------------------------------------------

        # Dropout
        self.dropout = nn.Dropout(lstm_cnn_cfg['dropout'])

        # 全连接层
        self.fc_layers = nn.ModuleList()
        fc_input_size = lstm_cnn_cfg['lstm_hidden_size']

        for fc_hidden in lstm_cnn_cfg['fc_hidden_sizes']:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_hidden))
            fc_input_size = fc_hidden

        # 输出层
        self.fc_out = nn.Linear(fc_input_size, lstm_cnn_cfg['num_classes'])

    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, features)
        Returns:
            output: shape (batch, num_classes)
        """
        batch_size, seq_len, features = x.shape

        # 重塑为 (batch*seq_len, features, 1) 用于CNN
        x = x.reshape(batch_size * seq_len, features, 1)
        x = x.permute(0, 2, 1)  # (batch*seq_len, 1, features)

        # CNN特征提取
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = pool(x)

        # 重塑为 (batch, seq_len, cnn_features)
        x = x.permute(0, 2, 1)  # (batch*seq_len, reduced_features, channels)
        _, reduced_features, channels = x.shape
        x = x.reshape(batch_size, seq_len, -1)  # 展平CNN输出

        # LSTM处理
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 取最后一个时间步的输出
        x = lstm_out[:, -1, :]
        x = self.dropout(x)

        # 全连接层
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)

        # 输出层
        output = self.fc_out(x)

        return output


class LSTM_Autoencoder(nn.Module):
    """
    LSTM自编码器用于异常检测
    基于重构误差
    """

    def __init__(self, config):
        super(LSTM_Autoencoder, self).__init__()

        self.config = config  # (修复) 这一行被你漏掉了
        lstm_ae_cfg = config.LSTM_AE_CONFIG

        # 编码器
        self.encoder_layers = nn.ModuleList()

        # ------------------------------------------------------------------
        # ↓↓↓ (修复) input_size 必须从 config.INPUT_FEATURES (101) 开始 ↓↓↓
        # ------------------------------------------------------------------
        if not hasattr(config, 'INPUT_FEATURES'):
            raise ValueError("Config 中缺少 INPUT_FEATURES 属性! (请在 config.py 中设置)")

        # 初始 input_size 是特征维度 (101)
        input_size = config.INPUT_FEATURES
        # ------------------------------------------------------------------
        # ↑↑↑ 修复结束 ↑↑↑
        # ------------------------------------------------------------------

        for hidden_size in lstm_ae_cfg['encoder_hidden_sizes']:
            self.encoder_layers.append(
                nn.LSTM(input_size,  # <-- (修复) 移除 "if input_size else hidden_size"
                        hidden_size, batch_first=True)
            )
            # 更新 input_size 为当前层的输出, 作为下一层的输入
            input_size = hidden_size

        # 解码器
        self.decoder_layers = nn.ModuleList()
        for hidden_size in lstm_ae_cfg['decoder_hidden_sizes']:
            self.decoder_layers.append(
                nn.LSTM(input_size, hidden_size, batch_first=True)
            )
            input_size = hidden_size

        # 输出层
        self.output_layer = None  # 将在forward时动态创建
        self.dropout = nn.Dropout(lstm_ae_cfg['dropout'])

    def forward(self, x):
        """
        Args:
            x: shape (batch, seq_len, features)
        Returns:
            reconstructed: shape (batch, seq_len, features)
        """
        batch_size, seq_len, features = x.shape

        # 编码
        encoded = x
        for encoder in self.encoder_layers:
            encoded, _ = encoder(encoded)
            encoded = self.dropout(encoded)

        # 解码
        decoded = encoded
        for decoder in self.decoder_layers:
            decoded, _ = decoder(decoded)
            decoded = self.dropout(decoded)

        # 重构
        if self.output_layer is None:
            self.output_layer = nn.Linear(decoded.shape[-1], features).to(x.device)

        reconstructed = self.output_layer(decoded)

        return reconstructed

    def get_reconstruction_error(self, x):
        """计算重构误差"""
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return mse


class CNN_1D(nn.Module):
    """
    1D-CNN模型
    用于空间模式识别
    """

    def __init__(self, config):
        super(CNN_1D, self).__init__()

        cnn_cfg = config.CNN_1D_CONFIG

        # 卷积块
        self.conv_blocks = nn.ModuleList()
        in_channels = 1

        for out_channels, kernel_size, pool_size in zip(
                cnn_cfg['conv_channels'],
                cnn_cfg['kernel_sizes'],
                cnn_cfg['pool_sizes']
        ):
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(cnn_cfg['dropout'])
            )
            self.conv_blocks.append(block)
            in_channels = out_channels

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc_layers = nn.ModuleList()
        fc_input_size = cnn_cfg['conv_channels'][-1]

        for fc_hidden in cnn_cfg['fc_hidden_sizes']:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_hidden))
            fc_input_size = fc_hidden

        # 输出层
        self.fc_out = nn.Linear(fc_input_size, cnn_cfg['num_classes'])
        self.dropout = nn.Dropout(cnn_cfg['dropout'])

    def forward(self, x):
        """
        Args:
            x: shape (batch, features, 1) or (batch, features)
        Returns:
            output: shape (batch, num_classes)
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(2)  # (batch, features, 1)

        x = x.permute(0, 2, 1)  # (batch, 1, features)

        # 卷积块
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)  # (batch, channels)

        # 全连接层
        for fc in self.fc_layers:
            x = F.relu(fc(x))
            x = self.dropout(x)

        # 输出
        output = self.fc_out(x)

        return output


# 模型工厂函数
def create_model(model_type: str, config):
    """
    创建模型

    Args:
        model_type: 'lstm_cnn', 'lstm_ae', 'cnn_1d'
        config: 配置对象
    """
    model_dict = {
        'lstm_cnn': LSTM_CNN,
        'lstm_ae': LSTM_Autoencoder,
        'cnn_1d': CNN_1D,
        'cnn_2d': CNN_2D_SpatioTemporal
    }

    if model_type not in model_dict:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model_dict[model_type](config)

    # 移到设备
    model = model.to(config.DEVICE)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n模型: {model_type}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"设备: {config.DEVICE}")

    return model


# 测试代码
if __name__ == "__main__":
    from src.utils.config import Config

    # 测试LSTM-CNN
    print("=" * 60)
    print("测试 LSTM-CNN 模型")
    print("=" * 60)

    model = create_model('lstm_cnn', Config)

    # 创建测试输入
    batch_size = 8
    seq_len = Config.WINDOW_SIZE
    features = 100  # 假设100个空间测量点

    x = torch.randn(batch_size, seq_len, features).to(Config.DEVICE)

    # 前向传播
    output = model(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 测试LSTM Autoencoder
    print("\n" + "=" * 60)
    print("测试 LSTM Autoencoder 模型")
    print("=" * 60)

    model_ae = create_model('lstm_ae', Config)
    reconstructed = model_ae(x)
    error = model_ae.get_reconstruction_error(x)

    print(f"\n输入形状: {x.shape}")
    print(f"重构形状: {reconstructed.shape}")
    print(f"重构误差形状: {error.shape}")

    # 测试1D-CNN
    print("\n" + "=" * 60)
    print("测试 1D-CNN 模型")
    print("=" * 60)

    model_cnn = create_model('cnn_1d', Config)
    x_2d = torch.randn(batch_size, features).to(Config.DEVICE)
    output_cnn = model_cnn(x_2d)

    print(f"\n输入形状: {x_2d.shape}")
    print(f"输出形状: {output_cnn.shape}")