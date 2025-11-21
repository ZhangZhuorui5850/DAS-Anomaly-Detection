"""
深度学习模型集合 - src/models/deep_learning/deep_models.py
整合了 LSTM-CNN, LSTM-AE, 1D-CNN, 2D-CNN 所有深度模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. LSTM-CNN 混合模型
# ==========================================
class LSTM_CNN(nn.Module):
    """
    LSTM-CNN混合网络: 先用CNN提取空间特征,再用LSTM捕获时序依赖
    """

    def __init__(self, config):
        super(LSTM_CNN, self).__init__()
        self.config = config
        lstm_cnn_cfg = config.LSTM_CNN_CONFIG

        if not hasattr(config, 'INPUT_FEATURES'):
            raise ValueError("Config 中缺少 INPUT_FEATURES 属性! (请在 config.py 中设置)")

        # 计算 CNN 池化后的输出长度
        cnn_output_len = config.INPUT_FEATURES
        for _ in lstm_cnn_cfg['cnn_filters']:
            cnn_output_len = cnn_output_len // 2

        # CNN分支
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        in_channels = 1
        for filters, kernel_size in zip(lstm_cnn_cfg['cnn_filters'], lstm_cnn_cfg['cnn_kernel_sizes']):
            self.conv_layers.append(nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size // 2))
            self.bn_layers.append(nn.BatchNorm1d(filters))
            self.pool_layers.append(nn.MaxPool1d(2))
            in_channels = filters

        # LSTM分支
        self.lstm = nn.LSTM(
            input_size=cnn_output_len * lstm_cnn_cfg['cnn_filters'][-1],
            hidden_size=lstm_cnn_cfg['lstm_hidden_size'],
            num_layers=lstm_cnn_cfg['lstm_num_layers'],
            batch_first=True,
            dropout=lstm_cnn_cfg['dropout'] if lstm_cnn_cfg['lstm_num_layers'] > 1 else 0
        )

        self.dropout = nn.Dropout(lstm_cnn_cfg['dropout'])

        # 全连接层
        self.fc_layers = nn.ModuleList()
        fc_input_size = lstm_cnn_cfg['lstm_hidden_size']
        for fc_hidden in lstm_cnn_cfg['fc_hidden_sizes']:
            self.fc_layers.append(nn.Linear(fc_input_size, fc_hidden))
            fc_input_size = fc_hidden

        self.fc_out = nn.Linear(fc_input_size, lstm_cnn_cfg['num_classes'])

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape

        # 重塑用于CNN: (batch*seq_len, 1, features)
        x = x.reshape(batch_size * seq_len, features, 1).permute(0, 2, 1)

        # CNN特征提取
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = pool(F.relu(bn(conv(x))))

        # 重塑用于LSTM: (batch, seq_len, flattened_features)
        x = x.permute(0, 2, 1).reshape(batch_size, seq_len, -1)

        # LSTM处理
        lstm_out, _ = self.lstm(x)

        # 取最后一个时间步
        x = self.dropout(lstm_out[:, -1, :])

        # 全连接分类
        for fc in self.fc_layers:
            x = self.dropout(F.relu(fc(x)))

        return self.fc_out(x)


# ==========================================
# 2. 2D-CNN 时空模型 (原 cnn_2d.py)
# ==========================================
class CNN_2D_SpatioTemporal(nn.Module):
    """处理 (Freq/Space, Time) 二维图的 CNN"""

    def __init__(self, config):
        super(CNN_2D_SpatioTemporal, self).__init__()
        cnn_2d_cfg = getattr(config, 'CNN_2D_CONFIG', {
            'conv_channels': [32, 64, 128], 'fc_hidden': [256], 'dropout': 0.7, 'num_classes': 2
        })

        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_2d_cfg['conv_channels']:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
            in_channels = out_channels

        # 动态计算 Flatten 大小
        if not (hasattr(config, 'WINDOW_SIZE') and hasattr(config, 'INPUT_FEATURES')):
            raise ValueError("Config 缺少 WINDOW_SIZE 或 INPUT_FEATURES")

        with torch.no_grad():
            dummy = torch.randn(1, 1, config.WINDOW_SIZE, config.INPUT_FEATURES)
            for layer in self.conv_layers:
                dummy = layer(dummy)
            self.flattened_size = dummy.view(1, -1).size(1)

        self.fc_layers = nn.ModuleList()
        fc_input = self.flattened_size
        for fc_hidden in cnn_2d_cfg['fc_hidden']:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(fc_input, fc_hidden),
                nn.ReLU(),
                nn.Dropout(cnn_2d_cfg['dropout'])
            ))
            fc_input = fc_hidden

        self.fc_out = nn.Linear(fc_input, cnn_2d_cfg['num_classes'])

    def forward(self, x):
        # x shape: (batch, Time, Features) -> 需要变为 (batch, 1, Time, Features)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        for layer in self.conv_layers:
            x = layer(x)

        x = x.view(x.size(0), -1)

        for fc in self.fc_layers:
            x = fc(x)

        return self.fc_out(x)


# ==========================================
# 模型工厂函数
# ==========================================
def create_model(model_type: str, config):
    """统一的模型创建入口"""
    model_dict = {
        'lstm_cnn': LSTM_CNN,
        'cnn_2d': CNN_2D_SpatioTemporal
    }

    if model_type not in model_dict:
        raise ValueError(f"未知模型类型: {model_type}")

    model = model_dict[model_type](config)
    model = model.to(config.DEVICE)

    # 打印统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型创建成功: {model_type.upper()}")
    print(f"参数量: {trainable_params:,}")

    return model