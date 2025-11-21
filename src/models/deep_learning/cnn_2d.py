import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_2D_SpatioTemporal(nn.Module):
    """
    [!! 已修改 !!]
    一个新的2D-CNN模型, 用于处理 (Freq, Time) 时频图。
    """

    def __init__(self, config):
        super(CNN_2D_SpatioTemporal, self).__init__()

        cnn_2d_cfg = config.CNN_2D_CONFIG if hasattr(config, 'CNN_2D_CONFIG') else {
            'conv_channels': [32, 64, 128],
            'fc_hidden': [256],
            'dropout': 0.7,
            'num_classes': 2
        }

        self.conv_layers = nn.ModuleList()
        in_channels = 1  # 我们的 (1, F, T) 图像是单通道的

        for out_channels in cnn_2d_cfg['conv_channels']:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2) # [!! 已修复 !!] 同时压缩时间和空间
                )
            )
            in_channels = out_channels

        # ==========================================================
        # [!! 已修复 !!]
        # 动态计算展平后的大小 (时空模式)
        # ==========================================================
        if not (hasattr(config, 'WINDOW_SIZE') and hasattr(config, 'INPUT_FEATURES')):
            raise ValueError("Config 中缺少 WINDOW_SIZE 或 INPUT_FEATURES!")

        dummy_input = torch.randn(
            1, 1,
            config.WINDOW_SIZE,  # e.g., 20
            config.INPUT_FEATURES  # e.g., 101
        )
        # ==========================================================

        with torch.no_grad():
            dummy_output = self._forward_conv(dummy_input)

        self.flattened_size = dummy_output.view(1, -1).size(1)

        # 全连接层
        self.fc_layers = nn.ModuleList()
        fc_input_size = self.flattened_size

        for fc_hidden in cnn_2d_cfg['fc_hidden']:
            self.fc_layers.append(
                nn.Sequential(
                    nn.Linear(fc_input_size, fc_hidden),
                    nn.ReLU(),
                    nn.Dropout(cnn_2d_cfg['dropout'])
                )
            )
            fc_input_size = fc_hidden

        self.fc_out = nn.Linear(fc_input_size, cnn_2d_cfg['num_classes'])

    def _forward_conv(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def forward(self, x):
        """
        Args:
            x: shape (batch, 1, Freq_bins, Time_bins) - (N, 1, 33, 7)
        """

        # ==========================================================
        # [!! 已修复 !!]
        # 我们的数据加载器提供 (N, T, S) 形状的数据
        # T = window_size, S = 101
        # 我们必须在这里添加通道维度 (Channel) -> (N, 1, T, S)
        # ==========================================================
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # ==========================================================


        # 2. 通过 2D 卷积层
        x = self._forward_conv(x)

        # 3. 展平 (Flatten)
        x = x.view(x.size(0), -1)

        # 4. 通过全连接层
        for fc in self.fc_layers:
            x = fc(x)

        # 5. 输出
        output = self.fc_out(x)
        return output