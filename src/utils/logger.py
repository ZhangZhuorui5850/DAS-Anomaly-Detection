"""
日志记录模块 - src/utils/logger.py
用于记录训练过程、评估结果、错误信息等
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """统一的日志记录器"""

    def __init__(self, log_dir, name="DAS", level=logging.INFO):
        """
        初始化日志记录器

        Args:
            log_dir: 日志保存目录
            name: 日志器名称
            level: 日志级别
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 设置日志文件路径
        self.log_file = self.log_dir / f"{name}_{self.timestamp}.log"

        # 配置logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 清除已有的handlers
        self.logger.handlers.clear()

        # 文件handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(level)

        # 控制台handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.info(f"日志文件: {self.log_file}")

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


class TrainingLogger:
    """
    训练过程日志记录器
    包含TensorBoard可视化
    """

    def __init__(self, log_dir, model_name, use_tensorboard=True):
        """
        Args:
            log_dir: 日志目录
            model_name: 模型名称
            use_tensorboard: 是否使用TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建模型专用日志目录
        self.model_log_dir = self.log_dir / f"{model_name}_{self.timestamp}"
        self.model_log_dir.mkdir(parents=True, exist_ok=True)

        # 基础logger
        self.logger = Logger(self.model_log_dir, name=model_name).logger

        # TensorBoard writer
        self.writer = None
        if use_tensorboard:
            tensorboard_dir = self.model_log_dir / "tensorboard"
            tensorboard_dir.mkdir(exist_ok=True)
            self.writer = SummaryWriter(str(tensorboard_dir))
            self.logger.info(f"TensorBoard日志: {tensorboard_dir}")

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }

        # 最佳指标
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_epoch': 0
        }

        self.logger.info(f"开始训练 {model_name}")
        self.logger.info("=" * 80)

    def log_epoch(self, epoch, train_loss, train_acc, val_loss=None,
                  val_acc=None, lr=None, **kwargs):
        """
        记录每个epoch的训练信息

        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失
            val_acc: 验证准确率
            lr: 学习率
            **kwargs: 其他指标
        """
        # 保存历史
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        if lr is not None:
            self.history['learning_rate'].append(lr)

        # 日志输出
        log_msg = f"Epoch [{epoch}] "
        log_msg += f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"

        if val_loss is not None and val_acc is not None:
            log_msg += f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"

        if lr is not None:
            log_msg += f" | LR: {lr:.6f}"

        # 添加其他指标
        for key, value in kwargs.items():
            log_msg += f" | {key}: {value:.4f}"

        self.logger.info(log_msg)

        # TensorBoard记录
        if self.writer:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)

            if val_loss is not None:
                self.writer.add_scalar('Loss/val', val_loss, epoch)
            if val_acc is not None:
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            if lr is not None:
                self.writer.add_scalar('Learning_Rate', lr, epoch)

            # 记录其他指标
            for key, value in kwargs.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)

        # 更新最佳指标
        if val_loss is not None and val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_loss
            self.best_metrics['best_val_acc'] = val_acc if val_acc else 0.0
            self.best_metrics['best_epoch'] = epoch
            self.logger.info(f"✓ 新的最佳模型! Val Loss: {val_loss:.4f}")

    def log_metrics(self, metrics_dict, step=None, prefix=''):
        """
        记录评估指标

        Args:
            metrics_dict: 指标字典
            step: 步骤(可选)
            prefix: 前缀(如'test/', 'train/')
        """
        self.logger.info(f"\n{prefix}评估指标:")
        self.logger.info("-" * 60)

        for key, value in metrics_dict.items():
            self.logger.info(f"{key}: {value:.4f}")

            # TensorBoard记录
            if self.writer and step is not None:
                self.writer.add_scalar(f'{prefix}{key}', value, step)

    def log_confusion_matrix(self, cm, class_names, epoch=None):
        """
        记录混淆矩阵

        Args:
            cm: 混淆矩阵 (numpy array)
            class_names: 类别名称列表
            epoch: epoch数(可选)
        """
        self.logger.info("\n混淆矩阵:")
        self.logger.info("-" * 60)

        # 打印混淆矩阵
        header = "真实\\预测 | " + " | ".join([f"{name:>8}" for name in class_names])
        self.logger.info(header)
        self.logger.info("-" * len(header))

        for i, row in enumerate(cm):
            row_str = f"{class_names[i]:>10} | " + " | ".join([f"{val:>8}" for val in row])
            self.logger.info(row_str)

        # TensorBoard记录
        if self.writer and epoch is not None:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix')

            self.writer.add_figure('Confusion_Matrix', fig, epoch)
            plt.close(fig)

    def log_model_summary(self, model):
        """
        记录模型摘要

        Args:
            model: PyTorch模型或sklearn模型
        """
        self.logger.info("\n模型摘要:")
        self.logger.info("=" * 80)

        if isinstance(model, torch.nn.Module):
            # PyTorch模型
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.logger.info(f"模型类型: {model.__class__.__name__}")
            self.logger.info(f"总参数量: {total_params:,}")
            self.logger.info(f"可训练参数: {trainable_params:,}")
            self.logger.info(f"不可训练参数: {total_params - trainable_params:,}")

            # 打印模型结构
            self.logger.info("\n模型结构:")
            self.logger.info(str(model))
        else:
            # sklearn模型
            self.logger.info(f"模型类型: {type(model).__name__}")
            if hasattr(model, 'get_params'):
                params = model.get_params()
                self.logger.info(f"模型参数: {params}")

    def log_hyperparameters(self, hparams):
        """
        记录超参数

        Args:
            hparams: 超参数字典
        """
        self.logger.info("\n超参数配置:")
        self.logger.info("-" * 60)

        for key, value in hparams.items():
            self.logger.info(f"{key}: {value}")

        # TensorBoard记录
        if self.writer:
            self.writer.add_hparams(
                hparams,
                {'hparam/accuracy': 0}  # 占位符
            )

    def save_history(self):
        """保存训练历史到JSON文件"""
        history_file = self.model_log_dir / "training_history.json"

        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=4)

        self.logger.info(f"\n训练历史已保存: {history_file}")

    def save_best_metrics(self):
        """保存最佳指标"""
        metrics_file = self.model_log_dir / "best_metrics.json"

        with open(metrics_file, 'w') as f:
            json.dump(self.best_metrics, f, indent=4)

        self.logger.info(f"最佳指标已保存: {metrics_file}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        import matplotlib.pyplot as plt

        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 损失曲线
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        if self.history['val_loss']:
            axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 准确率曲线
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        if self.history['val_acc']:
            axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        fig_path = self.model_log_dir / "training_curves.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"训练曲线已保存: {fig_path}")

        # TensorBoard记录
        if self.writer:
            self.writer.add_figure('Training_Curves', fig)

    def close(self):
        """关闭logger"""
        self.logger.info("\n训练完成!")
        self.logger.info("=" * 80)
        self.logger.info(f"最佳模型 - Epoch: {self.best_metrics['best_epoch']}, "
                         f"Val Loss: {self.best_metrics['best_val_loss']:.4f}, "
                         f"Val Acc: {self.best_metrics['best_val_acc']:.4f}")

        # 保存历史和指标
        self.save_history()
        self.save_best_metrics()

        # 绘制训练曲线
        if len(self.history['train_loss']) > 0:
            self.plot_training_curves()

        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
            self.logger.info(f"\nTensorBoard命令: tensorboard --logdir {self.model_log_dir / 'tensorboard'}")


# 使用示例
if __name__ == "__main__":
    # 1. 基础logger使用
    logger = Logger("logs", name="test")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")

    # 2. 训练logger使用
    train_logger = TrainingLogger("logs", "test_model", use_tensorboard=True)

    # 记录超参数
    train_logger.log_hyperparameters({
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 100
    })

    # 模拟训练过程
    for epoch in range(1, 11):
        train_loss = 1.0 / epoch
        train_acc = 0.5 + 0.04 * epoch
        val_loss = 1.2 / epoch
        val_acc = 0.45 + 0.04 * epoch

        train_logger.log_epoch(
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            lr=0.001 * (0.9 ** epoch)
        )

    # 记录混淆矩阵
    import numpy as np

    cm = np.array([[85, 15], [10, 90]])
    train_logger.log_confusion_matrix(cm, ['Normal', 'Anomaly'], epoch=10)

    # 记录评估指标
    metrics = {
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.88,
        'f1_score': 0.89
    }
    train_logger.log_metrics(metrics, step=10, prefix='test/')

    # 关闭logger
    train_logger.close()

    print("\n✓ 日志示例运行完成!")
    print(f"查看日志文件: logs/test_model_*/")
    print(f"启动TensorBoard: tensorboard --logdir logs/")