"""
深度学习模型训练脚本 - src/training/train_deep.py
支持: LSTM-CNN, LSTM-AE, 1D-CNN
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.utils.config import Config
from src.utils.logger import TrainingLogger
from src.models.deep_learning.lstm_cnn import create_model
from src.evaluation.metrics import ModelEvaluator


class DeepModelTrainer:
    """深度学习模型训练器"""

    def __init__(self, model_type: str, config, timestamp: str = None):
        """
        Args:
            model_type: 'lstm_cnn', 'lstm_ae', 'cnn_1d'
            config: 配置对象
        """
        self.model_type = model_type
        self.config = config
        self.device = config.DEVICE
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 初始化logger
        self.logger = TrainingLogger(
            log_dir=config.LOGS_DIR,
            model_name=f"{model_type}_{self.timestamp}",  # Logger 也用时间戳
            use_tensorboard=True
        )

        # 创建模型
        self.model = create_model(model_type, config)
        self.logger.log_model_summary(self.model)

        # 损失函数和优化器
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # 最佳模型追踪
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def setup_training(self):
        """设置训练组件"""
        # 损失函数
        if self.model_type == 'lstm_ae':
            self.criterion = nn.MSELoss()  # 自编码器用MSE
        else:
            self.criterion = nn.CrossEntropyLoss()  # 分类用CE

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # 学习率调度器
        if self.config.LR_SCHEDULER == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.LR_FACTOR,
                patience=self.config.LR_PATIENCE,
                #verbose=True
            )

        # 记录超参数
        hparams = {
            'model_type': self.model_type,
            'learning_rate': self.config.LEARNING_RATE,
            'batch_size': self.config.BATCH_SIZE,
            'num_epochs': self.config.NUM_EPOCHS,
            'window_size': self.config.WINDOW_SIZE,
            'device': str(self.config.DEVICE)
        }
        self.logger.log_hyperparameters(hparams)

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载并准备数据"""
        self.logger.logger.info("加载数据...")

        # 加载序列数据
        seq_train = pd.read_pickle(self.config.FEATURES_DIR / "sequences_train.pkl")
        seq_val = pd.read_pickle(self.config.FEATURES_DIR / "sequences_val.pkl")
        seq_test = pd.read_pickle(self.config.FEATURES_DIR / "sequences_test.pkl")

        # 转换为Tensor
        X_train = torch.FloatTensor(seq_train['X_seq'])
        #y_train = torch.LongTensor(seq_train['y_seq'])
        y_train = torch.tensor(seq_train['y_seq'], dtype=torch.long)

        X_val = torch.FloatTensor(seq_val['X_seq'])
        #y_val = torch.LongTensor(seq_val['y_seq'])
        y_val = torch.tensor(seq_val['y_seq'], dtype=torch.long)

        X_test = torch.FloatTensor(seq_test['X_seq'])
        #y_test = torch.LongTensor(seq_test['y_seq'])
        y_test = torch.tensor(seq_test['y_seq'], dtype=torch.long)

        self.logger.logger.info(f"训练集: {X_train.shape}")
        self.logger.logger.info(f"验证集: {X_val.shape}")
        self.logger.logger.info(f"测试集: {X_test.shape}")

        # 创建DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0  # Windows兼容性
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()

            if self.model_type == 'lstm_ae':
                # 自编码器: 输入=输出
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_X)
            else:
                # 分类模型
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

            # 反向传播
            loss.backward()

            # 梯度裁剪(防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # 统计
            total_loss += loss.item()

            if self.model_type != 'lstm_ae':
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                if self.model_type == 'lstm_ae':
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_X)
                else:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()

                if self.model_type != 'lstm_ae':
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """完整训练流程"""
        self.logger.logger.info("\n开始训练...")
        self.logger.logger.info("="*80)

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录日志
            self.logger.log_epoch(
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                lr=current_lr
            )

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0

                # 保存模型
                model_path = self.config.CHECKPOINTS_DIR / f"{self.model_type}_best_{self.timestamp}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, model_path)

                self.logger.logger.info(f"✓ 最佳模型已保存: {model_path}")
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.logger.info(f"\n早停触发 (patience={self.patience_counter})")
                break

        self.logger.logger.info("\n训练完成!")

    def evaluate_on_test(self, test_loader: DataLoader):
        """在测试集上评估"""
        self.logger.logger.info("\n在测试集上评估...")

        # 加载最佳模型
        model_path = self.config.CHECKPOINTS_DIR / f"{self.model_type}_best_{self.timestamp}.pth"
        if model_path.exists():
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.logger.info(f"已加载最佳模型 (Epoch {checkpoint['epoch']})")

        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)

                if self.model_type != 'lstm_ae':
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

        # 评估
        if self.model_type != 'lstm_ae':
            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate(
                np.array(all_labels),
                np.array(all_preds),
                np.array(all_probs),
                self.model_type
            )

            evaluator.print_metrics(metrics, self.model_type)

            # 记录到logger
            self.logger.log_metrics(metrics, prefix='test/')

            # 绘制混淆矩阵
            cm = metrics['confusion_matrix']
            self.logger.log_confusion_matrix(
                cm,
                ['Normal', 'Anomaly'],
                epoch=checkpoint.get('epoch', 0) if model_path.exists() else 0
            )

            # 保存评估结果
            results_path = self.config.RESULTS_DIR / f"{self.model_type}_test_results.csv"
            evaluator.save_results(results_path)

            return metrics  # <--- 1. 在这里返回 metrics

        return None  # <--- 2. 为其他情况（如AE）返回 None

    def run(self):
        """运行完整训练流程"""
        metrics = None  # <--- 3. 初始化 metrics 变量
        try:
            # 设置训练
            self.setup_training()

            # 加载数据
            train_loader, val_loader, test_loader = self.load_data()

            # 训练
            self.train(train_loader, val_loader)

            # 测试集评估
            metrics = self.evaluate_on_test(test_loader)

        finally:
            # 关闭logger
            self.logger.close()

        return metrics  # <--- 5. 最终返回 metrics


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='训练深度学习模型')
    parser.add_argument('--model', type=str, default='lstm_cnn',
                       choices=['lstm_cnn', 'lstm_ae', 'cnn_1d'],
                       help='模型类型')

    args = parser.parse_args()

    # 加载配置
    config = Config()

    print("\n" + "="*80)
    print(f"训练模型: {args.model.upper()}")
    print("="*80)

    # 创建训练器并运行
    trainer = DeepModelTrainer(args.model, config)
    trainer.run()

    print("\n✓ 训练完成!")
    print(f"查看日志: {config.LOGS_DIR}")
    print(f"查看模型: {config.CHECKPOINTS_DIR}")
    print(f"查看结果: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()