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
from sklearn.utils.class_weight import compute_class_weight # [!! 修复 !!] 导入
from sklearn.metrics import precision_recall_fscore_support, f1_score, confusion_matrix # [!! 新增 !!]

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

        self.logger = TrainingLogger(
            log_dir=config.LOGS_DIR,
            model_name=f"{model_type}_{self.timestamp}",
            use_tensorboard=True
        )
        self.model = create_model(model_type, config)
        self.logger.log_model_summary(self.model)

        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def setup_training(self):
        """设置训练组件"""
        # 损失函数
        if self.model_type == 'lstm_ae':
            self.criterion = nn.MSELoss()  # 自编码器用MSE
            model_cfg = self.config.LSTM_AE_CONFIG
        else:
            # [!! 关键修改：论文级 TDR 调优 !!]
            self.logger.logger.info("计算类别权重 (高惩罚模式)...")
            try:
                # ==========================================================
                # [!! 关键修复 !!]
                # 根据模型类型, 加载 *正确的* 标签文件来计算权重
                # ==========================================================
                if self.model_type == 'cnn_2d':
                    self.logger.logger.info("...使用 (时频图) 'spectrograms_train.pkl' 的标签")
                    label_file_path = self.config.FEATURES_DIR / "sequences_train.pkl"
                    labels_key = 'y_spec'
                else:  # 默认 (lstm_cnn 等)
                    self.logger.logger.info("...使用 (时空序列) 'sequences_train.pkl' 的标签")
                    label_file_path = self.config.FEATURES_DIR / "sequences_train.pkl"
                    labels_key = 'y_seq'

                train_labels = pd.read_pickle(label_file_path)[labels_key]
                # ==========================================================

                # 1. 计算 "正常" 类的基础权重
                weight_0_array = compute_class_weight(
                    'balanced',
                    classes=np.unique(train_labels),
                    y=train_labels
                )
                weight_0 = weight_0_array[0]  # 获取 "正常" 类的权重

                # 2. [!! 在这里调优 !!]
                #    手动设置一个高的 '异常' 类惩罚权重
                #    这个值需要您来实验 (5.0, 10.0, 15.0 ...)
                #    越高的值 TDR 越高, Precision 越低
                high_penalty_weight_1 = 10.0  # <--- 在这里调整

                class_weights = torch.tensor(
                    [weight_0, high_penalty_weight_1],
                    dtype=torch.float32
                ).to(self.device)

                self.logger.logger.info(f"[!! TDR 调优 !!] 使用高惩罚加权损失, 权重: {class_weights}")
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)

            except FileNotFoundError:
                self.logger.logger.warning(f"文件 'sequences_train.pkl' 未找到。回退到默认权重。")
                weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            except Exception as e:
                self.logger.logger.error(f"动态计算权重失败: {e}。回退到默认权重。")
                weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            # [!! 修复结束 !!]

            # 动态获取模型配置
            if self.model_type == 'lstm_cnn':
                model_cfg = self.config.LSTM_CNN_CONFIG
            elif self.model_type == 'cnn_2d':
                model_cfg = self.config.CNN_2D_CONFIG
            elif self.model_type == 'cnn_1d':
                model_cfg = self.config.CNN_1D_CONFIG
            else:
                raise ValueError(f"未知的模型类型 {self.model_type} 无法找到配置")

        learning_rate = model_cfg['learning_rate']
        # ... (方法的其余部分保持不变) ...
        self.logger.logger.info(f"为 {self.model_type} 设置的学习率: {learning_rate}")

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # ... (方法的其余部分保持不变) ...
        if self.config.LR_SCHEDULER == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.LR_FACTOR,
                patience=self.config.LR_PATIENCE,
            )

        # 记录超参数
        hparams = {
            'model_type': self.model_type,
            'learning_rate': learning_rate,
            'batch_size': self.config.BATCH_SIZE,
            'num_epochs': self.config.NUM_EPOCHS,
            'window_size': self.config.WINDOW_SIZE,
            'device': str(self.config.DEVICE),
            'penalty_weight_1': high_penalty_weight_1 if 'high_penalty_weight_1' in locals() else 'N/A'  # 记录权重
        }
        self.logger.log_hyperparameters(hparams)

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """加载并准备数据"""
        self.logger.logger.info("加载数据...")

        try:
            # ==========================================================
            # [!! 关键修改 !!]
            # 根据模型类型, 加载不同的数据文件
            # ==========================================================
            if self.model_type in ['lstm_cnn', 'lstm_ae', 'cnn_2d']:
                if self.model_type == 'cnn_2d':
                    self.logger.logger.info("加载 (时空序列) 数据用于 2D-CNN (时空模式)...")
                else:
                    self.logger.logger.info(f"加载 (时空序列) 数据用于 {self.model_type.upper()}...")

                seq_train = pd.read_pickle(self.config.FEATURES_DIR / "sequences_train.pkl")
                seq_val = pd.read_pickle(self.config.FEATURES_DIR / "sequences_val.pkl")
                seq_test = pd.read_pickle(self.config.FEATURES_DIR / "sequences_test.pkl")

                X_train_np = seq_train['X_seq']
                y_train_np = seq_train['y_seq']
                X_val_np = seq_val['X_seq']
                y_val_np = seq_val['y_seq']
                X_test_np = seq_test['X_seq']
                y_test_np = seq_test['y_seq']

            else:
                self.logger.logger.error(f"模型类型 {self.model_type} 的数据加载逻辑未定义!")
                # (您可能需要为 cnn_1d 单独添加逻辑, 暂时回退)
                raise FileNotFoundError(f"未知模型 {self.model_type} 的数据加载路径")

            # ==========================================================
        except FileNotFoundError:
            self.logger.logger.error("序列文件未找到! 请确保在 'train' 模式前运行 'extract' 模式。")
            raise

        X_train = torch.FloatTensor(X_train_np)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        X_val = torch.FloatTensor(X_val_np)
        y_val = torch.tensor(y_val_np, dtype=torch.long)
        X_test = torch.FloatTensor(X_test_np)
        y_test = torch.tensor(y_test_np, dtype=torch.long)

        self.logger.logger.info(f"训练集: {X_train.shape}")
        self.logger.logger.info(f"验证集: {X_val.shape}")
        self.logger.logger.info(f"测试集: {X_test.shape}")

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=0)

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
            self.optimizer.zero_grad()

            if self.model_type == 'lstm_ae':
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_X)
            else:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

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
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr=current_lr)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
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

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.logger.info(f"\n早停触发 (patience={self.patience_counter})")
                break
        self.logger.logger.info("\n训练完成!")

    def _get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        辅助函数：获取模型的原始预测概率和真实标签。
        """
        self.model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)  # [N, 2]

                if self.model_type != 'lstm_ae':
                    probs = torch.softmax(outputs, dim=1)  # [N, 2]
                    all_labels.extend(batch_y.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                else:
                    # (此处可以添加 AE 的重建误差逻辑)
                    pass

        return np.array(all_labels), np.array(all_probs)

    def evaluate_on_test(self, test_loader: DataLoader, val_loader: DataLoader):
        """
        在测试集上评估。
        [!! 已重写 !!] 增加在验证集上搜索最佳阈值的步骤。
        """
        self.logger.logger.info("\n在测试集上评估...")
        model_path = self.config.CHECKPOINTS_DIR / f"{self.model_type}_best_{self.timestamp}.pth"

        # 1. 加载最佳模型
        if not model_path.exists():
            self.logger.logger.warning(f"未找到最佳模型: {model_path}。评估将使用当前模型状态。")
            epoch_num = 'N/A'
        else:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            epoch_num = checkpoint.get('epoch', 'N/A')
            self.logger.logger.info(f"已加载最佳模型 (Epoch {epoch_num})")

        if self.model_type == 'lstm_ae':
            self.logger.logger.warning("AE 模型的评估逻辑尚未实现。")
            return None

        # 2. [!! 关键 !!] 在验证集 (val_loader) 上搜索最佳阈值
        self.logger.logger.info("在验证集上搜索最佳阈值...")
        val_labels, val_probs = self._get_predictions(val_loader)
        val_probs_anomaly = val_probs[:, 1]  # 只获取“异常”类的概率

        # 2. [!! 关键 !!] 在验证集 (val_loader) 上搜索最佳阈值 (新TDR/FAR目标)
        self.logger.logger.info("在验证集上搜索最佳阈值 (新TDR/FAR目标)...")
        val_labels, val_probs = self._get_predictions(val_loader)
        val_probs_anomaly = val_probs[:, 1]  # 只获取“异常”类的概率

        # 搜索参数
        thresholds = np.arange(0.01, 1.0, 0.01)  # 阈值
        best_f1_at_goal = -1.0  # 达标下的最佳F1
        best_threshold_at_goal = -1.0  # 达标下的最佳阈值
        best_f1_overall = -1.0  # 纯F1最高
        best_threshold_overall = 0.5  # 纯F1最高的阈值 (用于回退)

        # 业务目标
        TDR_GOAL = self.config.TDR_THRESHOLD  # 0.8
        FAR_GOAL = self.config.FAR_THRESHOLD  # 0.1

        for th in thresholds:
            preds = (val_probs_anomaly > th).astype(int)

            # 计算 F1
            f1 = f1_score(val_labels, preds, zero_division=0)

            # 计算 TDR 和 FAR
            cm = confusion_matrix(val_labels, preds)

            # [!! 健壮性修复 !!]
            # 处理cm.ravel()在某些极端阈值下 (例如预测全为0或全为1) 失败的问题
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            elif cm.shape == (1, 1):
                if np.unique(val_labels)[0] == 0:  # 真实值全为0, 预测也全为0
                    tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                else:  # 真实值全为1, 预测也全为1
                    tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
            else:
                # 预测全为0 或 预测全为1 (但标签是混合的)
                if np.unique(preds).shape[0] == 1:
                    if np.unique(preds)[0] == 0:  # 预测全为 0
                        tn = np.sum(val_labels == 0)
                        fn = np.sum(val_labels == 1)
                        fp, tp = 0, 0
                    else:  # 预测全为 1
                        fp = np.sum(val_labels == 0)
                        tp = np.sum(val_labels == 1)
                        tn, fn = 0, 0
                else:
                    # 出现其他未预料的cm形状, 跳过此阈值
                    continue

            tdr = tp / (tp + fn) if (tp + fn) > 0 else 0
            far = fp / (fp + tn) if (fp + tn) > 0 else 0

            # --- [!! 关键的自动化逻辑 !!] ---
            # 1. 首先，我们只寻找“达标”的阈值
            if tdr >= TDR_GOAL and far <= FAR_GOAL:
                # 2. 在所有“达标”的阈值中，选择 F1 最高的那个
                if f1 > best_f1_at_goal:
                    best_f1_at_goal = f1
                    best_threshold_at_goal = th

            # (同时，我们仍然记录 F1 最高的阈值，作为回退)
            if f1 > best_f1_overall:
                best_f1_overall = f1
                best_threshold_overall = th

        # --- 决定使用哪个阈值 ---
        if best_threshold_at_goal != -1.0:
            # 找到了完美的阈值！
            best_threshold = best_threshold_at_goal
            self.logger.logger.info(
                f"✓ [自动化] 找到了满足 TDR(>={TDR_GOAL})/FAR(<={FAR_GOAL}) 目标的最佳阈值 (基于验证集): {best_threshold:.2f}")
        else:
            # 没找到，回退到 F1 最高的那个
            best_threshold = best_threshold_overall
            self.logger.logger.warning(f"✗ [自动化] 未找到能同时满足 TDR/FAR 的阈值。")
            self.logger.logger.warning(f"  回退到 F1 最高的阈值: {best_threshold:.2f} (TDR/FAR可能不达标)")

        self.logger.logger.info(f"✓ [自动化] 最佳阈值 (最终使用): {best_threshold:.2f}")
        self.logger.log_hyperparameters({'best_threshold': best_threshold})

        # 3. [!! 关键 !!] 在测试集 (test_loader) 上使用最佳阈值进行评估
        self.logger.logger.info(f"在测试集上使用阈值 {best_threshold:.2f} 进行最终评估...")
        test_labels, test_probs = self._get_predictions(test_loader)

        # 使用新阈值生成最终预测
        test_preds = (test_probs[:, 1] > best_threshold).astype(int)

        # 4. 计算和报告指标
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(
            test_labels,
            test_preds,  # 使用基于新阈值的预测
            test_probs,  # 仍然传递原始概率用于 ROC-AUC
            self.model_type
        )

        self.logger.logger.info(f"--- 测试集结果 (阈值 = {best_threshold:.2f}) ---")
        evaluator.print_metrics(metrics, self.model_type)
        self.logger.log_metrics(metrics, prefix='test/')
        self.logger.log_confusion_matrix(
            metrics['confusion_matrix'],
            ['Normal', 'Anomaly'],
            epoch=epoch_num
        )
        results_path = self.config.RESULTS_DIR / f"{self.model_type}_test_results.csv"
        evaluator.save_results(results_path)
        return metrics

    def run(self):
        """运行完整训练流程"""
        metrics = None
        try:
            self.setup_training()
            train_loader, val_loader, test_loader = self.load_data()
            self.train(train_loader, val_loader)

            # [!! 关键修改 !!] 将 val_loader 传递给评估方法
            metrics = self.evaluate_on_test(test_loader, val_loader)

        except Exception as e:
            self.logger.logger.error(f"训练器 'run' 方法中发生错误: {e}")
            import traceback
            self.logger.logger.error(traceback.format_exc())
        finally:
            self.logger.close()
        return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description='训练深度学习模型')
    parser.add_argument('--model', type=str, default='lstm_cnn',
                       choices=['lstm_cnn', 'lstm_ae', 'cnn_1d', 'cnn_2d'],
                       help='模型类型')
    args = parser.parse_args()
    config = Config()
    print("\n" + "="*80)
    print(f"训练模型: {args.model.upper()}")
    print("="*80)
    trainer = DeepModelTrainer(args.model, config)
    trainer.run()
    print("\n✓ 训练完成!")
    print(f"查看日志: {config.LOGS_DIR}")
    print(f"查看模型: {config.CHECKPOINTS_DIR}")
    print(f"查看结果: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()