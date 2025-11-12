"""
主程序入口 - main.py
DAS光纤感测仪异常事件侦测系统
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.utils.config import Config
from src.data.preprocessing import DASDataLoader, DASPreprocessor, split_dataset
from src.features.feature_extraction import FeatureExtractor, SequenceGenerator
from src.models.classical.classical_models import create_classical_model
from src.models.deep_learning.lstm_cnn import create_model


class DASAnomalyDetectionPipeline:
    """完整的DAS异常检测流程"""

    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_preprocessing(self):
        """数据预处理流程"""
        print("\n" + "=" * 80)
        print("步骤 1: 数据预处理")
        print("=" * 80)

        # 加载数据
        loader = DASDataLoader(self.config.RAW_DATA_DIR / self.config.DATA_FILE)
        df = loader.load_data()

        # 预处理
        preprocessor = DASPreprocessor(self.config)
        df_processed = preprocessor.preprocess_pipeline(
            df,
            fit=True,
            apply_denoising=True
        )

        # 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            df_processed, self.config, stratify=True
        )

        # 保存
        output_path = self.config.PROCESSED_DATA_DIR / "processed_data.pkl"
        pd.to_pickle({
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }, output_path)
        print(f"\n✓ 预处理完成! 数据已保存: {output_path}")

        # 保存归一化器
        preprocessor.save_scaler(self.config.PROCESSED_DATA_DIR / "scaler.pkl")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_feature_extraction(self):
        """特征提取流程"""
        print("\n" + "=" * 80)
        print("步骤 2: 特征提取")
        print("=" * 80)

        # 加载预处理数据
        data_path = self.config.PROCESSED_DATA_DIR / "processed_data.pkl"
        if not data_path.exists():
            print("预处理数据不存在,先运行预处理...")
            self.run_preprocessing()

        data = pd.read_pickle(data_path)

        # 提取特征(用于经典ML)
        print("\n提取特征用于经典机器学习...")
        extractor = FeatureExtractor(self.config)

        features_train = extractor.extract_features_batch(data['X_train'])
        features_val = extractor.extract_features_batch(data['X_val'])
        features_test = extractor.extract_features_batch(data['X_test'])

        # 保存特征
        for split, features, labels in [
            ('train', features_train, data['y_train']),
            ('val', features_val, data['y_val']),
            ('test', features_test, data['y_test'])
        ]:
            output_path = self.config.FEATURES_DIR / f"features_{split}.pkl"
            pd.to_pickle({'features': features, 'labels': labels}, output_path)
            print(f"✓ {split} 特征已保存: {output_path}")

        # 生成序列(用于深度学习)
        print("\n生成序列用于深度学习...")
        seq_gen = SequenceGenerator(
            window_size=self.config.WINDOW_SIZE,
            stride=self.config.WINDOW_STRIDE
        )

        for split, X, y in [
            ('train', data['X_train'].values, data['y_train'].values),
            ('val', data['X_val'].values, data['y_val'].values),
            ('test', data['X_test'].values, data['y_test'].values)
        ]:
            X_seq, y_seq = seq_gen.create_sequences(X, y)
            output_path = self.config.FEATURES_DIR / f"sequences_{split}.pkl"
            pd.to_pickle({'X_seq': X_seq, 'y_seq': y_seq}, output_path)
            print(f"✓ {split} 序列已保存: {output_path}")

        print("\n✓ 特征提取完成!")

    def train_classical_models(self, models=['xgboost']):
        """训练经典机器学习模型"""
        print("\n" + "=" * 80)
        print("步骤 3: 训练经典机器学习模型")
        print("=" * 80)

        # 加载特征
        features_train = pd.read_pickle(self.config.FEATURES_DIR / "features_train.pkl")
        features_val = pd.read_pickle(self.config.FEATURES_DIR / "features_val.pkl")

        X_train = features_train['features'].values
        y_train = features_train['labels'].values
        X_val = features_val['features'].values
        y_val = features_val['labels'].values

        results = {}

        for model_name in models:
            print(f"\n训练 {model_name.upper()} 模型...")

            # 创建并训练模型
            model = create_classical_model(model_name, self.config)
            model.train(X_train, y_train, X_val, y_val)

            # 保存模型
            model_path = self.config.CHECKPOINTS_DIR / f"{model_name}_{self.timestamp}.pkl"
            model.save(model_path)

            # 评估
            from sklearn.metrics import accuracy_score, f1_score, classification_report

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)

            results[model_name] = {'accuracy': acc, 'f1_score': f1}

            print(f"\n{model_name.upper()} 验证集性能:")
            print(f"准确率: {acc:.4f}")
            print(f"F1分数: {f1:.4f}")
            print(classification_report(y_val, y_pred, target_names=['Normal', 'Anomaly']))

        # 保存结果
        results_df = pd.DataFrame(results).T
        results_path = self.config.RESULTS_DIR / f"classical_results_{self.timestamp}.csv"
        results_df.to_csv(results_path)
        print(f"\n✓ 结果已保存: {results_path}")

        return results

    def train_deep_models(self, models=['lstm_cnn']):
        """训练深度学习模型"""
        print("\n" + "=" * 80)
        print("步骤 4: 训练深度学习模型")
        print("=" * 80)

        # (修复) 导入 DataLoader
        from torch.utils.data import TensorDataset, DataLoader

        # 加载序列数据
        seq_train = pd.read_pickle(self.config.FEATURES_DIR / "sequences_train.pkl")
        seq_val = pd.read_pickle(self.config.FEATURES_DIR / "sequences_val.pkl")

        X_train_tensor = torch.FloatTensor(seq_train['X_seq'])
        y_train_tensor = torch.LongTensor(seq_train['y_seq'])
        X_val_tensor = torch.FloatTensor(seq_val['X_seq'])
        y_val_tensor = torch.LongTensor(seq_val['y_seq'])

        print(f"训练集: {X_train_tensor.shape}")
        print(f"验证集: {X_val_tensor.shape}")

        # (修复) 创建 TensorDataset 和 DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )

        results = {}

        for model_name in models:
            print(f"\n训练 {model_name.upper()} 模型...")

            model = create_model(model_name, self.config)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)

            best_val_loss = float('inf')
            patience_counter = 0

            # (修复) 使用 config 中的 NUM_EPOCHS, 而不是硬编码 10
            for epoch in range(self.config.NUM_EPOCHS):
                model.train()
                train_loss_total = 0

                # (修复) 真正的批次训练
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.config.DEVICE), y_batch.to(self.config.DEVICE)

                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss_total += loss.item()

                # 验证
                model.eval()
                val_loss_total = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(self.config.DEVICE), y_val_batch.to(
                            self.config.DEVICE)

                        val_outputs = model(X_val_batch)
                        val_loss = criterion(val_outputs, y_val_batch)
                        val_loss_total += val_loss.item()

                        _, predicted = torch.max(val_outputs, 1)
                        total += y_val_batch.size(0)
                        correct += (predicted == y_val_batch).sum().item()

                avg_train_loss = train_loss_total / len(train_loader)
                avg_val_loss = val_loss_total / len(val_loader)
                val_acc = correct / total

                print(f"Epoch [{epoch + 1}/{self.config.NUM_EPOCHS}] "
                      f"Train Loss: {avg_train_loss:.4f} "
                      f"Val Loss: {avg_val_loss:.4f} "
                      f"Val Acc: {val_acc:.4f}")

                # (修复) 使用 config 中的早停
                if avg_val_loss < best_val_loss - self.config.EARLY_STOPPING_DELTA:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(),
                               self.config.CHECKPOINTS_DIR / f"{model_name}_best_{self.timestamp}.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                        print(f"早停触发 (Patience={self.config.EARLY_STOPPING_PATIENCE})")
                        break

            # (修复) 确保加载回最佳模型进行结果记录
            # 加载最佳模型
            if Path(self.config.CHECKPOINTS_DIR / f"{model_name}_best_{self.timestamp}.pth").exists():
                model.load_state_dict(
                    torch.load(self.config.CHECKPOINTS_DIR / f"{model_name}_best_{self.timestamp}.pth"))

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    X_val_batch, y_val_batch = X_val_batch.to(self.config.DEVICE), y_val_batch.to(self.config.DEVICE)
                    val_outputs = model(X_val_batch)
                    _, predicted = torch.max(val_outputs, 1)
                    total += y_val_batch.size(0)
                    correct += (predicted == y_val_batch).sum().item()

            best_val_acc = correct / total
            results[model_name] = {'val_accuracy': best_val_acc, 'best_val_loss': best_val_loss}
            print(f"\n✓ {model_name.upper()} 训练完成! 最佳验证准确率: {best_val_acc:.4f}")

        results_df = pd.DataFrame(results).T
        results_path = self.config.RESULTS_DIR / f"deep_results_{self.timestamp}.csv"
        results_df.to_csv(results_path)
        print(f"\n✓ 结果已保存: {results_path}")

        return results

    def evaluate_all_models(self):
        """评估所有训练好的模型"""
        print("\n" + "=" * 80)
        print("步骤 5: 模型评估")
        print("=" * 80)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, classification_report

        results = {}

        # -----------------------------------------------------------------
        # 1. 加载并评估经典ML模型 (XGBoost, etc.)
        # -----------------------------------------------------------------

        features_test = pd.read_pickle(self.config.FEATURES_DIR / "features_test.pkl")
        X_test_ml = features_test['features'].values
        y_test_ml = features_test['labels'].values

        print(f"经典ML测试集: {X_test_ml.shape}")
        print(f"标签分布 (ML): {np.bincount(y_test_ml.astype(int))}")

        # (修复) 使用 self.timestamp 精确查找本次运行保存的 .pkl 文件
        for model_file in self.config.CHECKPOINTS_DIR.glob(f"*_{self.timestamp}.pkl"):
            # (修复) 把 try...except 移到循环内部
            try:
                model_name = model_file.stem.replace(f"_{self.timestamp}", "")
                print(f"\n评估 {model_name.upper()} (Classic)...")

                model = create_classical_model(model_name, self.config)
                model.load(model_file)
                y_pred_ml = model.predict(X_test_ml)

                results[model_name] = {
                    'accuracy': accuracy_score(y_test_ml, y_pred_ml),
                    'precision': precision_score(y_test_ml, y_pred_ml),
                    'recall': recall_score(y_test_ml, y_pred_ml),
                    'f1_score': f1_score(y_test_ml, y_pred_ml)
                }

                print(f"准确率: {results[model_name]['accuracy']:.4f}")
                print(f"F1分数: {results[model_name]['f1_score']:.4f}")
                print("\n分类报告:")
                print(classification_report(y_test_ml, y_pred_ml,
                                            target_names=['Normal', 'Anomaly']))
            except Exception as e:
                print(f"!! 评估 {model_file.name} 失败: {e}")  # 打印失败信息并继续

        # -----------------------------------------------------------------
        # 2. 加载并评估深度学习模型 (LSTM_CNN, etc.)
        # -----------------------------------------------------------------
        try:
            seq_test = pd.read_pickle(self.config.FEATURES_DIR / "sequences_test.pkl")
            X_test_dl = torch.FloatTensor(seq_test['X_seq']).to(self.config.DEVICE)
            y_test_dl_numpy = seq_test['y_seq']  # 用于评估的Numpy标签

            print(f"\n深度学习测试集: {X_test_dl.shape}")

            # (修复) 使用 self.timestamp 精确查找本次运行保存的 .pth 文件
            for model_file in self.config.CHECKPOINTS_DIR.glob(f"*_best_{self.timestamp}.pth"):
                # 从 "lstm_cnn_best_..." 中提取 "lstm_cnn"
                model_name = model_file.stem.split('_best_')[0]
                print(f"\n评估 {model_name.upper()} (Deep)...")

                model = create_model(model_name, self.config)
                model.load_state_dict(torch.load(model_file, map_location=self.config.DEVICE))
                model.eval()

                with torch.no_grad():
                    outputs = model(X_test_dl)
                    _, predicted = torch.max(outputs, 1)
                    y_pred_dl = predicted.cpu().numpy()

                # 使用Numpy标签进行评估
                results[model_name] = {
                    'accuracy': accuracy_score(y_test_dl_numpy, y_pred_dl),
                    'precision': precision_score(y_test_dl_numpy, y_pred_dl),
                    'recall': recall_score(y_test_dl_numpy, y_pred_dl),
                    'f1_score': f1_score(y_test_dl_numpy, y_pred_dl)
                }

                print(f"准确率: {results[model_name]['accuracy']:.4f}")
                print(f"F1分数: {results[model_name]['f1_score']:.4f}")
                print("\n分类报告:")
                print(classification_report(y_test_dl_numpy, y_pred_dl,
                                            target_names=['Normal', 'Anomaly']))
        except Exception as e:
            print(f"评估深度学习模型失败: {e}")

        # -----------------------------------------------------------------
        # 3. 汇总所有结果
        # -----------------------------------------------------------------
        if results:
            results_df = pd.DataFrame(results).T
            results_path = self.config.RESULTS_DIR / f"evaluation_results_{self.timestamp}.csv"
            results_df.to_csv(results_path)
            print(f"\n✓ 评估结果已保存: {results_path}")

            # 打印对比表
            print("\n" + "=" * 80)
            print("模型性能对比")
            print("=" * 80)
            print(results_df.round(4))
        else:
            print("没有找到模型进行评估。")

        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DAS异常检测系统')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'extract', 'train', 'eval', 'all'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default='all',
                        help='模型类型 (svm/rf/xgboost/gmm/lstm_cnn/lstm_ae/cnn_1d/all)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径(可选)')

    args = parser.parse_args()

    # 加载配置
    config = Config()
    if args.config:
        config.load_config(args.config)

    config.print_config()

    # 创建pipeline
    pipeline = DASAnomalyDetectionPipeline(config)

    # 根据模式运行
    if args.mode == 'preprocess':
        pipeline.run_preprocessing()

    elif args.mode == 'extract':
        pipeline.run_feature_extraction()

    elif args.mode == 'train':
        if args.model == 'all':
            # 训练所有经典ML模型
            pipeline.train_classical_models(['svm', 'random_forest', 'xgboost', 'gmm'])
            # 训练深度学习模型
            pipeline.train_deep_models(['lstm_cnn', 'cnn_1d'])
        elif args.model in ['svm', 'random_forest', 'xgboost', 'gmm']:
            pipeline.train_classical_models([args.model])
        elif args.model in ['lstm_cnn', 'lstm_ae', 'cnn_1d']:
            pipeline.train_deep_models([args.model])
        else:
            print(f"未知模型: {args.model}")

    elif args.mode == 'eval':
        pipeline.evaluate_all_models()

    elif args.mode == 'all':
        # 完整流程
        pipeline.run_preprocessing()
        pipeline.run_feature_extraction()
        pipeline.train_classical_models(['svm', 'random_forest', 'xgboost', 'gmm'])
        pipeline.train_deep_models(['lstm_cnn'])
        pipeline.evaluate_all_models()

    print("\n" + "=" * 80)
    print("✓ 任务完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()