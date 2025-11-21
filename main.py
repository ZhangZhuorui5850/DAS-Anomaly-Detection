"""
主程序入口 - main.py
DAS光纤感测仪异常事件侦测系统
"""
from sklearn.preprocessing import StandardScaler
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
from src.data_process.preprocessing import DASDataLoader, DASPreprocessor, temporal_split_dataset
from src.features.feature_extraction import FeatureExtractor, SequenceGenerator
from src.models.classical.classical_models import create_classical_model
from src.models.deep_learning.deepmodels import create_model
from src.utils.logger import Logger, TrainingLogger
from src.training.train_classical import ClassicalModelTrainer
from src.training.train_deep import DeepModelTrainer
from src.evaluation.metrics import ModelEvaluator


class DASAnomalyDetectionPipeline:
    """完整的DAS异常检测流程"""

    def __init__(self, config):
        self.config = config
        self.logger = Logger(
            log_dir=config.LOGS_DIR,
            name="DAS_Pipeline"
        ).logger

        self.preprocessor = DASPreprocessor(self.config)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.all_model_metrics = {}

    def run_preprocessing(self):
        """
        执行数据预处理流程。
        [!! 已修复数据泄露: 流程重构为 Split-then-Process !!]
        """
        self.logger.info("步骤 1: 数据预处理 (已修复数据泄露)")

        # 1. 加载
        self.logger.info(f"从 {self.config.DATA_FILE} 加载数据...")
        loader = DASDataLoader(
            self.config.RAW_DATA_DIR / self.config.DATA_FILE
        )
        df_raw = loader.load_data()
        if df_raw is None:
            self.logger.error("加载数据失败。")
            return

        # 2. 先执行不依赖统计的步骤 (标签处理)
        self.logger.info("处理标签...")
        df_labeled = self.preprocessor.handle_labels(df_raw)

        # 3. !! 先划分 !! (Split)
        self.logger.info("按时间顺序划分数据集...")
        df_train, df_val, df_test = temporal_split_dataset(
            df_labeled,
            self.config
        )

        # 4. 处理训练集 (Fit=True)
        self.logger.info(f"处理训练集 (Fit=True, Method={self.config.DENOISE_METHOD})...")
        df_train_processed = self.preprocessor.preprocess_pipeline(
            df_train,
            fit=True
        )

        # 5. 处理验证集 (Fit=False)
        self.logger.info("处理验证集 (Fit=False)...")
        df_val_processed = self.preprocessor.preprocess_pipeline(
            df_val,
            fit=False
        )

        # 6. 处理测试集 (Fit=False)
        self.logger.info("处理测试集 (Fit=False)...")
        df_test_processed = self.preprocessor.preprocess_pipeline(
            df_test,
            fit=False
        )

        # 5. 保存处理后的数据 (分离 X 和 y)
        self.logger.info("分离 X, y 并保存处理后的数据...")
        output_path = self.config.PROCESSED_DATA_DIR / "processed_data.pkl"

        try:
            spatial_cols = [col for col in df_train_processed.columns if col.startswith('X')]

            data_to_save = {
                'X_train': df_train_processed[spatial_cols],
                'y_train': df_train_processed['label'],
                'X_val': df_val_processed[spatial_cols],
                'y_val': df_val_processed['label'],
                'X_test': df_test_processed[spatial_cols],
                'y_test': df_test_processed['label']
            }

            pd.to_pickle(data_to_save, output_path)
            self.logger.info(f"✓ 预处理完成! 数据已保存: {output_path}")

        except KeyError:
            self.logger.error("处理后的DataFrame中缺少 'label' 或 'X' 列。")
            return
        except Exception as e:
            self.logger.error(f"保存处理后数据失败: {e}")
            return

        # 6. 保存归一化器 (Scaler)
        self.preprocessor.save_scaler(
            self.config.PROCESSED_DATA_DIR / "scaler.pkl"
        )

    def run_feature_extraction(self):
        """特征提取流程"""
        print("\n" + "=" * 80)
        print("步骤 2: 特征提取")
        print("=" * 80)

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
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤 3: 训练经典机器学习模型")
        self.logger.info("=" * 80)

        for model_name in models:
            self.logger.info(f"\n--- 开始训练 {model_name.upper()} ---")
            try:
                trainer = ClassicalModelTrainer(
                    model_type=model_name,
                    config=self.config,
                    timestamp=self.timestamp
                )
                metrics = trainer.run(with_cv=False)
                if metrics:
                    self.all_model_metrics[model_name] = metrics
                self.logger.info(f"--- {model_name.upper()} 训练完成 ---")
            except Exception as e:
                self.logger.error(f"训练 {model_name} 失败: {e}")

    def train_deep_models(self, models=['lstm_cnn']):
        """训练深度学习模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤 4: 训练深度学习模型")
        self.logger.info("=" * 80)
        results = {}
        for model_name in models:
            self.logger.info(f"\n--- 开始训练 {model_name.upper()} ---")
            try:
                trainer = DeepModelTrainer(
                    model_type=model_name,
                    config=self.config,
                    timestamp=self.timestamp
                )
                metrics = trainer.run()
                if metrics:
                    self.all_model_metrics[model_name] = metrics
                self.logger.info(f"--- {model_name.upper()} 训练完成 ---")
            except Exception as e:
                self.logger.error(f"训练 {model_name} 失败: {e}")
        return results

    def compare_all_models(self):
        """
        步骤 5: 汇总所有模型结果,并生成对比图表
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤 5: 汇总模型对比")
        self.logger.info("=" * 80)

        if not self.all_model_metrics:
            self.logger.warning("没有可对比的模型指标。请先运行 'train' 模式。")
            return

        evaluator = ModelEvaluator()
        evaluator.results = self.all_model_metrics

        compare_path = self.config.RESULTS_DIR / f"comparison_summary_{self.timestamp}.png"
        csv_path = self.config.RESULTS_DIR / f"comparison_summary_{self.timestamp}.csv"

        comparison_df = evaluator.compare_models(save_path=compare_path)

        if comparison_df is not None:
            comparison_df.to_csv(csv_path, index=False)
            self.logger.info(f"✓ 完整对比表格已保存: {csv_path}")
            try:
                master_log_path = self.config.RESULTS_DIR / "master_results.csv"
                log_data = pd.DataFrame(self.all_model_metrics).T
                log_data = log_data.stack().to_frame().T
                log_data.index = [self.timestamp]
                log_data.columns = ['_'.join(col) for col in log_data.columns.values]

                if not master_log_path.exists():
                    log_data.to_csv(master_log_path, index=True, index_label='timestamp', mode='w', header=True)
                    self.logger.info(f"✓ 已创建主成绩单: {master_log_path}")
                else:
                    log_data.to_csv(master_log_path, index=True, mode='a', header=False)
                    self.logger.info(f"✓ 已追加到主成绩单: {master_log_path}")
            except Exception as e:
                self.logger.error(f"追加到主成绩单失败: {e}")

        self.logger.info("\n✓ 模型对比完成!")
        return comparison_df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DAS异常检测系统')

    # [!! 修复 !!] 从 .add.argument 改为 .add_argument (使用下划线)
    parser.add_argument('--mode', type=str, required=True,
                        choices=['preprocess', 'extract', 'train', 'eval', 'all'],
                        help='运行模式')
    parser.add_argument('--model', type=str, default='all',
                        help='模型类型 (svm/rf/xgboost/gmm/lstm_cnn/all)')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径(可选)')

    args = parser.parse_args()

    config = Config()
    if args.config:
        config.load_config(args.config)

    config.print_config()
    pipeline = DASAnomalyDetectionPipeline(config)

    if args.mode == 'preprocess':
        pipeline.run_preprocessing()
    elif args.mode == 'extract':
        pipeline.run_feature_extraction()
    elif args.mode == 'train':
        if args.model == 'all':
            pipeline.train_classical_models(['svm', 'random_forest', 'xgboost', 'gmm'])
            pipeline.train_deep_models(['lstm_cnn', 'cnn_2d'])
            pipeline.compare_all_models()
        elif args.model in ['svm', 'random_forest', 'xgboost', 'gmm']:
            pipeline.train_classical_models([args.model])
        elif args.model in ['lstm_cnn', 'lstm_ae', 'cnn_1d', 'cnn_2d']:
            pipeline.train_deep_models([args.model])
        else:
            print(f"未知模型: {args.model}")
    elif args.mode == 'eval':
        pipeline.logger.info("'eval' 模式现在执行 'compare_all_models'...")
        pipeline.logger.info("注意: 这只会对比在本次运行中刚刚训练过的模型。")
        pipeline.compare_all_models()
    elif args.mode == 'all':
        pipeline.run_preprocessing()
        pipeline.run_feature_extraction()
        pipeline.train_classical_models(['svm', 'random_forest', 'xgboost', 'gmm'])
        pipeline.train_deep_models(['lstm_cnn', 'cnn_2d'])
        pipeline.compare_all_models()

    print("\n" + "=" * 80)
    print("✓ 任务完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()