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
from src.utils.logger import Logger, TrainingLogger
from src.training.train_classical import ClassicalModelTrainer
from src.training.train_deep import DeepModelTrainer
from src.evaluation.metrics import ModelEvaluator


class DASAnomalyDetectionPipeline:
    """完整的DAS异常检测流程"""

    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = Logger(config.LOGS_DIR, name="DAS_Pipeline")

        # [新增] 用于存储所有模型评估结果的字典
        self.all_model_metrics = {}

    def run_preprocessing(self):
        """数据预处理流程"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤 1: 数据预处理")
        self.logger.info("=" * 80)

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
        self.logger.info(f"\n✓ 预处理完成! 数据已保存: {output_path}")

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
        """训练经典机器学习模型 (已重构)"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤 3: 训练经典机器学习模型")
        self.logger.info("=" * 80)

        # results = {} # [修改] 不再使用局部变量 results

        for model_name in models:
            self.logger.info(f"\n--- 开始训练 {model_name.upper()} ---")

            try:
                # 1. 创建专用的训练器实例
                # 注意: ClassicalModelTrainer 会创建自己的专用 logger
                trainer = ClassicalModelTrainer(
                    model_type=model_name,
                    config=self.config,
                    timestamp=self.timestamp  # 传递总时间戳
                )

                # 2. 运行完整的训练和评估流程
                # .run() 方法会自动处理: 加载数据、训练、(可选CV)、测试集评估
                # 它会返回测试集上的 metrics
                metrics = trainer.run(with_cv=False)

                #results[model_name] = metrics
                # [修改] 将结果存储到类的属性中
                if metrics:
                    self.all_model_metrics[model_name] = metrics

                self.logger.info(f"--- {model_name.upper()} 训练完成 ---")

            except Exception as e:
                self.logger.error(f"训练 {model_name} 失败: {e}")

        # ClassicalModelTrainer 已经自己保存了详细结果，
        # 这里 pipeline 只是收集一个总览。

        # (可选) 你可以把汇总结果也保存一下
        # if results:
        #     results_df = pd.DataFrame(results).T
        #     results_path = self.config.RESULTS_DIR / f"classical_summary_{self.timestamp}.csv"
        #     results_df.to_csv(results_path)
        #     self.logger.info(f"\n✓ 经典模型训练汇总已保存: {results_path}")
        #
        # return results

    def train_deep_models(self, models=['lstm_cnn']):
        """训练深度学习模型 (已重构)"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤 4: 训练深度学习模型")
        self.logger.info("=" * 80)

        results = {}

        for model_name in models:
            self.logger.info(f"\n--- 开始训练 {model_name.upper()} ---")

            try:
                # 1. 创建专用的训练器实例
                # 注意：你可能需要确保 DeepModelTrainer 的 logger
                # 和 pipeline 的 logger 不会冲突，或者让它使用 pipeline 的 logger
                trainer = DeepModelTrainer(
                    model_type=model_name,
                    config=self.config,
                    timestamp=self.timestamp  # 传递总时间戳
                )

                # 2. 运行完整的训练和评估流程
                # （run() 方法会处理加载数据、训练、验证、早停、测试）
                metrics = trainer.run()

                if metrics:
                    self.all_model_metrics[model_name] = metrics

                # 3. (可选) 从训练器获取结果
                # 你可能需要修改 DeepModelTrainer.run()
                # 让它返回一个 metrics 字典
                # results[model_name] = trainer.get_best_metrics()

                self.logger.info(f"--- {model_name.upper()} 训练完成 ---")

            except Exception as e:
                self.logger.error(f"训练 {model_name} 失败: {e}")

        return results


    def compare_all_models(self):
        """
        步骤 5: 汇总所有模型结果,并生成对比图表
        (不再重新加载和评估,只使用训练时保存的指标)
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("步骤 5: 汇总模型对比")
        self.logger.info("=" * 80)

        if not self.all_model_metrics:
            self.logger.warning("没有可对比的模型指标。请先运行 'train' 模式。")  # <--- 错误在这里
            return

        # 创建一个评估器实例, 仅用于对比
        evaluator = ModelEvaluator()

        # [关键] 直接将收集到的指标赋给评估器
        evaluator.results = self.all_model_metrics

        # 定义保存路径
        compare_path = self.config.RESULTS_DIR / f"comparison_summary_{self.timestamp}.png"
        csv_path = self.config.RESULTS_DIR / f"comparison_summary_{self.timestamp}.csv"

        # 生成对比图 (雷达图和柱状图)
        comparison_df = evaluator.compare_models(save_path=compare_path)

        if comparison_df is not None:
            # 保存对比的CSV
            comparison_df.to_csv(csv_path, index=False)
            self.logger.info(f"✓ 完整对比表格已保存: {csv_path}")

            # ⬇⬇⬇ [新功能：追加到主成绩单] ⬇⬇⬇
            try:
                master_log_path = self.config.RESULTS_DIR / "master_results.csv"

                # 1. 准备要追加的数据
                # (我们不存 comparison_df, 我们存原始的 metrics 字典更灵活)
                # (把 {'xgboost': {'f1': 0.9}, 'lstm': {'f1': 0.8}} 转换)

                # '压扁' 字典, 变成一行
                log_data = pd.DataFrame(self.all_model_metrics).T
                log_data = log_data.stack().to_frame().T

                # 设置索引为时间戳
                log_data.index = [self.timestamp]

                # 重新组织列名 (例如: ('xgboost', 'f1_score') -> 'xgboost_f1_score')
                log_data.columns = ['_'.join(col) for col in log_data.columns.values]

                if not master_log_path.exists():
                    # 如果文件不存在，创建新文件并写入表头
                    log_data.to_csv(master_log_path, index=True, index_label='timestamp', mode='w', header=True)
                    self.logger.info(f"✓ 已创建主成绩单: {master_log_path}")
                else:
                    # 如果文件已存在，追加数据 (不写表头)
                    log_data.to_csv(master_log_path, index=True, mode='a', header=False)
                    self.logger.info(f"✓ 已追加到主成绩单: {master_log_path}")
            except Exception as e:
                self.logger.error(f"追加到主成绩单失败: {e}")
            # ⬆⬆⬆ [新功能结束] ⬆⬆⬆

        self.logger.info("\n✓ 模型对比完成!")
        return comparison_df

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
            # [新增] 训练后立即进行对比
            pipeline.compare_all_models()
        elif args.model in ['svm', 'random_forest', 'xgboost', 'gmm']:
            pipeline.train_classical_models([args.model])
        elif args.model in ['lstm_cnn', 'lstm_ae', 'cnn_1d']:
            pipeline.train_deep_models([args.model])
        else:
            print(f"未知模型: {args.model}")


    elif args.mode == 'eval':
        # [修改] 'eval' 模式现在只负责对比
        # 注意: 这只会对比在 *同一次运行* 中已训练的模型
        # (如果想加载历史结果进行对比, 逻辑会更复杂)
        pipeline.logger.info("'eval' 模式现在执行 'compare_all_models'...")
        pipeline.logger.info("注意: 这只会对比在本次运行中刚刚训练过的模型。")
        pipeline.compare_all_models()

    elif args.mode == 'all':
        # 完整流程
        pipeline.run_preprocessing()
        pipeline.run_feature_extraction()
        pipeline.train_classical_models(['svm', 'random_forest', 'xgboost', 'gmm'])
        pipeline.train_deep_models(['lstm_cnn'])
        pipeline.compare_all_models()

    print("\n" + "=" * 80)
    print("✓ 任务完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()