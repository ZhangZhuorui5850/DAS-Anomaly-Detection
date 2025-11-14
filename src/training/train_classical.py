"""
经典机器学习模型训练脚本 - src/training/train_classical.py
支持: SVM, Random Forest, XGBoost, GMM
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple

from src.utils.config import Config
from src.utils.logger import Logger
from src.models.classical.classical_models import create_classical_model
from src.evaluation.metrics import ModelEvaluator, CrossValidationEvaluator
from sklearn.model_selection import cross_val_score


class ClassicalModelTrainer:
    """经典机器学习模型训练器"""

    def __init__(self, model_type: str, config, timestamp: str = None):
        """
        Args:
            model_type: 'svm', 'random_forest', 'xgboost', 'gmm'
            config: 配置对象
        """
        self.model_type = model_type
        self.config = config
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 初始化logger
        self.logger = Logger(
            log_dir=config.LOGS_DIR / f"{model_type}_{self.timestamp}",
            name=model_type
        )

        # 创建模型
        self.model = create_classical_model(model_type, config)
        self.logger.info(f"创建模型: {model_type.upper()}")

    def load_data(self) -> Tuple:
        """加载特征数据"""
        self.logger.info("加载特征数据...")

        # 加载训练/验证/测试特征
        features_train = pd.read_pickle(self.config.FEATURES_DIR / "features_train.pkl")
        features_val = pd.read_pickle(self.config.FEATURES_DIR / "features_val.pkl")
        features_test = pd.read_pickle(self.config.FEATURES_DIR / "features_test.pkl")

        X_train = features_train['features'].values
        y_train = features_train['labels'].values.astype(int)
        X_val = features_val['features'].values
        y_val = features_val['labels'].values.astype(int)
        X_test = features_test['features'].values
        y_test = features_test['labels'].values.astype(int)

        self.logger.info(f"训练集: {X_train.shape}, 标签分布: {np.bincount(y_train)}")
        self.logger.info(f"验证集: {X_val.shape}, 标签分布: {np.bincount(y_val)}")
        self.logger.info(f"测试集: {X_test.shape}, 标签分布: {np.bincount(y_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train, y_train, X_val, y_val):
        """训练模型"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("开始训练...")
        self.logger.info("=" * 80)

        # 记录训练参数
        if self.model_type == 'svm':
            params = self.config.SVM_CONFIG
        elif self.model_type == 'random_forest':
            params = self.config.RF_CONFIG
        elif self.model_type == 'xgboost':
            params = self.config.XGB_CONFIG
        else:
            params = {}

        self.logger.info(f"\n模型参数:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

        # 训练
        import time
        start_time = time.time()

        self.model.train(X_train, y_train, X_val, y_val)

        training_time = time.time() - start_time
        self.logger.info(f"\n训练耗时: {training_time:.2f} 秒")

        # 保存模型
        model_path = self.config.CHECKPOINTS_DIR / f"{self.model_type}_{self.timestamp}.pkl"
        self.model.save(model_path)
        self.logger.info(f"模型已保存: {model_path}")

    def cross_validate(self, X, y):
        """交叉验证"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("执行交叉验证...")
        self.logger.info("=" * 80)

        cv_evaluator = CrossValidationEvaluator(n_folds=self.config.CV_FOLDS)

        # 执行CV
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=self.config.CV_FOLDS,
                              shuffle=True,
                              random_state=self.config.RANDOM_SEED)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            self.logger.info(f"\nFold {fold_idx}/{self.config.CV_FOLDS}")

            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            # 创建新模型实例
            fold_model = create_classical_model(self.model_type, self.config)
            fold_model.train(X_train_fold, y_train_fold)

            # 预测
            y_pred_fold = fold_model.predict(X_val_fold)

            # 记录结果
            cv_evaluator.add_fold_result(y_val_fold, y_pred_fold, fold_idx)

        # 获取汇总
        summary = cv_evaluator.get_summary()

        return summary

    def evaluate_on_test(self, X_test, y_test):
        """在测试集上评估"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("测试集评估...")
        self.logger.info("=" * 80)

        # 预测
        y_pred = self.model.predict(X_test)

        # 预测概率
        y_pred_proba = None
        if hasattr(self.model.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_test)

        # 评估
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred, y_pred_proba, self.model_type)

        # 打印结果
        evaluator.print_metrics(metrics, self.model_type)

        # 保存结果
        results_path = self.config.RESULTS_DIR / f"{self.model_type}_test_results_{self.timestamp}.csv"
        evaluator.save_results(results_path)

        # 绘制混淆矩阵
        cm_path = self.config.RESULTS_DIR / f"{self.model_type}_confusion_matrix_{self.timestamp}.png"
        evaluator.plot_confusion_matrix(y_test, y_pred, self.model_type, save_path=cm_path)

        # 绘制ROC曲线
        if y_pred_proba is not None:
            roc_path = self.config.RESULTS_DIR / f"{self.model_type}_roc_curve_{self.timestamp}.png"
            evaluator.plot_roc_curve(y_test, y_pred_proba, self.model_type, save_path=roc_path)

        return metrics

    def run(self, with_cv: bool = False):
        """运行完整训练流程"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"训练 {self.model_type.upper()} 模型")
        self.logger.info("=" * 80)

        # 加载数据
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()

        # 训练
        self.train(X_train, y_train, X_val, y_val)

        # 交叉验证(可选)
        if with_cv:
            # 合并训练和验证集用于CV
            X_combined = np.vstack([X_train, X_val])
            y_combined = np.hstack([y_train, y_val])
            self.cross_validate(X_combined, y_combined)

        # 测试集评估
        metrics = self.evaluate_on_test(X_test, y_test)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("训练完成!")
        self.logger.info("=" * 80)
        self.logger.info(f"日志目录: {self.logger.log_dir}")

        return metrics


def train_all_models(config, with_cv: bool = False):
    """训练所有经典ML模型"""
    models = ['svm', 'random_forest', 'xgboost', 'gmm']

    all_results = {}

    for model_type in models:
        print(f"\n{'=' * 80}")
        print(f"训练 {model_type.upper()}")
        print(f"{'=' * 80}")

        try:
            trainer = ClassicalModelTrainer(model_type, config)
            metrics = trainer.run(with_cv=with_cv)
            all_results[model_type] = metrics
        except Exception as e:
            print(f"训练 {model_type} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 对比所有模型
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("模型性能对比")
        print(f"{'=' * 80}")

        evaluator = ModelEvaluator()
        evaluator.results = all_results

        comparison_df = evaluator.compare_models(
            save_path=config.RESULTS_DIR / f"models_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )

        # 保存对比结果
        comparison_df.to_csv(
            config.RESULTS_DIR / f"models_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='训练经典机器学习模型')
    parser.add_argument('--model', type=str, default='all',
                        choices=['svm', 'random_forest', 'xgboost', 'gmm', 'all'],
                        help='模型类型')
    parser.add_argument('--cv', action='store_true',
                        help='是否执行交叉验证')

    args = parser.parse_args()

    # 加载配置
    config = Config()

    if args.model == 'all':
        # 训练所有模型
        train_all_models(config, with_cv=args.cv)
    else:
        # 训练单个模型
        trainer = ClassicalModelTrainer(args.model, config)
        trainer.run(with_cv=args.cv)

    print("\n✓ 训练完成!")
    print(f"查看日志: {config.LOGS_DIR}")
    print(f"查看模型: {config.CHECKPOINTS_DIR}")
    print(f"查看结果: {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()