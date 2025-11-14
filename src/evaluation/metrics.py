"""
评估指标模块 - src/evaluation/metrics.py
提供完整的模型评估功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, class_names=['Normal', 'Anomaly']):
        self.class_names = class_names
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_pred_proba: Optional[np.ndarray] = None,
                 model_name: str = "Model") -> Dict:
        """
        完整的模型评估

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率 (可选)
            model_name: 模型名称

        Returns:
            评估指标字典
        """
        metrics = {}

        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')

        # DAS特定指标: TDR 和 FAR
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['TDR'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Detection Rate
        metrics['FAR'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Rate

        # 特异性和灵敏度
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # AUC-ROC (需要概率)
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1:
                # 多类概率,取正类概率
                y_pred_proba = y_pred_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_pred_proba)

        # 混淆矩阵
        metrics['confusion_matrix'] = cm

        # 保存结果
        self.results[model_name] = metrics

        return metrics

    def print_metrics(self, metrics: Dict, model_name: str = "Model"):
        """打印评估指标"""
        print("\n" + "=" * 80)
        print(f"模型评估结果: {model_name}")
        print("=" * 80)

        # 基础指标
        print("\n基础分类指标:")
        print("-" * 80)
        print(f"准确率 (Accuracy):     {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision):    {metrics['precision']:.4f}")
        print(f"召回率 (Recall):       {metrics['recall']:.4f}")
        print(f"F1分数 (F1-Score):     {metrics['f1_score']:.4f}")

        # DAS特定指标
        print("\nDAS威胁检测指标:")
        print("-" * 80)
        print(f"真实检测率 (TDR):     {metrics['TDR']:.4f}  {'✓' if metrics['TDR'] >= 0.8 else '✗'} (目标 ≥ 0.80)")
        print(f"误报率 (FAR):         {metrics['FAR']:.4f}  {'✓' if metrics['FAR'] <= 0.1 else '✗'} (目标 ≤ 0.10)")
        print(f"特异性 (Specificity): {metrics['specificity']:.4f}")
        print(f"灵敏度 (Sensitivity): {metrics['sensitivity']:.4f}")

        # ROC-AUC
        if 'roc_auc' in metrics:
            print(f"\nROC-AUC:              {metrics['roc_auc']:.4f}")
            print(f"平均精度 (AP):        {metrics['avg_precision']:.4f}")

        # 混淆矩阵
        print("\n混淆矩阵:")
        print("-" * 80)
        cm = metrics['confusion_matrix']

        # 格式化输出
        print(f"{'':>12} | {'预测为Normal':>15} | {'预测为Anomaly':>15}")
        print("-" * 80)
        print(f"{'真实Normal':>12} | {cm[0, 0]:>15} | {cm[0, 1]:>15}")
        print(f"{'真实Anomaly':>12} | {cm[1, 0]:>15} | {cm[1, 1]:>15}")

        print("=" * 80)

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Model",
                              save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})

        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title(f'混淆矩阵 - {model_name}', fontsize=14, fontweight='bold')

        # 添加百分比
        total = np.sum(cm)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                         ha='center', va='center', fontsize=10, color='gray')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存: {save_path}")

        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str = "Model",
                       save_path: Optional[str] = None):
        """
        绘制ROC曲线
        """
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                 label='随机分类器')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        plt.title('ROC曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存: {save_path}")

        plt.show()

    def plot_precision_recall_curve(self, y_true: np.ndarray,
                                    y_pred_proba: np.ndarray,
                                    model_name: str = "Model",
                                    save_path: Optional[str] = None):
        """
        绘制Precision-Recall曲线
        """
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'{model_name} (AP = {avg_precision:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title('Precision-Recall曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线已保存: {save_path}")

        plt.show()

    def compare_models(self, save_path: Optional[str] = None):
        """
        对比多个模型的性能
        """
        if not self.results:
            print("没有可对比的模型结果!")
            return

        # 提取指标
        models = list(self.results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'TDR']

        # 创建对比DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            row = [model_name]
            row.extend([metrics.get(m, 0) for m in metrics_names])
            if 'roc_auc' in metrics:
                row.append(metrics['roc_auc'])
            else:
                row.append(None)
            comparison_data.append(row)

        columns = ['Model'] + [m.upper() for m in metrics_names] + ['ROC-AUC']
        df = pd.DataFrame(comparison_data, columns=columns)

        # 打印对比表
        print("\n" + "=" * 100)
        print("模型性能对比")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)

        # 可视化对比
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 柱状图对比
        df_plot = df.set_index('Model')
        df_plot[['ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE', 'TDR']].plot(
            kind='bar', ax=axes[0], rot=45
        )
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('模型性能对比', fontsize=14, fontweight='bold')
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.0])

        # 雷达图
        categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'TDR']
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(122, projection='polar')

        for idx, model_name in enumerate(models):
            values = [self.results[model_name].get(m, 0)
                      for m in ['accuracy', 'precision', 'recall', 'f1_score', 'TDR']]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('模型性能雷达图', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n对比图已保存: {save_path}")

        plt.show()

        return df

    def save_results(self, filepath: str):
        """保存评估结果到CSV"""
        if not self.results:
            print("没有可保存的结果!")
            return

        # 转换为DataFrame
        data = []
        for model_name, metrics in self.results.items():
            row = {'model': model_name}
            row.update({k: v for k, v in metrics.items()
                        if k != 'confusion_matrix'})
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"评估结果已保存: {filepath}")


# 交叉验证评估
class CrossValidationEvaluator:
    """交叉验证评估器"""

    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
        self.fold_results = []

    def add_fold_result(self, y_true: np.ndarray, y_pred: np.ndarray,
                        fold_idx: int):
        """添加一折的结果"""
        metrics = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        self.fold_results.append(metrics)

    def get_summary(self) -> Dict:
        """获取交叉验证汇总统计"""
        df = pd.DataFrame(self.fold_results)

        summary = {
            'mean_accuracy': df['accuracy'].mean(),
            'std_accuracy': df['accuracy'].std(),
            'mean_f1': df['f1_score'].mean(),
            'std_f1': df['f1_score'].std()
        }

        print("\n交叉验证结果:")
        print("-" * 60)
        print(f"平均准确率: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        print(f"平均F1分数: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")

        return summary


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true.copy()
    # 添加一些错误
    error_idx = np.random.choice(n_samples, size=100, replace=False)
    y_pred[error_idx] = 1 - y_pred[error_idx]

    y_pred_proba = np.random.rand(n_samples, 2)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    # 创建评估器
    evaluator = ModelEvaluator()

    # 评估模型
    metrics = evaluator.evaluate(y_true, y_pred, y_pred_proba, "XGBoost")
    evaluator.print_metrics(metrics, "XGBoost")

    # 绘图
    evaluator.plot_confusion_matrix(y_true, y_pred, "XGBoost")
    evaluator.plot_roc_curve(y_true, y_pred_proba, "XGBoost")
    evaluator.plot_precision_recall_curve(y_true, y_pred_proba, "XGBoost")

    # 添加另一个模型进行对比
    y_pred2 = y_true.copy()
    error_idx2 = np.random.choice(n_samples, size=80, replace=False)
    y_pred2[error_idx2] = 1 - y_pred2[error_idx2]
    y_pred_proba2 = np.random.rand(n_samples, 2)
    y_pred_proba2 = y_pred_proba2 / y_pred_proba2.sum(axis=1, keepdims=True)

    evaluator.evaluate(y_true, y_pred2, y_pred_proba2, "LSTM-CNN")

    # 模型对比
    comparison_df = evaluator.compare_models()

    # 保存结果
    evaluator.save_results("evaluation_results.csv")