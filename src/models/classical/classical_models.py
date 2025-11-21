"""
经典机器学习模型 - src/models/classical/classical_models.py
已简化：使用通用包装器处理 SVM/RF/XGB，保留 GMM 特有逻辑
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import xgboost as xgb

# ==========================================================
# 1. 通用模型包装器 (用于 SVM, RF, XGBoost)
# ==========================================================
class GenericModelWrapper:
    """
    通用的 sklearn/xgboost 模型包装器
    统一了 train/predict/save/load 接口
    """
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("=" * 60)
        print(f"训练 {self.model_type.upper()} 模型")
        print("=" * 60)

        # XGBoost 特殊处理：支持验证集监控
        if self.model_type == 'xgboost' and X_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)

        print("训练完成")

        # 简单的验证集评估
        if X_val is not None and y_val is not None:
            score = self.model.score(X_val, y_val)
            print(f"验证集准确率: {score:.4f}")

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, filepath):
        joblib.dump({'model': self.model, 'type': self.model_type}, filepath)
        print(f"模型已保存: {filepath}")

    def load(self, filepath):
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data.get('type', 'unknown')
        print(f"模型已加载: {filepath}")


# ==========================================================
# 2. GMM 模型 (保留原有的特殊逻辑)
# ==========================================================
class GMMModel:
    """高斯混合模型：为每个类别单独训练一个 GMM"""
    def __init__(self, config):
        self.config = config
        self.gmm_models = {}
        self.model_type = 'gmm'

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print(f"训练 GMM 模型 (Components 搜索范围: 1-5)")

        for label in np.unique(y_train):
            X_class = X_train[y_train == label]

            # 简单的 BIC 选择
            best_bic = np.inf
            best_gmm = None

            for n in range(1, 6):
                gmm = GaussianMixture(n_components=n, random_state=self.config.RANDOM_SEED)
                gmm.fit(X_class)
                bic = gmm.bic(X_class)
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm

            self.gmm_models[label] = best_gmm
            print(f"  类别 {label}: 最佳组件数 = {best_gmm.n_components}")

        if X_val is not None:
            y_pred = self.predict(X_val)
            acc = np.mean(y_pred == y_val)
            print(f"验证集准确率: {acc:.4f}")

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        # 计算每个类别的对数似然
        log_likelihoods = np.array([
            self.gmm_models[label].score_samples(X)
            for label in sorted(self.gmm_models.keys())
        ])
        # Softmax 归一化
        exp_ll = np.exp(log_likelihoods - np.max(log_likelihoods, axis=0))
        return (exp_ll / np.sum(exp_ll, axis=0)).T

    def save(self, filepath):
        joblib.dump(self.gmm_models, filepath)

    def load(self, filepath):
        self.gmm_models = joblib.load(filepath)


# ==========================================================
# 3. 工厂函数 (大大简化)
# ==========================================================
def create_classical_model(model_type, config):
    """统一创建入口"""
    if model_type == 'svm':
        model = SVC(**config.SVM_CONFIG)
        return GenericModelWrapper(model, 'svm')

    elif model_type == 'random_forest':
        model = RandomForestClassifier(**config.RF_CONFIG)
        return GenericModelWrapper(model, 'random_forest')

    elif model_type == 'xgboost':
        # 动态计算 scale_pos_weight (如果需要更精确可以在外部算好传入，这里简化处理)
        xgb_cfg = config.XGB_CONFIG.copy()
        model = xgb.XGBClassifier(**xgb_cfg)
        return GenericModelWrapper(model, 'xgboost')

    elif model_type == 'gmm':
        return GMMModel(config)

    else:
        raise ValueError(f"未知模型类型: {model_type}")