"""
经典机器学习模型 - src/models/classical/classical_models.py
包含: SVM, Random Forest, XGBoost, GMM
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
from pathlib import Path


class ClassicalMLModel:
    """经典机器学习模型基类"""

    def __init__(self, config, model_type='svm'):
        self.config = config
        self.model_type = model_type
        self.model = None
        self.best_params = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练模型"""
        raise NotImplementedError

    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练!")
        return self.model.predict(X)

    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练!")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("模型不支持概率预测")

    def save(self, filepath):
        """保存模型"""
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'best_params': self.best_params
        }, filepath)
        print(f"模型已保存: {filepath}")

    def load(self, filepath):
        """加载模型"""
        data = joblib.load(filepath)
        # (修复) GMM 的预测逻辑依赖 self.gmm_models, 而不是 self.model
        self.gmm_models = data['model']
        self.model = self.gmm_models # 保持 self.model 一致
        self.model_type = data['model_type']
        self.best_params = data.get('best_params')
        print(f"模型已加载: {filepath}")


class SVMModel(ClassicalMLModel):
    """支持向量机模型"""

    def __init__(self, config):
        super().__init__(config, 'svm')

    def train(self, X_train, y_train, X_val=None, y_val=None,
              grid_search=False):
        """
        训练SVM模型

        Args:
            grid_search: 是否进行网格搜索
        """
        print("=" * 60)
        print("训练 SVM 模型")
        print("=" * 60)

        if grid_search:
            print("\n进行网格搜索...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }

            svm = SVC(
                class_weight='balanced',
                probability=True,
                random_state=self.config.RANDOM_SEED
            )

            grid = GridSearchCV(
                svm, param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_train, y_train)

            self.model = grid.best_estimator_
            self.best_params = grid.best_params_

            print(f"\n最佳参数: {self.best_params}")
            print(f"最佳CV F1-score: {grid.best_score_:.4f}")

        else:
            # 使用配置文件中的参数
            self.model = SVC(**self.config.SVM_CONFIG)
            self.model.fit(X_train, y_train)
            print("\n使用预设参数训练完成")

        # 验证集评估
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            print(f"验证集准确率: {val_score:.4f}")

        return self


class RandomForestModel(ClassicalMLModel):
    """随机森林模型"""

    def __init__(self, config):
        super().__init__(config, 'random_forest')

    def train(self, X_train, y_train, X_val=None, y_val=None,
              grid_search=False):
        """训练随机森林模型"""
        print("=" * 60)
        print("训练 Random Forest 模型")
        print("=" * 60)

        if grid_search:
            print("\n进行网格搜索...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            rf = RandomForestClassifier(
                class_weight='balanced',
                random_state=self.config.RANDOM_SEED,
                n_jobs=-1
            )

            grid = GridSearchCV(
                rf, param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )

            grid.fit(X_train, y_train)

            self.model = grid.best_estimator_
            self.best_params = grid.best_params_

            print(f"\n最佳参数: {self.best_params}")
            print(f"最佳CV F1-score: {grid.best_score_:.4f}")

        else:
            self.model = RandomForestClassifier(**self.config.RF_CONFIG)
            self.model.fit(X_train, y_train)
            print("\n使用预设参数训练完成")

        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\n前10重要特征:")
        print(feature_importance.head(10))

        # 验证集评估
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            print(f"\n验证集准确率: {val_score:.4f}")

        return self


class XGBoostModel(ClassicalMLModel):
    """XGBoost模型"""

    def __init__(self, config):
        super().__init__(config, 'xgboost')

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练XGBoost模型"""
        print("=" * 60)
        print("训练 XGBoost 模型")
        print("=" * 60)

        # 计算类别权重
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"\n类别权重比例: {scale_pos_weight:.2f}")

        # 创建模型
        xgb_config = self.config.XGB_CONFIG.copy()
        xgb_config['scale_pos_weight'] = scale_pos_weight

        if X_val is not None and y_val is not None:
            # 添加到构造函数参数中 (兼容旧版xgboost)
            xgb_config['early_stopping_rounds'] = 20

        self.model = xgb.XGBClassifier(**xgb_config)

        # 训练
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                # early_stopping_rounds=20,
                verbose=True
            )
            print(f"\n最佳迭代: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train, verbose=True)

        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"\n前10重要特征:")
            print(feature_importance.head(10))

        return self


class GMMModel(ClassicalMLModel):
    """
    高斯混合模型
    为每个类别训练独立的GMM
    """

    def __init__(self, config):
        super().__init__(config, 'gmm')
        self.gmm_models = {}

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """训练GMM模型"""
        print("=" * 60)
        print("训练 GMM 模型")
        print("=" * 60)

        for label in np.unique(y_train):
            X_class = X_train[y_train == label]

            print(f"\n训练类别 {label} 的GMM...")
            print(f"样本数: {len(X_class)}")

            # 选择最优组件数(使用BIC)
            bic_scores = []
            n_components_range = range(1, 6)

            for n in n_components_range:
                gmm = GaussianMixture(
                    n_components=n,
                    random_state=self.config.RANDOM_SEED
                )
                gmm.fit(X_class)
                bic_scores.append(gmm.bic(X_class))

            best_n = n_components_range[np.argmin(bic_scores)]
            print(f"最优组件数: {best_n}")

            # 训练最优模型
            self.gmm_models[label] = GaussianMixture(
                n_components=best_n,
                random_state=self.config.RANDOM_SEED
            )
            self.gmm_models[label].fit(X_class)

        self.model = self.gmm_models  # 保持接口一致

        # 验证集评估
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            from sklearn.metrics import accuracy_score
            val_score = accuracy_score(y_val, y_pred)
            print(f"\n验证集准确率: {val_score:.4f}")

        return self

    def predict(self, X):
        """使用似然比进行预测"""
        if not self.gmm_models:
            raise ValueError("模型未训练!")

        # 计算每个类别的对数似然
        log_likelihoods = {}
        for label, gmm in self.gmm_models.items():
            log_likelihoods[label] = gmm.score_samples(X)

        # 选择最高似然的类别
        log_likelihood_matrix = np.array([log_likelihoods[label]
                                          for label in sorted(self.gmm_models.keys())])
        predictions = np.argmax(log_likelihood_matrix, axis=0)

        return predictions

    def predict_proba(self, X):
        """预测概率"""
        if not self.gmm_models:
            raise ValueError("模型未训练!")

        # 计算似然并转换为概率
        log_likelihoods = {}
        for label, gmm in self.gmm_models.items():
            log_likelihoods[label] = gmm.score_samples(X)

        log_likelihood_matrix = np.array([log_likelihoods[label]
                                          for label in sorted(self.gmm_models.keys())])

        # 使用softmax转换为概率
        exp_ll = np.exp(log_likelihood_matrix - np.max(log_likelihood_matrix, axis=0))
        probabilities = exp_ll / np.sum(exp_ll, axis=0)

        return probabilities.T


# 模型工厂
def create_classical_model(model_type, config):
    """
    创建经典ML模型

    Args:
        model_type: 'svm', 'random_forest', 'xgboost', 'gmm'
        config: 配置对象
    """
    model_dict = {
        'svm': SVMModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'gmm': GMMModel
    }

    if model_type not in model_dict:
        raise ValueError(f"Unknown model type: {model_type}")

    return model_dict[model_type](config)


# 测试代码
if __name__ == "__main__":
    from src.utils.config import Config
    import pickle

    # 加载特征数据
    feature_path = Config.FEATURES_DIR / "features_train.pkl"

    if feature_path.exists():
        data = pd.read_pickle(feature_path)
        X_train = data['features'].values[:1000]  # 测试用少量数据
        y_train = data['labels'].values[:1000]

        # 简单划分验证集
        split_idx = int(len(X_train) * 0.8)
        X_val = X_train[split_idx:]
        y_val = y_train[split_idx:]
        X_train = X_train[:split_idx]
        y_train = y_train[:split_idx]

        print(f"训练集: {X_train.shape}")
        print(f"验证集: {X_val.shape}")

        # 测试各个模型
        for model_type in ['svm', 'random_forest', 'xgboost', 'gmm']:
            print(f"\n{'=' * 60}")
            print(f"测试 {model_type.upper()} 模型")
            print(f"{'=' * 60}")

            model = create_classical_model(model_type, Config)
            model.train(X_train, y_train, X_val, y_val)

            # 预测
            y_pred = model.predict(X_val)
            from sklearn.metrics import accuracy_score, f1_score

            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)

            print(f"\n验证集性能:")
            print(f"准确率: {acc:.4f}")
            print(f"F1分数: {f1:.4f}")
    else:
        print(f"特征文件不存在: {feature_path}")
        print("请先运行 feature_extraction.py")