# DAS光纤感测仪异常事件侦测系统

## 📁 项目结构
```
das_anomaly_detection/
├── data/
│   ├── raw/              # 放置原始CSV数据
│   ├── processed/        # 预处理后的数据
│   └── features/         # 提取的特征
├── src/
│   ├── utils/
│   │   └── config.py          # ✅ 配置文件
│   ├── data/
│   │   └── preprocessing.py   # ✅ 数据预处理
│   ├── features/
│   │   └── feature_extraction.py  # ✅ 特征提取
│   └── models/
│       ├── classical/
│       │   └── classical_models.py  # ✅ SVM/RF/XGBoost/GMM
│       └── deep_learning/
│           └── lstm_cnn.py    # ✅ LSTM-CNN/LSTM-AE/1D-CNN
├── checkpoints/          # 模型保存
├── logs/                 # 训练日志
├── results/              # 结果输出
├── main.py              # ✅ 主程序
└── requirements.txt      # 依赖包
```

## 🚀 快速开始

### 1. 环境配置
```bash
# 创建虚拟环境
conda create -n das python=3.9
conda activate das

# 安装依赖
pip install numpy pandas scipy scikit-learn xgboost
pip install torch torchvision  # 或从官网安装适合你CUDA版本的PyTorch
pip install matplotlib seaborn tqdm pyyaml joblib
```

### 2. 数据准备
```bash
# 将示例数据.csv放入data/raw/目录
cp 示例数据.csv data_preprocess/raw/
```

### 3. 运行完整流程
```bash
# 方式1: 一键运行(包含预处理、特征提取、训练、评估)
python main.py --mode all

# 方式2: 分步运行
python main.py --mode preprocess    # 数据预处理
python main.py --mode extract       # 特征提取
python main.py --mode train --model all  # 训练所有模型
python main.py --mode eval          # 模型评估
```

### 4. 训练特定模型
```bash
# 训练XGBoost(推荐,速度快)
python main.py --mode train --model xgboost

# 训练LSTM-CNN(效果最好)
python main.py --mode train --model lstm_cnn

# 训练所有经典ML模型
python main.py --mode train --model svm
python main.py --mode train --model random_forest
python main.py --mode train --model gmm
```

## 📊 预期性能(基于文献)

| 模型 | F1-Score | TDR | FAR | 训练速度 |
|------|----------|-----|-----|---------|
| **LSTM-CNN** | **85-93%** | **≥80%** | **≤10%** | 慢 |
| XGBoost | 80-88% | 75-85% | 10-15% | 快 |
| Random Forest | 75-85% | 70-80% | 15-20% | 中 |
| SVM | 75-85% | 70-80% | 15-20% | 中 |
| GMM | 70-80% | 65-75% | 20-25% | 快 |

## 🔧 关键配置修改

编辑 `src/utils/config.py`:
```python
# 数据配置
RANDOM_SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2

# 归一化方法(重要!)
NORMALIZATION_METHOD = 'high_freq_energy'  # 应对距离衰减

# 时间窗口
WINDOW_SIZE = 5  # 滑动窗口大小

# 深度学习训练
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
```

## 📈 查看结果
```bash
# 结果保存位置
results/
├── classical_results_YYYYMMDD_HHMMSS.csv
├── deep_results_YYYYMMDD_HHMMSS.csv
└── evaluation_results_YYYYMMDD_HHMMSS.csv

# 模型保存位置
checkpoints/
├── xgboost_YYYYMMDD_HHMMSS.pkl
├── lstm_cnn_best_YYYYMMDD_HHMMSS.pth
└── ...
```

## 🎯 模块使用说明

### 单独使用数据预处理

```python
from src.utils.config import Config
from src.data_preprocess.preprocessing import DASDataLoader, DASPreprocessor

# 加载数据
loader = DASDataLoader(Config.RAW_DATA_DIR / "示例数据.csv")
df = loader.load_data()

# 预处理
preprocessor = DASPreprocessor(Config)
df_clean = preprocessor.preprocess_pipeline(df, fit=True)
```

### 单独使用特征提取
```python
from src.features.feature_extraction import FeatureExtractor

extractor = FeatureExtractor(Config)
features = extractor.extract_features_batch(X_data)
```

### 单独训练模型
```python
from src.models.classical.classical_models import create_classical_model

# 训练XGBoost
model = create_classical_model('xgboost', Config)
model.train(X_train, y_train, X_val, y_val)
model.save('checkpoints/my_model.pkl')
```

## 🔍 关键技术要点

### 1. 数据预处理
- **缺失值处理**: 空间插值
- **归一化**: 高频能量归一化(应对距离衰减)
- **去噪**: 频谱减法(可选)

### 2. 特征工程
- **时域**: 能量、峰值、过零率、偏度、峰度
- **频域**: 8个频带能量、谱质心、谱熵
- **空间**: 空间梯度、能量质心、峰值位置

### 3. 模型选择
- **经典ML**: 需要手动特征工程,可解释性强
- **深度学习**: 端到端学习,性能最优

### 4. 评估指标
- **主要**: F1-Score, TDR(真实检测率), FAR(误报率)
- **次要**: 准确率、精确率、召回率、AUC-ROC

## ⚠️ 常见问题

### 1. 内存不足
```python
# 修改批大小
Config.BATCH_SIZE = 16  # 从32降到16
```

### 2. CUDA错误
```python
# 使用CPU
Config.DEVICE = torch.device('cpu')
```

### 3. 数据不平衡
```python
# 已在模型中处理(class_weight='balanced')
# XGBoost自动计算scale_pos_weight
```

### 4. 训练太慢
```python
# 先用XGBoost快速验证
python main.py --mode train --model xgboost

# 再用深度学习优化
python main.py --mode train --model lstm_cnn
```

## 📚 参考文献

1. **Xu et al. (2018)** - "Pattern recognition based on time-frequency analysis and CNNs for vibrational events in φ-OTDR"
   - 贡献: CNN+频谱图方法, 准确率>90%

2. **Duraj et al. (2025)** - "Detection of Anomalies in Data Streams Using LSTM-CNN"
   - 贡献: LSTM-CNN混合架构, F1=88-93%

3. **Tejedor et al. (2016)** - "Towards Prevention of Pipeline Integrity Threats using Smart Fiber Optic"
   - 贡献: 高频能量归一化, GMM分类器, 真实管道数据

## 📧 联系方式

有问题请提Issue或发送邮件到: your_email@example.com

## 📄 License

MIT License