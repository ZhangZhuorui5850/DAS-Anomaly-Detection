SETUP_PY = """
from setuptools import setup, find_packages

setup(
    name='das_anomaly_detection',
    version='0.1.0',
    description='DAS光纤感测仪异常事件侦测系统',
    author='Your Name',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'torch>=2.0.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
    ],
    python_requires='>=3.8',
)
"""