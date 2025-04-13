import kagglehub
import os

# 下载400万数独数据集

# 确保dataset目录存在
os.makedirs("dataset", exist_ok=True)

# 下载数据集并指定保存路径
path = kagglehub.dataset_download("informoney/4-million-sudoku-puzzles-easytohard", path="dataset")

print("Path to dataset files:", path)