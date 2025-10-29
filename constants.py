# -*- coding: utf-8 -*-
"""
常量定义文件
统一管理项目中的常量，确保一致性
"""

# 类别标签定义 - 所有文件必须使用这个统一的定义
CLASS_NAMES = ['OK', 'NG']  # 索引0: OK(合格), 索引1: NG(不合格)
CLASS_NAMES_DISPLAY = ['OK (合格)', 'NG (不合格)']  # 用于显示的标签

# 类别到索引的映射
CLASS_TO_INDEX = {
    'OK': 0,
    'NG': 1,
    'qualified': 0,    # 兼容JSON标签中的qualified字段
    'unqualified': 1   # 兼容JSON标签中的unqualified字段
}

# 索引到类别的映射
INDEX_TO_CLASS = {v: k for k, v in CLASS_TO_INDEX.items()}

# 支持的图像格式
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# 模型相关常量
NUM_CLASSES = 2
IMAGE_SIZE = 224
DEFAULT_MODEL_PATH = 'best_resnet_model.pth'

# 数据路径
DATA_DIR = 'data'
UPLOADS_DIR = 'uploads'
STATIC_DIR = 'static'
TEMPLATES_DIR = 'templates'