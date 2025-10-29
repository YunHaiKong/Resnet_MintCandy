# 训练配置文件

class Config:
    """训练配置类"""
    
    # 数据相关配置
    DATA_DIR = './data'                    # 数据目录
    TRAIN_RATIO = 0.8                      # 训练集比例
    
    # 模型相关配置
    MODEL_NAME = 'resnet18'                # 模型架构 (resnet18, resnet34, resnet50)
    NUM_CLASSES = 2                        # 分类数量
    PRETRAINED = True                      # 是否使用预训练权重
    
    # 训练相关配置
    BATCH_SIZE = 16                        # 批次大小
    NUM_EPOCHS = 30                        # 训练轮数
    LEARNING_RATE = 0.001                  # 学习率
    WEIGHT_DECAY = 1e-4                    # 权重衰减
    
    # 学习率调度器配置
    SCHEDULER_STEP_SIZE = 15               # 学习率衰减步长
    SCHEDULER_GAMMA = 0.1                  # 学习率衰减因子
    
    # 数据增强配置
    RANDOM_FLIP_PROB = 0.5                 # 随机翻转概率
    RANDOM_ROTATION_DEGREES = 10           # 随机旋转角度
    COLOR_JITTER_BRIGHTNESS = 0.2          # 亮度抖动
    COLOR_JITTER_CONTRAST = 0.2            # 对比度抖动
    COLOR_JITTER_SATURATION = 0.2          # 饱和度抖动
    COLOR_JITTER_HUE = 0.1                 # 色调抖动
    
    # 正则化配置
    DROPOUT_RATE = 0.5                     # Dropout比例
    
    # 输入图像配置
    IMAGE_SIZE = 224                       # 输入图像尺寸
    
    # 保存配置
    MODEL_SAVE_PATH = 'best_resnet_model.pth'  # 模型保存路径
    SAVE_TRAINING_PLOTS = True             # 是否保存训练图表
    
    # 设备配置
    DEVICE = 'auto'                        # 设备选择 ('auto', 'cpu', 'cuda')
    NUM_WORKERS = 4                        # 数据加载器工作进程数
    
    # 早停配置
    EARLY_STOPPING = False                 # 是否启用早停
    EARLY_STOPPING_PATIENCE = 10           # 早停耐心值
    EARLY_STOPPING_MIN_DELTA = 0.001       # 早停最小改善值
    
    # 类别权重 (用于处理不平衡数据)
    USE_CLASS_WEIGHTS = False              # 是否使用类别权重
    
    # 验证配置
    VALIDATION_FREQUENCY = 1               # 验证频率 (每N个epoch验证一次)
    
    # 日志配置
    VERBOSE = True                         # 是否显示详细日志
    LOG_INTERVAL = 10                      # 日志打印间隔

# 不同模型架构的配置
MODEL_CONFIGS = {
    'resnet18': {
        'model_name': 'resnet18',
        'feature_dim': 512
    },
    'resnet34': {
        'model_name': 'resnet34',
        'feature_dim': 512
    },
    'resnet50': {
        'model_name': 'resnet50',
        'feature_dim': 2048
    }
}

# 数据增强策略配置
AUGMENTATION_CONFIGS = {
    'light': {
        'random_flip_prob': 0.3,
        'random_rotation_degrees': 5,
        'color_jitter_brightness': 0.1,
        'color_jitter_contrast': 0.1,
        'color_jitter_saturation': 0.1,
        'color_jitter_hue': 0.05
    },
    'medium': {
        'random_flip_prob': 0.5,
        'random_rotation_degrees': 10,
        'color_jitter_brightness': 0.2,
        'color_jitter_contrast': 0.2,
        'color_jitter_saturation': 0.2,
        'color_jitter_hue': 0.1
    },
    'heavy': {
        'random_flip_prob': 0.7,
        'random_rotation_degrees': 15,
        'color_jitter_brightness': 0.3,
        'color_jitter_contrast': 0.3,
        'color_jitter_saturation': 0.3,
        'color_jitter_hue': 0.15
    }
}