# ResNet 图像质量检测系统

一个基于ResNet深度学习模型的包装质量检测Web应用，能够自动识别包装图片的质量状态（合格/不合格）。

## 🚀 功能特性

- **智能检测**: 使用ResNet深度学习模型进行图像质量分析
- **Web界面**: 友好的Web用户界面，支持拖拽和点击上传
- **实时预测**: 快速获得预测结果和置信度
- **多格式支持**: 支持PNG、JPG、JPEG、BMP格式图片
- **自动模型加载**: 首次使用时自动加载模型
- **统计分析**: 提供数据集统计和模型性能分析
- **响应式设计**: 适配不同设备屏幕

## 📋 系统要求

- Python 3.8+
- PyTorch
- Flask
- PIL/Pillow
- NumPy
- 其他依赖见 `requirements.txt`

## 🛠️ 安装说明

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd resnet
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **准备模型文件**
   确保 `best_resnet_model.pth` 模型文件在项目根目录

## 🚀 快速开始

### 方法一：直接运行Python
```bash
python app.py
```

### 方法二：使用Flask开发服务器
```bash
flask run
```

启动后访问 `http://localhost:5000` 即可使用系统。

## 📖 使用指南

### 图像上传检测
1. 打开Web界面
<img width="969" height="860" alt="image" src="https://github.com/user-attachments/assets/ea80b49d-34e3-4edc-abfb-86c3ea91aeab" />

2. 拖拽图片到上传区域或点击"选择文件"按钮
<img width="969" height="543" alt="image" src="https://github.com/user-attachments/assets/a1af3e6b-ccf1-4102-9514-118cea64709c" />

3. 系统自动处理并显示预测结果
<img width="764" height="917" alt="image" src="https://github.com/user-attachments/assets/c68922d1-c2e8-4b36-ab31-ebf08e82aa61" />

4. 查看预测类别、置信度和详细概率分布
<img width="765" height="665" alt="image" src="https://github.com/user-attachments/assets/2661fe2e-ead1-48b1-b260-51149afee1b3" />

### 模型管理
- 访问 `/model_info` 查看模型状态
- 支持手动加载/卸载模型
- 查看模型详细信息

### 数据统计
- 访问 `/statistics` 查看数据集统计
- 分析样本分布和标签一致性

## 📁 项目结构

```
resnet/
├── app.py                 # Flask主应用
├── config.py              # 配置文件
├── constants.py           # 常量定义
├── train_resnet.py        # 模型训练脚本
├── inference.py           # 推理脚本
├── data_analysis.py       # 数据分析脚本
├── best_resnet_model.pth  # 训练好的模型
├── requirements.txt       # 依赖包列表
├── templates/             # HTML模板
│   ├── base.html
│   ├── index.html
│   ├── model_info.html
│   └── statistics.html
├── static/                # 静态资源
│   ├── css/
│   ├── js/
│   └── images/
├── data/                  # 数据集
├── uploads/               # 上传文件存储
└── __pycache__/           # Python缓存
```

## 🔧 配置说明

### 主要配置项（config.py）
- `UPLOAD_FOLDER`: 上传文件存储路径
- `MAX_CONTENT_LENGTH`: 最大文件大小限制
- `ALLOWED_EXTENSIONS`: 允许的文件格式

### 模型配置（constants.py）
- `CLASS_NAMES`: 分类标签
- `MODEL_PATH`: 模型文件路径
- 图像预处理参数

## 🎯 API接口

### POST /upload
上传图片进行质量检测

**请求参数:**
- `file`: 图片文件

**响应格式:**
```json
{
  "success": true,
  "result": {
    "predicted_class": 0,
    "predicted_label": "OK",
    "confidence": 0.95,
    "probabilities": {
      "OK": 0.95,
      "NG": 0.05
    }
  },
  "filename": "uploaded_image.jpg"
}
```

### GET /model_info
获取模型状态信息

### GET /statistics
获取数据集统计信息

## 🧪 模型训练

如需重新训练模型：

```bash
python train_resnet.py
```

训练脚本支持：
- 数据增强
- 学习率调度
- 早停机制
- 模型检查点保存

## 📊 数据分析

运行数据分析脚本：

```bash
python data_analysis.py
```

生成：
- 数据分布统计
- 标签一致性检查
- 可视化图表

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径
   - 确认PyTorch版本兼容性

2. **上传失败**
   - 检查文件格式和大小
   - 确认uploads目录权限

3. **预测错误**
   - 检查图片质量
   - 确认模型已正确加载

### 调试模式

启用Flask调试模式：
```bash
export FLASK_DEBUG=1
flask run
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue
- 发送邮件

## 🙏 致谢

感谢以下开源项目：
- PyTorch
- Flask
- Bootstrap
- Font Awesome

---

**注意**: 请确保在生产环境中使用HTTPS并配置适当的安全措施。
