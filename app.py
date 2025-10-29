import os
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
from train_resnet import ResNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO
import matplotlib
from constants import CLASS_NAMES_DISPLAY, ALLOWED_EXTENSIONS, DEFAULT_MODEL_PATH, DATA_DIR, UPLOADS_DIR
matplotlib.use('Agg')  # 使用非交互式后端

# 配置matplotlib中文字体 - 强化版配置
import matplotlib.font_manager as fm

# 强制重建字体缓存
try:
    fm._rebuild()
except:
    pass

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# 验证字体设置
try:
    # 测试中文字体是否可用
    test_fig, test_ax = plt.subplots(figsize=(1, 1))
    test_ax.text(0.5, 0.5, '测试中文', fontsize=12)
    plt.close(test_fig)
    print("中文字体配置成功")
except Exception as e:
    print(f"中文字体配置警告: {e}")
    # 备用方案：直接指定字体文件
    try:
        import matplotlib.font_manager as fm
        font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        yahei_font = None
        for path in font_paths:
            if 'msyh' in path.lower() or 'yahei' in path.lower():
                yahei_font = path
                break
        if yahei_font:
            prop = fm.FontProperties(fname=yahei_font)
            plt.rcParams['font.family'] = prop.get_name()
            print(f"使用备用字体方案: {yahei_font}")
    except Exception as e2:
        print(f"备用字体方案也失败: {e2}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 从constants.py导入允许的文件扩展名

# 全局变量存储模型
model = None
device = torch.device('cpu')  # 强制使用CPU

# 应用启动时自动尝试加载模型
def init_model():
    """应用启动时初始化模型"""
    global model
    if model is None:
        success = load_model()
        if success:
            print("应用启动时模型加载成功")
        else:
            print("应用启动时模型加载失败，请手动加载模型")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """加载训练好的模型"""
    global model
    model_path = DEFAULT_MODEL_PATH
    if os.path.exists(model_path):
        try:
            model = ResNetClassifier(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"模型已从 {model_path} 加载")
            return True
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return False
    else:
        print(f"模型文件 {model_path} 不存在")
        return False

def get_transform():
    """获取图像预处理变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_image(image_path):
    """预测单张图片"""
    global model
    if model is None:
        # 尝试自动加载模型
        success = load_model()
        if not success:
            return None, "模型加载失败，请检查模型文件是否存在"
    
    try:
        # 加载和预处理图像
        image = Image.open(image_path).convert('RGB')
        transform = get_transform()
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 类别映射 - 使用统一的常量定义
        predicted_label = CLASS_NAMES_DISPLAY[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'OK': probabilities[0][0].item(),  # 修正：索引0对应OK
                'NG': probabilities[0][1].item()   # 修正：索引1对应NG
            }
        }, None
    
    except Exception as e:
        return None, f"预测时出错: {str(e)}"

def get_data_statistics():
    """获取数据集统计信息"""
    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        return None
    
    stats = {
        'total_samples': 0,
        'ok_count': 0,
        'ng_count': 0,
        'image_formats': {'bmp': 0, 'jpeg': 0, 'jpg': 0, 'png': 0}
    }
    
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                json_path = os.path.join(data_dir, filename)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    stats['total_samples'] += 1
                    if data.get('qualified', False):
                        stats['ok_count'] += 1
                    else:
                        stats['ng_count'] += 1
                
                # 统计图像格式
                image_name = filename.replace('.json', '')
                ext = image_name.split('.')[-1].lower()
                if ext in stats['image_formats']:
                    stats['image_formats'][ext] += 1
        
        return stats
    except Exception as e:
        print(f"获取数据统计时出错: {e}")
        return None

def create_statistics_chart():
    """创建数据统计图表"""
    stats = get_data_statistics()
    if not stats:
        return None
    
    # 导入FontProperties用于直接指定字体
    from matplotlib.font_manager import FontProperties
    
    # 尝试创建中文字体对象
    try:
        chinese_font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')  # Microsoft YaHei
    except:
        try:
            chinese_font = FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # SimHei
        except:
            chinese_font = FontProperties(family='sans-serif')
    
    # 设置matplotlib参数
    plt.rcParams['axes.unicode_minus'] = False
    
    # 清除之前的图形
    plt.clf()
    plt.cla()
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    plt.style.use('seaborn-v0_8')
    
    # 1. 类别分布饼图
    labels = ['OK (合格)', 'NG (不合格)']
    sizes = [stats['ok_count'], stats['ng_count']]
    colors = ['#2ecc71', '#e74c3c']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontproperties': chinese_font})
    ax1.set_title('样本类别分布', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    
    # 2. 图像格式分布条形图
    formats = list(stats['image_formats'].keys())
    counts = list(stats['image_formats'].values())
    ax2.bar(formats, counts, color=['#3498db', '#9b59b6', '#f39c12', '#1abc9c'])
    ax2.set_title('图像格式分布', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    ax2.set_xlabel('格式', fontproperties=chinese_font)
    ax2.set_ylabel('数量', fontproperties=chinese_font)
    
    # 3. 总体统计
    ax3.text(0.1, 0.8, f"总样本数: {stats['total_samples']}", fontsize=16, transform=ax3.transAxes, fontproperties=chinese_font)
    ax3.text(0.1, 0.6, f"合格样本: {stats['ok_count']}", fontsize=16, color='green', transform=ax3.transAxes, fontproperties=chinese_font)
    ax3.text(0.1, 0.4, f"不合格样本: {stats['ng_count']}", fontsize=16, color='red', transform=ax3.transAxes, fontproperties=chinese_font)
    ax3.text(0.1, 0.2, f"合格率: {stats['ok_count']/stats['total_samples']*100:.1f}%", fontsize=16, transform=ax3.transAxes, fontproperties=chinese_font)
    ax3.set_title('数据集概览', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    ax3.axis('off')
    
    # 4. 类别对比条形图
    categories = ['OK', 'NG']
    values = [stats['ok_count'], stats['ng_count']]
    bars = ax4.bar(categories, values, color=['#2ecc71', '#e74c3c'])
    ax4.set_title('类别数量对比', fontsize=14, fontweight='bold', fontproperties=chinese_font)
    ax4.set_ylabel('样本数量', fontproperties=chinese_font)
    
    # 在条形图上添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', fontsize=12, fontproperties=chinese_font)
    
    plt.tight_layout()
    
    # 转换为base64字符串
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    """主页"""
    model_loaded = model is not None
    stats = get_data_statistics()
    return render_template('index.html', model_loaded=model_loaded, stats=stats)

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和预测"""
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 添加时间戳避免文件名冲突
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 预测
        result, error = predict_image(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'filename': filename,
            'result': result
        })
    
    return jsonify({'error': '不支持的文件格式'}), 400

@app.route('/statistics')
def statistics():
    """数据统计页面"""
    chart_base64 = create_statistics_chart()
    stats = get_data_statistics()
    return render_template('statistics.html', chart=chart_base64, stats=stats)

@app.route('/model_info')
def model_info():
    """模型信息页面"""
    model_loaded = model is not None
    model_path = DEFAULT_MODEL_PATH
    model_exists = os.path.exists(model_path)
    
    info = {
        'model_loaded': model_loaded,
        'model_exists': model_exists,
        'model_path': model_path,
        'device': str(device)
    }
    
    if model_exists:
        try:
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            info['model_size'] = f"{model_size:.2f} MB"
            info['last_modified'] = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass
    
    return render_template('model_info.html', info=info)

@app.route('/load_model', methods=['POST'])
def load_model_route():
    """加载模型"""
    success = load_model()
    if success:
        return jsonify({'success': True, 'message': '模型加载成功'})
    else:
        return jsonify({'success': False, 'message': '模型加载失败'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """获取上传的文件"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    # 启动时尝试加载模型
    init_model()
    
    print("Flask应用启动中...")
    print("访问 http://localhost:5000 查看Web界面")
    app.run(debug=True, host='0.0.0.0', port=5000)