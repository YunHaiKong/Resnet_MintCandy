import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os
import json
from datetime import datetime
from constants import CLASS_NAMES, SUPPORTED_IMAGE_FORMATS

class ResNetClassifier(nn.Module):
    """基于ResNet的包装分类器"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetClassifier, self).__init__()
        
        # 使用预训练的ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 修改最后的全连接层
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # 添加dropout防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.backbone(x)
        return x

def load_model(model_path, device):
    """加载训练好的模型"""
    model = ResNetClassifier(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_transform():
    """获取推理时的数据预处理变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_single_image(model, image_path, transform, device):
    """对单张图片进行预测"""
    try:
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        
        # 预处理
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 类别映射 - 使用统一的常量定义
        predicted_label = CLASS_NAMES[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'probabilities': {
                'OK': probabilities[0][0].item(),
                'NG': probabilities[0][1].item()
            }
        }
    
    except Exception as e:
        return {
            'error': str(e),
            'predicted_class': None,
            'predicted_label': None,
            'confidence': 0.0
        }

def predict_batch(model, image_dir, transform, device, output_file=None):
    """批量预测图片"""
    results = []
    
    # 支持的图片格式 - 使用统一的常量定义
    supported_formats = tuple(SUPPORTED_IMAGE_FORMATS)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(supported_formats)]
    
    print(f"找到 {len(image_files)} 张图片")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"处理: {image_file}")
        
        result = predict_single_image(model, image_path, transform, device)
        result['image_file'] = image_file
        result['image_path'] = image_path
        result['timestamp'] = datetime.now().isoformat()
        
        results.append(result)
        
        if 'error' not in result:
            print(f"  预测结果: {result['predicted_label']} (置信度: {result['confidence']:.4f})")
        else:
            print(f"  错误: {result['error']}")
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {output_file}")
    
    # 统计结果
    ok_count = sum(1 for r in results if r.get('predicted_label') == 'OK')
    ng_count = sum(1 for r in results if r.get('predicted_label') == 'NG')
    error_count = sum(1 for r in results if 'error' in r)
    
    print(f"\n统计结果:")
    print(f"OK: {ok_count}")
    print(f"NG: {ng_count}")
    print(f"错误: {error_count}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='包装质量检测推理')
    parser.add_argument('--model', type=str, default='best_resnet_model.pth',
                       help='模型文件路径')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--batch', type=str, help='批量预测的图片文件夹路径')
    parser.add_argument('--output', type=str, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.image and not args.batch:
        print("请指定 --image 或 --batch 参数")
        return
    
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = load_model(args.model, device)
    
    # 获取预处理变换
    transform = get_transform()
    
    if args.image:
        # 单张图片预测
        if not os.path.exists(args.image):
            print(f"图片文件不存在: {args.image}")
            return
        
        print(f"预测图片: {args.image}")
        result = predict_single_image(model, args.image, transform, device)
        
        if 'error' not in result:
            print(f"预测结果: {result['predicted_label']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"概率分布: OK={result['probabilities']['OK']:.4f}, NG={result['probabilities']['NG']:.4f}")
        else:
            print(f"预测失败: {result['error']}")
    
    elif args.batch:
        # 批量预测
        if not os.path.exists(args.batch):
            print(f"文件夹不存在: {args.batch}")
            return
        
        print(f"批量预测文件夹: {args.batch}")
        results = predict_batch(model, args.batch, transform, device, args.output)

if __name__ == '__main__':
    main()