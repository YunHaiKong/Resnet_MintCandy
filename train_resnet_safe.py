import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
import warnings
from constants import CLASS_NAMES, CLASS_TO_INDEX
warnings.filterwarnings('ignore')

# 设置环境变量避免内存问题
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class PackagingDataset(Dataset):
    """包装质量检测数据集"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # 加载所有图片和标签
        print("正在加载数据集...")
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                json_path = os.path.join(data_dir, filename)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                    
                    image_path = os.path.join(data_dir, annotation['imagePath'])
                    if os.path.exists(image_path):
                        # 获取qualified字段作为标签，使用统一的映射
                        qualified = annotation.get('qualified', True)
                        label_key = 'qualified' if qualified else 'unqualified'
                        label = CLASS_TO_INDEX[label_key]
                        self.samples.append((image_path, label))
                except Exception as e:
                    print(f"跳过损坏的标注文件: {filename}, 错误: {e}")
                    continue
        
        print(f"成功加载 {len(self.samples)} 个样本")
        
        # 统计标签分布
        labels = [sample[1] for sample in self.samples]
        ok_count = labels.count(0)
        ng_count = labels.count(1)
        print(f"OK样本: {ok_count}, NG样本: {ng_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # 加载图片
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"加载图片失败: {image_path}, 使用默认图片")
            # 返回一个默认图片
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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

def get_transforms():
    """获取数据预处理变换"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),  # 减少数据增强强度
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    """训练模型"""
    
    # 强制使用CPU避免CUDA内存问题
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 记录训练历史
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch+1}')
        for batch_idx, (images, labels) in enumerate(train_pbar):
            try:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                current_acc = 100. * train_correct / train_total
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
                
                # 定期清理内存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    
            except Exception as e:
                print(f"训练批次错误: {e}")
                continue
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'验证 Epoch {epoch+1}')
            for images, labels in val_pbar:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # 更新进度条
                    current_acc = 100. * val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.2f}%'
                    })
                    
                except Exception as e:
                    print(f"验证批次错误: {e}")
                    continue
        
        # 计算平均损失和准确率
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        train_acc = 100. * train_correct / train_total if train_total > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        
        print(f'训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_resnet_model_safe.pth')
            print(f'保存最佳模型，验证准确率: {best_val_acc:.2f}%')
        
        scheduler.step()
        
        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

def plot_training_history(history):
    """绘制训练历史"""
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(history['train_losses'], label='训练损失')
        ax1.plot(history['val_losses'], label='验证损失')
        ax1.set_title('模型损失')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['train_accuracies'], label='训练准确率')
        ax2.plot(history['val_accuracies'], label='验证准确率')
        ax2.set_title('模型准确率')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_safe.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("训练历史图表已保存为 training_history_safe.png")
    except Exception as e:
        print(f"绘制训练历史失败: {e}")

def evaluate_model(model, test_loader):
    """评估模型性能"""
    
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='评估模型'):
            try:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"评估批次错误: {e}")
                continue
    
    if len(all_predictions) == 0 or len(all_labels) == 0:
        print("评估失败：没有有效的预测结果")
        return 0.0
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'测试准确率: {accuracy:.4f}')
    
    # 分类报告
    class_names = CLASS_NAMES
    print('\n分类报告:')
    try:
        print(classification_report(all_labels, all_predictions, target_names=class_names))
    except Exception as e:
        print(f"生成分类报告失败: {e}")
    
    # 混淆矩阵
    try:
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig('confusion_matrix_safe.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("混淆矩阵已保存为 confusion_matrix_safe.png")
    except Exception as e:
        print(f"绘制混淆矩阵失败: {e}")
    
    return accuracy

def main():
    """主函数"""
    
    print("=" * 60)
    print("包装质量检测 - ResNet模型训练 (安全版本)")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 数据路径
    data_dir = './data'
    
    if not os.path.exists(data_dir):
        print(f"错误：数据目录不存在 {data_dir}")
        return
    
    try:
        # 获取数据变换
        train_transform, val_transform = get_transforms()
        
        # 创建数据集
        full_dataset = PackagingDataset(data_dir, transform=train_transform)
        
        if len(full_dataset) == 0:
            print("错误：没有找到有效的数据样本")
            return
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # 为验证集设置不同的变换
        val_dataset.dataset.transform = val_transform
        
        # 创建数据加载器 (使用较小的批次大小和0个工作进程)
        batch_size = 8  # 减小批次大小
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"批次大小: {batch_size}")
        
        # 创建模型
        print("\n创建ResNet模型...")
        model = ResNetClassifier(num_classes=2, pretrained=True)
        
        # 训练模型
        print("\n开始训练...")
        trained_model, history = train_model(model, train_loader, val_loader, num_epochs=15)  # 减少训练轮数
        
        # 绘制训练历史
        print("\n绘制训练历史...")
        plot_training_history(history)
        
        # 评估模型
        print("\n评估模型性能...")
        evaluate_model(trained_model, val_loader)
        
        print("\n训练完成！")
        print("生成的文件:")
        print("- best_resnet_model_safe.pth: 训练好的模型")
        print("- training_history_safe.png: 训练历史曲线")
        print("- confusion_matrix_safe.png: 混淆矩阵")
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()