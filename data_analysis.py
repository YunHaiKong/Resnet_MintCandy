import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import numpy as np
from datetime import datetime

# 配置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def analyze_dataset(data_dir):
    """分析数据集"""
    
    print("=" * 60)
    print("数据集分析报告")
    print("=" * 60)
    
    # 统计变量
    total_samples = 0
    ok_samples = 0
    ng_samples = 0
    image_formats = Counter()
    image_sizes = []
    corrupted_files = []
    missing_images = []
    
    # 遍历所有JSON文件
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    
    print(f"找到 {len(json_files)} 个标注文件")
    
    for json_file in json_files:
        json_path = os.path.join(data_dir, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # 检查对应的图片文件
            image_path = os.path.join(data_dir, annotation['imagePath'])
            
            if not os.path.exists(image_path):
                missing_images.append(annotation['imagePath'])
                continue
            
            # 尝试加载图片
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_sizes.append((width, height))
                    
                    # 统计图片格式
                    ext = os.path.splitext(annotation['imagePath'])[1].lower()
                    image_formats[ext] += 1
                    
            except Exception as e:
                corrupted_files.append((annotation['imagePath'], str(e)))
                continue
            
            # 统计标签
            total_samples += 1
            if annotation['qualified']:
                ok_samples += 1
            else:
                ng_samples += 1
                
        except Exception as e:
            print(f"解析JSON文件失败: {json_file}, 错误: {e}")
    
    # 打印基本统计信息
    print(f"\n基本统计信息:")
    print(f"总样本数: {total_samples}")
    print(f"OK样本数: {ok_samples} ({ok_samples/total_samples*100:.1f}%)")
    print(f"NG样本数: {ng_samples} ({ng_samples/total_samples*100:.1f}%)")
    print(f"缺失图片: {len(missing_images)}")
    print(f"损坏图片: {len(corrupted_files)}")
    
    # 图片格式统计
    print(f"\n图片格式分布:")
    for fmt, count in image_formats.items():
        print(f"  {fmt}: {count} ({count/total_samples*100:.1f}%)")
    
    # 图片尺寸统计
    if image_sizes:
        widths = [size[0] for size in image_sizes]
        heights = [size[1] for size in image_sizes]
        
        print(f"\n图片尺寸统计:")
        print(f"  宽度范围: {min(widths)} - {max(widths)} (平均: {np.mean(widths):.0f})")
        print(f"  高度范围: {min(heights)} - {max(heights)} (平均: {np.mean(heights):.0f})")
        print(f"  最常见尺寸: {Counter(image_sizes).most_common(1)[0]}")
    
    # 问题文件报告
    if missing_images:
        print(f"\n缺失的图片文件:")
        for img in missing_images[:10]:  # 只显示前10个
            print(f"  {img}")
        if len(missing_images) > 10:
            print(f"  ... 还有 {len(missing_images)-10} 个")
    
    if corrupted_files:
        print(f"\n损坏的图片文件:")
        for img, error in corrupted_files[:5]:  # 只显示前5个
            print(f"  {img}: {error}")
        if len(corrupted_files) > 5:
            print(f"  ... 还有 {len(corrupted_files)-5} 个")
    
    return {
        'total_samples': total_samples,
        'ok_samples': ok_samples,
        'ng_samples': ng_samples,
        'image_formats': dict(image_formats),
        'image_sizes': image_sizes,
        'missing_images': missing_images,
        'corrupted_files': corrupted_files
    }

def plot_data_distribution(stats):
    """绘制数据分布图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 标签分布饼图
    labels = ['OK', 'NG']
    sizes = [stats['ok_samples'], stats['ng_samples']]
    colors = ['#2ecc71', '#e74c3c']
    
    axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('标签分布')
    
    # 2. 图片格式分布柱状图
    formats = list(stats['image_formats'].keys())
    counts = list(stats['image_formats'].values())
    
    axes[0, 1].bar(formats, counts, color='skyblue')
    axes[0, 1].set_title('图片格式分布')
    axes[0, 1].set_xlabel('格式')
    axes[0, 1].set_ylabel('数量')
    
    # 3. 图片宽度分布直方图
    if stats['image_sizes']:
        widths = [size[0] for size in stats['image_sizes']]
        axes[1, 0].hist(widths, bins=20, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('图片宽度分布')
        axes[1, 0].set_xlabel('宽度 (像素)')
        axes[1, 0].set_ylabel('频次')
    
    # 4. 图片高度分布直方图
    if stats['image_sizes']:
        heights = [size[1] for size in stats['image_sizes']]
        axes[1, 1].hist(heights, bins=20, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('图片高度分布')
        axes[1, 1].set_xlabel('高度 (像素)')
        axes[1, 1].set_ylabel('频次')
    
    plt.tight_layout()
    plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(stats, output_file='data_analysis_report.txt'):
    """生成详细的分析报告"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("数据集分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. 基本统计信息\n")
        f.write("-" * 20 + "\n")
        f.write(f"总样本数: {stats['total_samples']}\n")
        f.write(f"OK样本数: {stats['ok_samples']} ({stats['ok_samples']/stats['total_samples']*100:.1f}%)\n")
        f.write(f"NG样本数: {stats['ng_samples']} ({stats['ng_samples']/stats['total_samples']*100:.1f}%)\n")
        f.write(f"缺失图片: {len(stats['missing_images'])}\n")
        f.write(f"损坏图片: {len(stats['corrupted_files'])}\n\n")
        
        f.write("2. 图片格式分布\n")
        f.write("-" * 20 + "\n")
        for fmt, count in stats['image_formats'].items():
            f.write(f"{fmt}: {count} ({count/stats['total_samples']*100:.1f}%)\n")
        f.write("\n")
        
        if stats['image_sizes']:
            widths = [size[0] for size in stats['image_sizes']]
            heights = [size[1] for size in stats['image_sizes']]
            
            f.write("3. 图片尺寸统计\n")
            f.write("-" * 20 + "\n")
            f.write(f"宽度范围: {min(widths)} - {max(widths)} (平均: {np.mean(widths):.0f})\n")
            f.write(f"高度范围: {min(heights)} - {max(heights)} (平均: {np.mean(heights):.0f})\n")
            f.write(f"最常见尺寸: {Counter(stats['image_sizes']).most_common(1)[0]}\n\n")
        
        if stats['missing_images']:
            f.write("4. 缺失的图片文件\n")
            f.write("-" * 20 + "\n")
            for img in stats['missing_images']:
                f.write(f"{img}\n")
            f.write("\n")
        
        if stats['corrupted_files']:
            f.write("5. 损坏的图片文件\n")
            f.write("-" * 20 + "\n")
            for img, error in stats['corrupted_files']:
                f.write(f"{img}: {error}\n")
            f.write("\n")
        
        # 数据质量评估
        f.write("6. 数据质量评估\n")
        f.write("-" * 20 + "\n")
        
        # 类别平衡性
        balance_ratio = min(stats['ok_samples'], stats['ng_samples']) / max(stats['ok_samples'], stats['ng_samples'])
        if balance_ratio > 0.8:
            balance_status = "良好"
        elif balance_ratio > 0.5:
            balance_status = "一般"
        else:
            balance_status = "不平衡"
        f.write(f"类别平衡性: {balance_status} (比例: {balance_ratio:.2f})\n")
        
        # 数据完整性
        completeness = (stats['total_samples'] - len(stats['missing_images']) - len(stats['corrupted_files'])) / stats['total_samples']
        if completeness > 0.95:
            completeness_status = "优秀"
        elif completeness > 0.9:
            completeness_status = "良好"
        else:
            completeness_status = "需要清理"
        f.write(f"数据完整性: {completeness_status} ({completeness*100:.1f}%)\n")
        
        # 建议
        f.write("\n7. 建议\n")
        f.write("-" * 20 + "\n")
        
        if balance_ratio < 0.7:
            f.write("- 建议增加少数类别的样本数量以改善类别平衡\n")
        
        if len(stats['missing_images']) > 0:
            f.write("- 建议检查并补充缺失的图片文件\n")
        
        if len(stats['corrupted_files']) > 0:
            f.write("- 建议修复或移除损坏的图片文件\n")
        
        if stats['total_samples'] < 1000:
            f.write("- 建议增加更多训练样本以提高模型性能\n")
    
    print(f"详细报告已保存到: {output_file}")

def main():
    """主函数"""
    data_dir = './data'
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 分析数据集
    stats = analyze_dataset(data_dir)
    
    # 绘制分布图
    plot_data_distribution(stats)
    
    # 生成详细报告
    generate_report(stats)
    
    print("\n数据分析完成！")
    print("生成的文件:")
    print("- data_analysis.png: 数据分布可视化")
    print("- data_analysis_report.txt: 详细分析报告")

if __name__ == '__main__':
    main()