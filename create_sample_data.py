import os
import json
import numpy as np

def create_sample_data():
    """创建示例数据"""
    # 创建示例文档
    documents = [
        {
            "text": "Recent advances in quantum computing have shown promising results in solving complex computational problems. This paper discusses the application of quantum algorithms for optimization tasks.",
            "label": 0
        },
        {
            "text": "Deep learning in medical imaging. Deep learning techniques have revolutionized medical image analysis. We present a novel approach using convolutional neural networks for tumor detection in MRI scans.",
            "label": 1
        },
        {
            "text": "Sustainable energy systems. This research explores innovative approaches to renewable energy integration in smart grids. We propose a hybrid system combining solar and wind energy with advanced storage solutions.",
            "label": 2
        }
    ]
    
    # 保存训练集
    with open("data/exaapd/train.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    # 保存验证集（使用相同的数据）
    with open("data/exaapd/dev.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    # 保存测试集（使用相同的数据）
    with open("data/exaapd/test.json", "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print("Sample data created successfully!")

def main():
    # 创建目录
    os.makedirs("data/exaapd", exist_ok=True)
    
    # 创建示例数据
    create_sample_data()

if __name__ == "__main__":
    main()
