import os
import sys
import requests
import pandas as pd
import json
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # 创建数据目录
    data_dir = "data/exaapd"
    os.makedirs(data_dir, exist_ok=True)
    
    # 使用示例数据进行测试
    print("Creating sample dataset...")
    sample_data = [
        {
            "title": "Advances in quantum computing",
            "abstract": "Recent advances in quantum computing have shown promising results in solving complex computational problems. This paper discusses the implementation of quantum algorithms for optimization tasks.",
            "authors": ["John Smith", "Jane Doe"],
            "venue": "Nature Quantum Computing",
            "year": 2024
        },
        {
            "title": "Deep learning in medical imaging",
            "abstract": "Deep learning techniques have revolutionized medical image analysis. We present a novel approach using convolutional neural networks for tumor detection in MRI scans.",
            "authors": ["Alice Johnson", "Bob Wilson"],
            "venue": "Medical AI Journal",
            "year": 2024
        },
        {
            "title": "Sustainable energy systems",
            "abstract": "This research explores innovative approaches to renewable energy integration in smart grids. We propose a hybrid system combining solar and wind energy with advanced storage solutions.",
            "authors": ["Maria Garcia", "David Chen"],
            "venue": "Renewable Energy",
            "year": 2024
        }
    ]
    
    # 保存示例数据
    with open(os.path.join(data_dir, "sample_data.json"), "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    # 提取文本到text.txt
    print("Processing dataset...")
    with open(os.path.join(data_dir, "text.txt"), "w", encoding="utf-8") as f:
        for item in sample_data:
            # 合并标题和摘要
            text = f"{item['title']}. {item['abstract']}"
            f.write(text + "\n")
    
    print(f"Sample dataset processed and saved to {data_dir}")
    print(f"Total documents: {len(sample_data)}")
    print("\nNote: This is a sample dataset for testing. Please replace it with the actual exAAPD dataset when available.")

if __name__ == "__main__":
    main()
