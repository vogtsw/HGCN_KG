import os
import sys
import requests
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
    # 创建模型目录
    model_dir = "models/wiki2vec"
    os.makedirs(model_dir, exist_ok=True)
    
    # Wikipedia2Vec模型URL
    model_url = "http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2"
    
    # 下载模型
    print("Downloading Wikipedia2Vec model...")
    compressed_file = os.path.join(model_dir, "enwiki_20180420_300d.pkl.bz2")
    download_file(model_url, compressed_file)
    
    # 解压模型
    print("Extracting model...")
    import bz2
    with bz2.BZ2File(compressed_file, 'rb') as source, open(os.path.join(model_dir, "enwiki_20180420_300d.pkl"), 'wb') as dest:
        dest.write(source.read())
    
    # 删除压缩文件
    os.remove(compressed_file)
    
    print("Model downloaded and extracted successfully!")

if __name__ == "__main__":
    main()
