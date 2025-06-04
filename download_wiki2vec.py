import os
import requests
from tqdm import tqdm

def download_file(url: str, filename: str):
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
    # Wikipedia2Vec预训练模型URL
    url = "https://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2"
    filename = "enwiki_20180420_300d.pkl.bz2"
    
    print(f"Downloading {filename}...")
    download_file(url, filename)
    
    print("Done!")

if __name__ == "__main__":
    main() 