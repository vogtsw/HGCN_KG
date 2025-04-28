import argparse

def get_args():
    """获取命令行参数"""
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录")
    parser.add_argument("--model", type=str, required=True,
                        help="预训练模型名称")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--hidden_dim", type=int, default=768,
                        help="隐层维度")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout率")
    parser.add_argument("--num_labels", type=int, required=True,
                        help="标签数量")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--save_path", type=str, required=True,
                        help="模型保存路径")
    
    args = parser.parse_args()
    return args
