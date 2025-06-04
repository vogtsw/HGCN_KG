import json

def check_output(file_path, num_samples=2):
    """检查输出文件的内容
    
    Args:
        file_path: 输出文件路径
        num_samples: 要检查的样本数量
    """
    print(f"检查文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n文件中共有 {len(data)} 条数据")
    
    # 检查前几个样本
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n样本 {i+1}:")
        print(f"标题: {sample.get('title', 'N/A')}")
        print(f"实体数量: {len(sample.get('entities', []))}")
        
        # 按类型统计实体
        entity_types = {}
        for entity, type_ in sample.get('entities', []):
            if type_ not in entity_types:
                entity_types[type_] = []
            entity_types[type_].append(entity)
        
        print("\n实体统计:")
        for type_, entities in entity_types.items():
            print(f"\n{type_} ({len(entities)}个):")
            for entity in entities[:5]:  # 只显示前5个
                print(f"- {entity}")
            if len(entities) > 5:
                print(f"... 还有 {len(entities)-5} 个")

if __name__ == "__main__":
    check_output("data/exaapd/train_with_entities.json") 