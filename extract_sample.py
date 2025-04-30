import json

def extract_sample_data(input_file, output_file, num_samples=10):
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # 找到第一个JSON对象的开始
        start_idx = content.find('{')
        if start_idx == -1:
            raise ValueError(f"No JSON content found in {input_file}")
        content = content[start_idx:]
        
        # 提取JSON对象
        data = []
        current_pos = 0
        while len(data) < num_samples:
            try:
                obj_start = content.find('{', current_pos)
                if obj_start == -1:
                    break
                
                decoder = json.JSONDecoder()
                obj, end = decoder.raw_decode(content[obj_start:])
                data.append(obj)
                current_pos = obj_start + end
            except json.JSONDecodeError:
                if current_pos >= len(content):
                    break
                current_pos += 1
        
        # 保存提取的数据
        with open(output_file, 'w', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
        
        print(f"Successfully extracted {len(data)} samples to {output_file}")

if __name__ == "__main__":
    input_file = "data/exaapd/exAAPD_train.json"
    output_file = "data/exaapd/sample_train.json"
    extract_sample_data(input_file, output_file)
