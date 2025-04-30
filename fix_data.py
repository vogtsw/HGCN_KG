import json
import re

def fix_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # 跳过文件开头的数字序列
        start_idx = content.find('{')
        if start_idx == -1:
            print(f"No JSON content found in {input_file}")
            return
        json_content = content[start_idx:]
        
        # 尝试提取所有JSON对象
        json_objects = []
        current_pos = 0
        while True:
            try:
                # 找到下一个JSON对象的开始
                obj_start = json_content.find('{', current_pos)
                if obj_start == -1:
                    break
                
                # 尝试解析JSON
                decoder = json.JSONDecoder()
                obj, end = decoder.raw_decode(json_content[obj_start:])
                json_objects.append(obj)
                current_pos = obj_start + end
            except json.JSONDecodeError:
                if current_pos >= len(json_content):
                    break
                current_pos += 1
        
        if json_objects:
            # 写入修复后的JSON
            with open(output_file, 'w', encoding='utf-8') as out:
                json.dump(json_objects, out, ensure_ascii=False, indent=2)
            print(f"Successfully fixed {input_file} -> {output_file}")
            print(f"Found {len(json_objects)} JSON objects")
        else:
            print(f"No valid JSON objects found in {input_file}")

# 修复所有数据文件
fix_json_file('data/exaapd/exAAPD_train.json', 'data/exaapd/train.json')
fix_json_file('data/exaapd/exAAPD_test.json', 'data/exaapd/test.json')
fix_json_file('data/exaapd/exAAPD_dev.json', 'data/exaapd/dev.json')
