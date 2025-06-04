import json
import argparse

def merge_kg_vectors(original_file, vectors_file, output_file, max_samples=None):
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = [json.loads(line.split('\t', 1)[1]) if '\t' in line else json.loads(line) for line in f]
    with open(vectors_file, 'r', encoding='utf-8') as f:
        vectors_data = json.load(f)
    if max_samples is not None:
        original_data = original_data[:max_samples]
        vectors_data = vectors_data[:max_samples]
    merged = []
    for orig, vec in zip(original_data, vectors_data):
        orig['entities'] = vec.get('entities', [])
        orig['entity_vectors'] = vec.get('entity_vectors', [])
        merged.append(orig)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"已融合前 {len(merged)} 条数据，输出到: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_file', type=str, required=True, help='原始train.json文件')
    parser.add_argument('--vectors_file', type=str, required=True, help='带有entity_vectors的json文件')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件名')
    parser.add_argument('--max_samples', type=int, default=None, help='只处理前N条数据')
    args = parser.parse_args()
    merge_kg_vectors(args.original_file, args.vectors_file, args.output_file, args.max_samples) 