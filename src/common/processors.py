class InputExample(object):
    def __init__(self, guid, text, label=None, entity_vectors=None):
        self.guid = guid
        self.text = text
        self.label = label
        self.entity_vectors = entity_vectors

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
        
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
        
    def get_test_examples(self, data_dir):
        raise NotImplementedError()
        
    def get_labels(self):
        raise NotImplementedError()
        
class ExAAPDProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train_kg.json")), "train")
            
    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_dev.json")), "dev")
            
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "exAAPD_test.json")), "test")
            
    def get_labels(self):
        return list(range(54))  # 假设有54个标签
        
    def _read_json(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line['text']
            label = line['label']
            entity_vectors = line.get('entity_vectors', None)
            examples.append(
                InputExample(guid=guid, text=text, label=label, entity_vectors=entity_vectors))
        return examples 