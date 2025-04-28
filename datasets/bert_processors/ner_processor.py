import os
from datasets.bert_processors.abstract_processor import BertProcessor, InputExample
from typing import List, Dict, Optional

class NERProcessor(BertProcessor):
    """Processor for the NER data."""
    
    # 定义实体类型
    NER_LABELS = [
        "O",  # Outside of a named entity
        "B-PER", "I-PER",  # Person
        "B-ORG", "I-ORG",  # Organization
        "B-LOC", "I-LOC",  # Location
        "B-MISC", "I-MISC"  # Miscellaneous
    ]
    
    def __init__(self):
        super().__init__()
        self.labels = self.NER_LABELS
        self.label_map = {label: i for i, label in enumerate(self.labels)}
    
    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_ner_file(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_ner_file(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_ner_file(os.path.join(data_dir, "test.txt")), "test"
        )

    def _read_ner_file(self, input_file):
        """Reads a NER file."""
        examples = []
        words = []
        labels = []
        
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append((words, labels))
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append((words, labels))
        return examples

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, labels) in enumerate(lines):
            guid = f"{set_type}-{i}"
            text_a = " ".join(sentence)  
            examples.append(InputExample(guid=guid, text_a=text_a, label=labels))
        return examples

    def get_labels(self):
        return self.labels
