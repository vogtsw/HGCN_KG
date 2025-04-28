import torch
import torch.nn as nn
from tqdm import tqdm
from common.trainers.trainer import Trainer
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

class NERTrainer(Trainer):
    def __init__(self, model, optimizer, processor, scheduler, tokenizer, args):
        super().__init__(model, None, None, None, None, None)  # 只使用必要的参数
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.args = args
        self.num_labels = len(processor.get_labels())

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, attention_mask, token_type_ids, label_ids = batch

            # 前向传播
            loss, _ = self.model(input_ids, token_type_ids, attention_mask, label_ids)
            
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            # 反向传播
            loss.backward()
            total_loss += loss.item()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

        return total_loss / len(train_dataloader)

    def get_pytorch_dataset(self, examples):
        features = []
        for example in examples:
            tokens = self.tokenizer.tokenize(example.text_a)  # 使用text_a
            if len(tokens) > self.args.max_seq_length - 2:
                tokens = tokens[:(self.args.max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.args.max_seq_length - len(input_ids))
            input_ids += padding
            attention_mask += padding
            token_type_ids += padding

            assert len(input_ids) == self.args.max_seq_length
            assert len(attention_mask) == self.args.max_seq_length
            assert len(token_type_ids) == self.args.max_seq_length

            # 处理标签
            if isinstance(example.label, str):
                label_ids = [self.processor.label_map[example.label]] * len(input_ids)
            else:
                # 对于序列标注，标签序列需要和token对齐
                label_ids = [self.processor.label_map[l] for l in example.label]
                # 添加特殊token的标签
                label_ids = [0] + label_ids + [0]  # [CLS]和[SEP]的标签设为O
                if len(label_ids) < self.args.max_seq_length:
                    label_ids += [0] * (self.args.max_seq_length - len(label_ids))
                else:
                    label_ids = label_ids[:self.args.max_seq_length]

            features.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'label_ids': label_ids
            })

        all_input_ids = torch.tensor([f['input_ids'] for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f['attention_mask'] for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f['token_type_ids'] for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f['label_ids'] for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)
        return dataset

    def train(self):
        train_examples = self.processor.get_train_examples(self.args.data_dir)
        train_dataset = self.get_pytorch_dataset(train_examples)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        for epoch in tqdm(range(self.args.epochs), desc="Epoch"):
            loss = self.train_epoch(train_dataloader)
            print(f"Epoch {epoch + 1}/{self.args.epochs} - Loss: {loss:.4f}")

            # 保存模型
            if (epoch + 1) % 1 == 0:  # 每个epoch保存一次
                model_path = f"{self.args.model_dir}/model_epoch_{epoch+1}.bin"
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
