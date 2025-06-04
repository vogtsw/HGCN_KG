class BertTrainer(object):
    def __init__(self, model, optimizer, processor, scheduler, tokenizer, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.processor = processor
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.train_examples = self.processor.get_train_examples(args.data_dir)
        self.train_batch_size = args.train_batch_size
        self.num_train_optimization_steps = int(
            len(self.train_examples) / self.train_batch_size) * args.num_train_epochs

    def train(self):
        train_features = convert_examples_to_features(
            self.train_examples, self.args.max_seq_length, self.tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_examples))
        logger.info("  Batch size = %d", self.train_batch_size)
        logger.info("  Num steps = %d", self.num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        
        # 添加entity_vectors的处理
        all_entity_vectors = []
        for f in train_features:
            if hasattr(f, 'entity_vectors') and f.entity_vectors:
                # 将entity_vectors转换为tensor
                entity_vectors = torch.tensor(f.entity_vectors, dtype=torch.float)
                # 如果实体数量不同，进行padding
                max_entities = 50  # 设置最大实体数
                if entity_vectors.size(0) < max_entities:
                    padding = torch.zeros((max_entities - entity_vectors.size(0), 300), dtype=torch.float)
                    entity_vectors = torch.cat([entity_vectors, padding], dim=0)
                else:
                    entity_vectors = entity_vectors[:max_entities]
            else:
                entity_vectors = torch.zeros((50, 300), dtype=torch.float)
            all_entity_vectors.append(entity_vectors)
        all_entity_vectors = torch.stack(all_entity_vectors)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_entity_vectors)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)

        self.model.train()
        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.args.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, entity_vectors = batch
                
                # 前向传播
                logits = self.model(input_ids, segment_ids, input_mask, label_ids, entity_vectors)
                
                if self.args.is_multilabel:
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1, self.args.num_labels), label_ids.view(-1, self.args.num_labels))
                else:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.args.num_labels), label_ids.view(-1))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

        # 保存模型
        torch.save(self.model, self.snapshot_path) 