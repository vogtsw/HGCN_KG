import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args, train_dataset, model, tokenizer, device, optimizer, scheduler):
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    model.train()
    global_step = 0
    tr_loss = 0.0
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}")
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)
            entity_vectors = batch['entity_vectors'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, token_type_ids, entity_vectors)
            if args.num_labels == 1:
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                loss_fn = torch.nn.BCEWithLogitsLoss() if labels.ndim > 1 else torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels) if labels.ndim > 1 else loss_fn(logits, labels.long())
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_loss += loss.item()
            global_step += 1
        print(f"Epoch {epoch+1} loss: {tr_loss/global_step:.4f}")
    return global_step, tr_loss/global_step

def evaluate(args, model, eval_dataset, device):
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False)
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            label = batch['label'].to(device)
            entity_vectors = batch['entity_vectors'].to(device)
            logits = model(input_ids, attention_mask, token_type_ids, entity_vectors)
            if args.num_labels == 1:
                pred = logits.view(-1).cpu().numpy()
            else:
                pred = torch.sigmoid(logits).cpu().numpy() if label.ndim > 1 else torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    # 简单准确率
    if args.num_labels == 1:
        acc = np.mean(np.abs(preds - labels) < 0.5)
    elif labels.ndim > 1:
        acc = np.mean((preds > 0.5) == (labels > 0.5))
    else:
        acc = np.mean(preds == labels)
    return {'accuracy': float(acc)} 