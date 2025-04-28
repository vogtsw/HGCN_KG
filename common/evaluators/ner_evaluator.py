import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from common.evaluators.evaluator import Evaluator

class NEREvaluator(Evaluator):
    def __init__(self, model, processor, tokenizer, args, split='dev'):
        super().__init__(model, processor, args, split)
        self.tokenizer = tokenizer
        self.label_map = processor.label_map

    def get_scores(self, silent=False):
        pred_labels = []
        true_labels = []

        self.model.eval()
        total_loss = 0

        for batch in self.data_loader:
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, attention_mask, token_type_ids, label_ids = batch

            with torch.no_grad():
                loss, logits = self.model(input_ids, token_type_ids, attention_mask, label_ids)
                total_loss += loss.item()

            # Convert logits to predictions
            preds = torch.argmax(logits, dim=2)
            
            # Remove padding from predictions and labels
            for pred, label, mask in zip(preds, label_ids, attention_mask):
                pred_label = [p.item() for p, m in zip(pred, mask) if m.item() == 1]
                true_label = [l.item() for l, m in zip(label, mask) if m.item() == 1]
                
                pred_labels.extend(pred_label)
                true_labels.extend(true_label)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted')

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': total_loss / len(self.data_loader)
        }

        if not silent:
            print(metrics)

        return metrics
