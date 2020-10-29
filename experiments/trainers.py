from catalyst.dl import Runner
import torch
import torch.nn.functional as F


class MyFineTuneTrainer(Runner):
    def predict_batch(self, batch):
        pass

    def _hadle_batch(self, batch):
        pass


class CorefRunner(Runner):
    def predict_batch(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            p_span_indeces=batch["p_span_indeces"],
            a_span_indeces=batch["a_span_indeces"],
            b_span_indeces=batch["b_span_indeces"]
        )

    def _handle_batch(self, batch):
        labels = batch['labels']

        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            p_span_indeces=batch["p_span_indeces"],
            a_span_indeces=batch["a_span_indeces"],
            b_span_indeces=batch["b_span_indeces"]
        )

        loss = F.cross_entropy(logits, labels.long())

        self.batch_metrics = {'loss': loss}
