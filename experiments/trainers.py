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
        p_indeces = batch["p_span_indeces"].unsqueeze(1)
        a_indeces = batch["a_span_indeces"].unsqueeze(1)
        b_indeces = batch["b_span_indeces"].unsqueeze(1)

        a_span_pair_indeces = torch.cat((a_indeces, p_indeces), dim=1)
        b_span_pair_indeces = torch.cat((b_indeces, p_indeces), dim=1)

        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            a_span_pair_indeces=a_span_pair_indeces,
            b_span_pair_indeces=b_span_pair_indeces,
        )

    def _handle_batch(self, batch):
        labels = batch['labels']

        p_indeces = batch["p_span_indeces"].unsqueeze(1)
        a_indeces = batch["a_span_indeces"].unsqueeze(1)
        b_indeces = batch["b_span_indeces"].unsqueeze(1)

        a_span_pair_indeces = torch.cat((a_indeces, p_indeces), dim=1)
        b_span_pair_indeces = torch.cat((b_indeces, p_indeces), dim=1)

        logits = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            a_span_pair_indeces=a_span_pair_indeces,
            b_span_pair_indeces=b_span_pair_indeces
        )

        loss = F.cross_entropy(logits, labels.long())

        self.batch_metrics = {'loss': loss}
