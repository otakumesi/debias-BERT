from catalyst.dl import Runner

import torch.nn.functional as F

class MyFineTuneTrainer(Runner):
    def predict_batch(self, batch):
        pass

    def _hadle_batch(self, batch):
        pass


class CorefRunner(Runner):
    def predict_batch(self, batch):
        batch = batch.to(self.device)
        outputs = self.model(input_ids=batch['input_ids'],
                                   attention_mask=batch['attention_mask'],
                                   token_type_ids=batch['token_type_ids'],
                                   offsets=batch['offsets'])
        return outputs

    def _handle_batch(self, batch):
        pass
