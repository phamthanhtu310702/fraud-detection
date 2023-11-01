from typing import Dict
import numpy as np
import torch


class TorchPredictor:
    def __init__(self, model):
        # Load a dummy neural network.
        # Set `self.model` to your pre-trained PyTorch model.
        self.model = model
        self.model.eval()

    # Logic for inference on 1 batch of data.
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        with torch.inference_mode():
            x = self.model(src_node_ids=batch['u'],
                            node_interact_times=batch['ts'],
                            labels = batch['label'],
                            num_neighbors=20,
                            time_gap=2000)
            predicts = x['logits'].sigmoid()
            predicts = predicts.cpu().detach().numpy().flatten()
            return {"output": predicts}
        