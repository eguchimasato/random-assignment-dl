import torch

class HParams:
    def __init__(
            self, 
            num_goods = 4, 
            num_hidden_nodes = 512,
            batch_size = 128, 
            epochs = 10000, 
            corr = 0.0, 
            device = 'mps', 
            prob = 0.0,
            lr = 0.001,
            ):
        self.num_goods = num_goods
        self.num_hidden_nodes = num_hidden_nodes
        self.batch_size = batch_size
        self.epochs = epochs
        self.corr = corr
        self.prob = prob
        self.device = device
        self.lr = lr