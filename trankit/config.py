import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    def __init__(self):
        self.embedding_name = 'xlm-roberta-base'
        self.embedding_dropout = 0.3
        self.hidden_num = 300
        self.linear_dropout = 0.1
        self.linear_bias = 1
        self.linear_activation = 'relu'
        self.adapter_learning_rate = 1e-4
        self.learning_rate = 1e-3
        self.adapter_weight_decay = 1e-4
        self.weight_decay = 1e-3
        self.grad_clipping = 4.5
        self.working_dir = os.path.dirname(os.path.realpath(__file__))
        self.lowercase = False


# configuration
config = Config()
