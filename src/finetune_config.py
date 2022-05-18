"""
Define config file (JSON) for Huggingface Finetuning
"""

import json

class Config():
    # define default parameters
    def __init__(self,
        model: str = 'xlm-roberta-base',
        seed: int = 11,
        num_epochs: int = 5,
        batch_size: int = 4,
        lr: float = 5e-5,
        num_samples: int = None
    ):
        self.model = model
        self.seed = seed
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        
    @classmethod
    def from_json(cls, json_file: str):
        """
        Constructs a config from a .json file
        """
        
        with open(json_file, mode="r", encoding='utf8') as file:
            text = file.read()
            
        json_config = json.loads(text)
        config = Config(**json_config)
            
        return(config)
        
        
    def to_json_file(self, file: str):
        """
        Writes Config to a .json file
        """
        args = self.__dict__
        json_string = json.dumps(args, indent=2, sort_keys=True)
        
        with open(file, mode="w+") as f:
            f.write(json_string)
        