"""
    Define config file (JSON) for LSTM classifier
"""


import json

from copy import deepcopy

class LSTMConfig():
    # define default parameters
    def __init__(self,
            seed: int = 11,
            num_epochs: int = 100,
            batch_size: int = 64,
            lr: float = 1e-3,
            freeze_embeds: bool = False,
            embedding_dim: int = 200,
            l2: float = 0.0,
            dropout: float = 0.0,
            num_layers: int = 1,
            hidden_dim: int = 300,
            glove_embeds: str = "./data/glove.twitter.27B.200d.txt"
            ):
        self.seed = seed
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.freeze_embeds = freeze_embeds
        self.embedding_dim = embedding_dim
        self.l2 = l2
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.glove_embeds = glove_embeds
        
    @classmethod
    def from_json(cls, json_file: str):
        """
            Constructs a config from a .json file
        """
        config = LSTMConfig()
        
        with open(json_file, mode="r", encoding='utf8') as file:
            text = file.read()
            
        json_config = json.loads(text)
        
        for key, value in json_config.items():
            config.__dict__[key] = value
            
            
        return(config)
        
        
    def to_json_file(self, file: str):
        """
            Writes an LSTMConfig to a .json file
        """
        args = self.__dict__
        json_string = json.dumps(args, indent=2, sort_keys=True)
        
        with open(file, mode="w+") as f:
            f.write(json_string)
        
    @classmethod
    def write_files(cls, directory: str = None, **kwargs):
        """
        Writes a set of config files for specified parameter values. 
        kwargs should be list values.
        Config files will be named based on parameter values

        """
        if directory:
            name = directory + "/"
            
        else:
            name = ""
        
        curr_configs = [(name, LSTMConfig())]
        
        for arg_name, arg_values in kwargs.items():
            file_suffix = f"{arg_name}_"
            new_configs = []
            
            for pair in curr_configs:
                name = pair[0]
                config = pair[1]
                for value in arg_values:
                    new_config = deepcopy(config)
                    new_config.__dict__[arg_name] = value
                    new_name = name + file_suffix + str(value)
                    
                    new_configs.append((new_name, new_config))
                    
            curr_configs = new_configs
            
            
        for (name, config) in curr_configs:
            config.to_json_file(f"{name}.json")
        
        