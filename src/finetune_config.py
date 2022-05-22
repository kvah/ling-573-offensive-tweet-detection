"""
Define config file (JSON) for Huggingface Finetuning
"""

import json

from copy import deepcopy

class Config():
    # define default parameters
    def __init__(self,
        model: str = 'xlm-roberta-base',
        seed: int = 11,
        num_epochs: int = 5,
        batch_size: int = 4,
        lr: float = 5e-5,
        num_samples: int = None,
        num_warmup_steps = 0
    ):
        self.model = model
        self.seed = seed
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        self.num_warmup_steps = num_warmup_steps
        
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
            
    @classmethod
    def write_files(cls, directory: str=None, write_file: bool=True, **kwargs):
        """
        Writes a set of config files for specified parameter values. 
        kwargs should be list values.
        Config files will be named based on parameter values
        
        example usage:
        Config.write_files(directory='configs',write_file=True,
                           lr=[5e-5,5e-4],l2=[1e-5,1e-4])
        
        generates four config files:
            lr_0.0005l2_0.0001.json
            lr_0.0005l2_1e-05.json
            lr_5e-05l2_0.0001.json
            lr_5e-05l2_1e-05.json
        
        In which lr and l2 values are as specified by file name and the 
        remaining parameters are default values.
        
        If write_files = True, call generates run.args file, which 
        contains line-separated file names:
            configs/lr_5e-05l2_1e-05.json
            configs/lr_5e-05l2_0.0001.json
            configs/lr_0.0005l2_1e-05.json
            configs/lr_0.0005l2_0.0001.json

        """
        if directory:
            name = directory + "/"
            
        else:
            name = ""
        
        curr_configs = [(name, Config())]
        
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
            
        if write_file:
            with open('run.args', mode="w+") as file:
                for (name, _) in curr_configs:
                    file.write(f"{name}.json\n")
                    
            file.close()
        