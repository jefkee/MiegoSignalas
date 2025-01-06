import yaml

class Config:
    def __init__(self, config_file='config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
    @property
    def model_params(self):
        return self.config['model']
        
    @property
    def preprocessing_params(self):
        return self.config['preprocessing'] 