import os
import yaml
import glob

class ModelRegistry:
    def __init__(self, config_dir="configs/models"):
        self.config_dir = config_dir
        self.models = {}
        self._load_models()

    def _load_models(self):
        pattern = os.path.join(self.config_dir, "*.yaml")
        for fpath in glob.glob(pattern):
            name = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath, 'r') as f:
                self.models[name] = yaml.safe_load(f)
    
    def list_models(self):
        return list(self.models.keys())
    
    def get_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {self.list_models()}")
        return self.models[name]
