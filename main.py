
import hydra
import torch
import gc

from torchvision import models
from omegaconf import OmegaConf, DictConfig
@hydra.main(version_base=None, config_path='.\configs', config_name='config.yaml')
def main(config: DictConfig):
    from train import train
    train(config)

if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    
    main()


