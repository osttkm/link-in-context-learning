import torch
from transformers import Swinv2Config, Swinv2Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
swinv2_config = Swinv2Config(image_size=224, patch_size=16)
swin = Swinv2Model(swinv2_config).to(device)
import pdb;pdb.set_trace()