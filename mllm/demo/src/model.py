
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel


class CLIP_MOFRL(torch.nn.Module):
    def __init__(self, vision_model, add_adapter:bool, model_path:str):
        super().__init__()
        if add_adapter=='True':
            self.add_adapter = True
        else:
            self.add_adapter = False

        print(f'add_adapter:{self.add_adapter}')

        self.model_path = model_path
        device = torch.device('cuda')

        self.clip_model = vision_model.float()
        for param in self.clip_model.model.vision_tower[0].parameters():
           param.requires_grad = False

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # model_dtype = next(self.clip_model.parameters()).dtype

        if self.add_adapter:
            self.mm_projector = nn.Linear(1024, 4096)
            print('==================================')
            print(self.model_path+'/pytorch_model-00003-of-00003.bin')
            print('==================================')
            weights = torch.load(self.model_path+'/pytorch_model-00003-of-00003.bin', map_location='cpu')
            self.mm_projector_weights = weights['model.mm_projector.weight']
            self.mm_projector.bias.data = weights['model.mm_projector.bias']
            self.mm_projector.weight.data = self.mm_projector_weights
            for param in self.mm_projector.parameters():
                param.requires_grad = False

            self.classifier = nn.Linear(4096, 2)
            
            self.mm_projector.float()
            self.classifier.float()
        
        else:
            self.classifier = nn.Linear(1024, 2)
            # classifierの重みを初期化する
            self.classifier.float()
    

    def get_processor(self):
        return self.processor
    
    def forward(self, x):
        with torch.no_grad():
            x = self.clip_model.model.vision_tower[0](x,output_hidden_states=True)
            x = x['hidden_states'][-2]
            # layernormの適応
            # class tokenのみ獲得
            # x = x[:, 0, :]
            x = x[:, 1:]
            if self.add_adapter:
                x = self.mm_projector(x)
        x = torch.mean(x,dim=1)
        x = self.classifier(x)
        return x
