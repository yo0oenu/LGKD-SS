import torch
import torch.nn as nn

class WeightMask(nn.Module):
    def __init__(self, num_prompt, batch_size, latent_height, latent_width):
        super().__init__()
        self.num_prompt = num_prompt
        self.batch_size = batch_size
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.weights = nn.Parameter(torch.ones(num_prompt, 1, latent_height, latent_width))  #[class+1, 1, H, W]
        
    def forward(self, value):  # value: latent * binary mask [prompt_cnt * batch_size, C, L_H, L_W] 
        weights = self.weights.view(self.num_prompt * self.batch_size, 1, self.latent_height, self.latent_width)  
        #weights: [num_prompts_cnt * batch_size, 1,  L_H, L_W]
        weighted_value = ((value * weights).view(self.num_prompt, self.batch_size, -1, self.latent_height, self.latent_width)).sum(dim = 0)  #[batch_sizeC, L_H, L_W]
        return weighted_value

