import torch
import torch.nn as nn
import torch.nn.functional as F
from ...subNets.transformers_encoder.transformer import TransformerEncoder   

class LengthAdaptiveEncoder(TransformerEncoder):
    def __init__(self, input_len, output_len, **kwargs):
        super().__init__(**kwargs)
        self.input_len = input_len
        self.output_len = output_len
        #print(self.input_len)
        #print(self.output_len)
        if self.output_len > self.input_len:
        # 可学习的插值token
            self.length_tokens = nn.Parameter(torch.randn(output_len - input_len, self.embed_dim))
        else:
            # 压缩情况：使用线性投影
            self.downsample = nn.Linear(input_len, output_len)
    def forward(self, x_in):
        # x_in: [batch, embed_dim, src_len] (原始输入)l:[32,50,35]
        batch_size = x_in.size(0)
        print(x_in.shape)
        # 拼接长度调整token
        if self.output_len > self.input_len:
            adjust_tokens = self.length_tokens.T.unsqueeze(0).expand(batch_size, -1, -1)
            x = torch.cat([x_in, adjust_tokens], dim=2)  
        else:
            # 压缩序列
            #x = x_in.permute(0, 2, 1) 
            print(x_in.shape) 
            x = self.downsample(x_in)      
            #x = x.permute(0, 2, 1)#[32,50,46]
            
        # 添加位置编码（需修改原位置编码）
        if self.embed_positions is not None:
            pos = self.embed_positions(torch.arange(x.size(2)).unsqueeze(0)).cuda()
            x =x.permute(0,2,1) + pos
            x =x.permute(0,2,1)
        
        return x