import math
import torch.nn as nn
import torch.nn.functional as F

class MHSA(nn.Module):
    def __init__(self, embed_dim=768, num_mha_heads=12):
        super(MHSA, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, text_embeds, vision_embeds):
        num_texts, _ = text_embeds.shape
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = vision_embeds.shape
        k = self.k_proj(vision_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        k = k.permute(0,2,1,3)

        v = self.v_proj(vision_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        v = v.permute(0,2,3,1)

        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        attention = v @ attention_weights
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        o = self.out_proj(attention)
        return o


class CrossAttn(nn.Module):
    def __init__(self, num_mha_heads=16, embed_dim = 1024, transformer_dropout=0.05):
        super(CrossAttn, self).__init__()
        self.embed_dim = embed_dim
        dropout = transformer_dropout
        self.cross_attn = MHSA(num_mha_heads=num_mha_heads, embed_dim=embed_dim)
        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, vision_embeds):
        text_embeds = self.layer_norm1(text_embeds)
        vision_embeds = self.layer_norm1(vision_embeds)

        attn_out = self.cross_attn(text_embeds, vision_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)
        return out
