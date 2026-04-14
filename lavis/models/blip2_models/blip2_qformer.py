import logging
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import time
import numpy as np
import cv2
import math
import os
from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from lavis.models.blip2_models.Qformer import BertAttention
from lavis.models.blip2_models.utils import *
from scipy.special import comb as n_over_k
from scipy.optimize import linear_sum_assignment
from lavis.models.blip2_models.Fusion import CrossAttn
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel

class LatePooler(nn.Module):
    def __init__(self, 
                 input_dim=1024,
                 hidden_dim=512,
                 output_dim=1024,
                 num_attention_heads=4
                 ):
        super().__init__()
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.hidden_size = hidden_dim
        encoder_config.num_attention_heads = num_attention_heads
        self.pooler_attention = BertAttention(
                encoder_config, is_cross_attention=False
            )
        
    def forward(self, 
                hidden_states,
                attention_mask=None):
        visual_output,_ = self.pooler_attention(hidden_states=hidden_states,
                                        attention_mask=attention_mask
        )
        return visual_output

def watershed_seg(attention_map):
    def get_mask(img_binary):
        img_binary = img_binary.astype(np.uint8)
        n_labels, labels, stats, controids = cv2.connectedComponentsWithStats(img_binary, connectivity=8)
        mask = np.zeros_like(img_binary)
        for y0,x0,y_bias,x_bias,n in stats[1:]:
            if n>2:
                mask[x0:x0+x_bias, y0:y0+y_bias] = 1
        return mask
    
    sorted_map = np.sort(attention_map.flatten())
    min_threshold = sorted_map[int(len(attention_map)**2*0.7)]
    img_binary = (attention_map>min_threshold).astype(np.uint8)
    foreground_mask = get_mask(img_binary)
    background_mask = 1-foreground_mask

    threshold = sorted_map[int(len(attention_map)**2*0.9)]
    attention_map[attention_map<=threshold]==0

    binary_map = (attention_map>threshold).astype(np.uint8)
    top10_indices = np.argsort(-attention_map.flatten())[:20]
    top10_row_indices, top10_col_indices = np.unravel_index(top10_indices, attention_map.shape)
    top10_idx = list(zip(top10_row_indices, top10_col_indices))
    unkonwn = binary_map
    markers = binary_map #background:1, unkonwn: 0
    markers = markers+1
    markers[unkonwn==1]=0
    markers = markers.astype(np.int32)
    for idx in top10_idx:
        markers[idx] = 2
    attention_map = ((attention_map/attention_map.max())*255).astype(np.uint8)
    attention_map = cv2.cvtColor(attention_map, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(attention_map, markers)
    
    markers[0]=1
    markers[-1]=1
    markers[:,0]=1
    markers[:,-1]=1
    markers = np.maximum(0, -markers)
    map = get_mask(markers)
    map = map
    return map             

def sigma_batch(maps):
    '''
    maps: bs, grid_size, grid_size
    '''  
    masks = np.stack([watershed_seg(map) for map in maps])
    return masks.astype(np.float32)   
        

def concat_all_gather_irr(text_feat, vec_length):
    l, d = text_feat.shape
    max_l = vec_length
    text_max_feat = torch.zeros((max_l, d), device=text_feat.device, dtype=text_feat.dtype)
    text_max_feat[:l, :] = text_feat
    text_mask = torch.zeros(max_l, device=text_feat.device, dtype=int)
    text_mask[:l] = torch.ones(l, device=text_feat.device, dtype=int)

    text_max_feat_all = concat_all_gather(text_max_feat)  # [batch_size*num_gpu, embed_dim]
    text_mask_all = concat_all_gather(text_mask)  # [batch_size*num_gpu, embed_dim]
    indices = torch.nonzero(text_mask_all == 1).squeeze()
    text_feat_all = text_max_feat_all[indices]
    return text_feat_all, text_mask_all
        
@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        # "pretrain_vitL364": "configs/models/blip2/blip2_pretrain_vitL364.yaml",

        # word retrieval
        "pretrain_siglip512_word_hug": "configs/models/blip2/word/stage1_pt_512.yaml",
        "ft_siglip640_word_hug": "configs/models/blip2/word/stage2_ft_640.yaml",
        "ft_siglip800_word_hug": "configs/models/blip2/word/stage3_ft_800.yaml",
        "ft_siglip800_word_hug_rnn": "configs/models/blip2/word/stage4_ft_800.yaml",

        # multi-query text retrieval
        "pretrain_siglip512_hug_cross": "configs/models/blip2/mqtr/stage1_pt_512.yaml",
        "ft_siglip640_hug_cross": "configs/models/blip2/mqtr/stage2_ft_640.yaml",
        "ft_siglip800_hug_cross": "configs/models/blip2/mqtr/stage3_ft_800.yaml",
        "ft_siglip800_hug_cross_rnn": "configs/models/blip2/mqtr/stage4_ft_800.yaml",

    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        is_learnable_query=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        num_text_token=12,
        vision_width=1024,
        kg_loss_weight=0,
        loss_itc_weight=1,
        agg_method='plain',
        is_recurrent=False
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.t2i_weight = 1/0.7
        self.kg_loss_weight = kg_loss_weight
        self.max_txt_len = max_txt_len
        self.is_learnable_query = is_learnable_query
        self.loss_itc_weight = loss_itc_weight
        self.agg_method = agg_method
        self.is_recurrent = is_recurrent
        self.save_recurrent_attn = False

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            logging.info("freeze vision encoder")
        self.visual_encoder.float()
        self.num_query_token = num_query_token
                    
        self.Qformer, self.query_tokens, _, text_prompt = self.init_Qformer(
            num_query_token, vision_width, cross_attention_freq, num_text_token=num_text_token
        )
        self.text_prompt = text_prompt if self.is_learnable_query else None
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
                
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
        self.temp = nn.Parameter(0.05 * torch.ones([]))
        self.temp.requires_grad = False
        
        if self.vis_name=='siglip':
            self.mlp_proj = nn.Sequential(
                nn.Linear(768, 768, bias=True),
                nn.GELU(),
                nn.Linear(768, 1024, bias=True),
            )
        elif self.vis_name=='RN50x4':
            self.mlp_proj = nn.Sequential(
                nn.Linear(640, 768, bias=True),
                nn.GELU(),
                nn.Linear(768, 1024, bias=True),
            )
    
        if agg_method=='cross' or agg_method=='hug_cross':
            self.aggregator = CrossAttn(num_mha_heads=8, embed_dim = 768, transformer_dropout=0.)
        self.recurrent_steps = 0
        if self.is_recurrent:
            self.recurrent_steps = 1
            self.latePooler = LatePooler(input_dim=768, hidden_dim=768, output_dim=1024, num_attention_heads=6)
            
    def prepare_query_batch(self, types=None):
        queries = [self.query_tokens for type in types]
        return torch.cat(queries, dim=0)
    
    def encode_image(self, image):
        if self.vis_name=='siglip':
            visual_output = self.visual_encoder(image)['last_hidden_state']
            image_embeds = self.mlp_proj(visual_output)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            return image_embeds, image_atts, visual_output
            
        elif self.vis_name=='vit':
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            return image_embeds, image_atts, image_embeds
        
        elif self.vis_name=='RN50x16':
            image_embeds = self.visual_encoder(image).permute(1,0,2).contiguous()
            image_embeds = self.mlp_proj(image_embeds)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            return image_embeds, image_atts, image_embeds
        else :

            image_embeds = self.visual_encoder(image).permute(1,0,2).contiguous()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )
            return image_embeds, image_atts, image_embeds
    
    def sas(self, 
            latePooler,
            query_visual_sim,
            visual_tokens, 
            history_mask=None, 
            query_tokens=None,
            image_atts=None):
        
        bs, sqs_len, dim = visual_tokens.shape
        grid_size = int(math.sqrt(sqs_len))
        
        query_visual_sim_numpy = query_visual_sim.detach().cpu().numpy()
        query_visual_sim_numpy = query_visual_sim_numpy.reshape((bs, grid_size, grid_size))
        
        binary_mask = 1-sigma_batch(query_visual_sim_numpy)
        binary_mask = torch.tensor(binary_mask, device=visual_tokens.device).view(bs, -1)
        if history_mask!=None:
            binary_mask = binary_mask*history_mask
        image_attention_mask = binary_mask.clone()
        image_attention_mask = image_attention_mask.masked_fill_(image_attention_mask==0, -10000)
        image_attention_mask = image_attention_mask[:, None, None, :]
        
        poolered_visual_output = latePooler(
            hidden_states=visual_tokens,
            attention_mask=image_attention_mask)
        poolered_image_embeds = self.mlp_proj(poolered_visual_output)
        
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=poolered_image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,
            return_dict=True,
        )
        return query_output, binary_mask, poolered_visual_output
    
    def pve(self, poolered_visual_output, query_output, query_tokens, image_atts):
        fine_image_output = [] 
        history_mask = None
        for i in range(self.recurrent_steps):
            #[bs,token_len,dim] × [bs,query_len,dim] 
            attn_probs_ = torch.stack([query_output.cross_attentions[k]
                for k in range(0, 12, 2)
            ], dim=0)            
            attn_probs = attn_probs_.mean(0).mean(1).mean(1)
            latePooler1 = self.latePooler
            query_output, history_mask, poolered_visual_output = \
                        self.sas(latePooler1,
                                 attn_probs, 
                                 poolered_visual_output, 
                                 history_mask, 
                                 query_tokens=query_tokens, 
                                 image_atts=image_atts)
            fine_image_output.append(query_output.last_hidden_state)
        fine_image_output = torch.cat(fine_image_output, dim=1)
        return fine_image_output
    
    def get_pos_idx_from_world(self, text, samples, device):
        image_ids = samples['image_id'].view(-1, 1)
        image_repeated_ids = torch.stack([samples['image_id'][i] for i, caption in enumerate(samples['text_input']) for c in caption]).view(-1, 1)
        all_image_repeated_ids, _ = concat_all_gather_irr(image_repeated_ids, 100*len(image_ids))
        all_image_ids = concat_all_gather(image_ids)        
        
        i2t_pos_idx_pre = torch.eq(image_ids, all_image_repeated_ids.t()).int()
        t2i_pos_idx_pre = torch.eq(image_repeated_ids, all_image_ids.t()).int()  
            
        caption_using = np.array(text) #all captions using in this device
        image_caption_using = [caption for caption in samples["text_input"]] #all ocr cues for each iamge
        padded_length = 6000
        captions=  samples["text_input"]
        caption_list = [torch.tensor(list('#*#'.join(caption).encode()), dtype=torch.uint8) for caption in captions]
        assert sum([cap.shape[0]>padded_length for cap in caption_list])==0, 'sentence > padded_length'

        image_caption_list = torch.stack([torch.nn.functional.pad(caption, (0, padded_length-len(caption)), value=0) for caption in caption_list]).to(device)
        image_caption_list = concat_all_gather(image_caption_list)
        image_caption_all = image_caption_list.cpu().numpy()
        image_caption_all = [np.asarray(caption, dtype=np.uint8).tobytes().decode().replace('\x00', '') for caption in image_caption_all]
        image_caption_all = [cap.split('#*#') for cap in image_caption_all]
        caption_all = np.array([c for cap in image_caption_all for c in cap])

        #get pos indexs
        i2t_pos_idx = []
        for ocr_list in image_caption_using:
            ocr_list = np.array(ocr_list)
            mask = np.isin(caption_all, ocr_list)
            i2t_pos_idx.append(torch.from_numpy(mask))
        i2t_pos_idx = torch.stack(i2t_pos_idx).to(device, dtype=int)

        t2i_pos_idx = []
        for ocr_list in image_caption_all:
            ocr_list = np.array(ocr_list)
            mask = np.isin(caption_using, ocr_list)
            t2i_pos_idx.append(torch.from_numpy(mask))
        t2i_pos_idx = torch.stack(t2i_pos_idx).T.to(device, dtype=int)
        
        i2t_pos_idx = i2t_pos_idx|i2t_pos_idx_pre
        t2i_pos_idx = t2i_pos_idx|t2i_pos_idx_pre
        return i2t_pos_idx, t2i_pos_idx
    
    def hugrain_match(self, image_feats, image_feats_all, word_text_feat, batch_size, word_length, max_l):
        word_text_feat_all, word_text_mask_all = concat_all_gather_irr(word_text_feat, max_l)
        word_sim_q2t = torch.matmul(
                image_feats.unsqueeze(1), word_text_feat_all.unsqueeze(-1)
            ).squeeze(-1)   
        
        word_sim_i2t, _ = word_sim_q2t.max(-1)
        if len(word_text_feat):
            rank = dist.get_rank()
            before_this_rank_num = sum(word_text_mask_all[:rank*max_l]) if rank else 0
            before_this_rank_num = int(before_this_rank_num)
            assert word_text_feat[0][0] == word_text_feat_all[before_this_rank_num][0], f'rank {rank} unequal'
            word_text_sim_q2t_numpy = word_sim_q2t.detach().cpu().numpy()
            cols=[]
            for i in range(batch_size):
                start_idx = sum(word_length[:i]) if i>0 else 0
                end_idx = sum(word_length[:i+1])
                text_indices = np.arange(start_idx+before_this_rank_num, end_idx+before_this_rank_num)
                row, col = linear_sum_assignment(-word_text_sim_q2t_numpy[i][text_indices])
                for j, k in zip(text_indices, col):
                    word_sim_i2t[i][j] = word_sim_q2t[i][j][k]
                cols.append(col)        
        word_sim_i2t = word_sim_i2t / self.temp
        
            
        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        word_sim_t2q = torch.matmul(
            word_text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze(-2)
        # text-image similarity: aggregate across all quersy tokens
        word_sim_t2i, _ = word_sim_t2q.max(-1)
        if len(word_text_feat):
            assert image_feats[0][0][0] == image_feats_all[rank*batch_size][0][0], f'rank {rank} unequal'
            for i in range(batch_size):
                start_idx = sum(word_length[:i]) if i>0 else 0
                end_idx = sum(word_length[:i+1])
                text_indices = np.arange(start_idx, end_idx)            
                col = cols[i]
                for j, k in zip(text_indices, col):
                    word_sim_t2i[j][i+rank*batch_size] = word_sim_t2q[j][i+rank*batch_size][k]        
        word_sim_t2i = word_sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        return word_sim_i2t, word_sim_t2i
    
    def muti_word(self, image_feat_for_agg, image_feat_for_agg_all, non_word_text_feat, batch_size):
        out = self.aggregator(non_word_text_feat, image_feat_for_agg_all)
        # non_word_text_feat_normed = F.normalize(non_word_text_feat, dim=-1)
        non_word_text_feat_normed = F.normalize(
            self.text_proj(non_word_text_feat), dim=-1)
        out = F.normalize(self.vision_proj(out), dim=-1)
        #[bs, text_len, 1, dim]×[1, text_len, dim, 1]
        non_word_sim_t2i = torch.matmul(
            out.unsqueeze(-2), non_word_text_feat_normed.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)
        non_word_sim_t2i = non_word_sim_t2i.T/self.temp
        
        max_l = 100*batch_size
        non_word_text_feat_all, non_word_text_mask_all = concat_all_gather_irr(non_word_text_feat, max_l)
        out = self.aggregator(non_word_text_feat_all, image_feat_for_agg)
        non_word_text_feat_all_normed = F.normalize(non_word_text_feat_all)
        out = F.normalize(out, dim=-1)
        non_word_sim_i2t = torch.matmul(
            out.unsqueeze(-2), non_word_text_feat_all_normed.unsqueeze(0).unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)
        non_word_sim_i2t = non_word_sim_i2t/self.temp
        return non_word_sim_i2t, non_word_sim_t2i
        
    def forward(self, samples):
        image = samples["image"]
        text = samples["text_input"]
        types = samples["type"]
        word_length = [sum([t=='word' for t in _type]) for _type in types]
        batch_size = image.size(0)
        device = image.device
        text_length = [len(t) for t in text]
        text = [s for t in text for s in t] #batch*forge_num
        
        _types = [t for _type in types for t in _type]
        word_indices = [i for i,t in enumerate(_types) if t=='word']
        non_word_indices = [i for i,t in enumerate(_types) if t!='word']
        is_type_word = [1 if t=='word' else 0 for t in _types ]
        
        #### image embedding
        image_embeds, image_atts, visual_output = self.encode_image(image)
        query_tokens = self.prepare_query_batch(types).to(device)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        image_output = query_output.last_hidden_state
        
        ####text embedding
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        prompt_text_mask = text_tokens.attention_mask
        text_prompt = None
        if self.is_learnable_query:
            text_prompt = self.text_prompt.expand(len(text), -1, -1)
            prompt_mask = torch.ones(text_prompt.size()[:-1], dtype=torch.long).to(
                device
            )
            prompt_text_mask = torch.cat([prompt_mask, text_tokens.attention_mask], dim=1)
        #learnable prompt text feature
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            prompt_embedding = text_prompt,
            attention_mask=prompt_text_mask,
            return_dict=True,
        ) 
        text_output = text_output.last_hidden_state[:, 0, :]#batch*text_length×256
        
        text_feat = self.text_proj(text_output)#batch*text_length×256
        image_feats = self.vision_proj(image_output)
        max_l = 100*batch_size
        text_feat_all, text_mask_all = concat_all_gather_irr(text_feat, max_l)
        
        ####progressive image embedding
        if self.is_recurrent:
            fine_image_outputs = self.pve(visual_output.clone(), query_output, query_tokens, image_atts)
            if self.agg_method != 'hungarian':
                image_output = torch.cat((image_output, fine_image_outputs), dim=1)
        
        if self.agg_method=='plain':
            image_feats = self.vision_proj(image_output)
            image_feats = F.normalize(image_feats, dim=-1)
            text_feat = F.normalize(text_feat, dim=-1)
            image_feats_all = concat_all_gather(
                image_feats
            )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
            text_feat_all, text_mask_all = concat_all_gather_irr(text_feat, max_l)
            
            sim_q2t = torch.matmul(
                image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
            ).squeeze(-1)
            # [batch_size, batch_size*num_gpu, num_query_tokens]

            # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            if dist.get_rank()==0:
                torch.set_printoptions(threshold=torch.inf)
                # print(_[0][:20])
            sim_i2t = sim_i2t / self.temp

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
            ).squeeze(-2)

            # text-image similarity: aggregate across all query tokens
            sim_t2i, _ = sim_t2q.max(-1)
            sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
            
        elif self.agg_method=='hungarian':
            image_feats = F.normalize(image_feats, dim=-1)
            text_feat = F.normalize(text_feat, dim=-1)[:len(text)]
            image_feats_all = concat_all_gather(
                image_feats
            )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
            text_feat_all, text_mask_all = concat_all_gather_irr(text_feat, max_l)

            #[image_num, num_ocr*gpu, num_query_token]
            sim_q2t = torch.matmul(
                image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
            ).squeeze(-1)   

            # image-text similarity: aggregate across all query tokens
            sim_i2t, _ = sim_q2t.max(-1)
            rank = dist.get_rank()
            before_this_rank_num = sum(text_mask_all[:rank*max_l]) if rank else 0
            before_this_rank_num = int(before_this_rank_num)
            assert text_feat[0][0] == text_feat_all[before_this_rank_num][0], f'rank {rank} unequal'
            sim_q2t_numpy = sim_q2t.detach().cpu().numpy()
            cols=[]
            for i in range(batch_size):
                start_idx = sum(text_length[:i]) if i>0 else 0
                end_idx = sum(text_length[:i+1])
                text_indices = np.arange(start_idx+before_this_rank_num, end_idx+before_this_rank_num)
                row, col = linear_sum_assignment(-sim_q2t_numpy[i][text_indices])
                for j, k in zip(text_indices, col):
                    sim_i2t[i][j] = sim_q2t[i][j][k]
                cols.append(col)
            sim_i2t = sim_i2t / self.temp
            
            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
            ).squeeze(-2)
            # text-image similarity: aggregate across all quersy tokens
            sim_t2i, _ = sim_t2q.max(-1)
            assert image_feats[0][0][0] == image_feats_all[rank*batch_size][0][0], f'rank {rank} unequal'
            for i in range(batch_size):
                start_idx = sum(text_length[:i]) if i>0 else 0
                end_idx = sum(text_length[:i+1])
                text_indices = np.arange(start_idx, end_idx)            
                col = cols[i]
                for j, k in zip(text_indices, col):
                    sim_t2i[j][i+rank*batch_size] = sim_t2q[j][i+rank*batch_size][k]        
            sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]
        
        elif self.agg_method=='hug_cross':
            image_feats = self.vision_proj(image_output)
            image_feats = F.normalize(image_feats, dim=-1)
            image_feats_all = concat_all_gather(
                image_feats
            )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
            word_text_feat = F.normalize(text_feat[word_indices], dim=-1)
            non_word_text_feat = F.normalize(text_feat[non_word_indices], dim=-1)
            
            # image-text similarity: aggregate across all query tokens
            word_sim_i2t, word_sim_t2i = self.hugrain_match(
                            image_feats[:, :self.num_query_token, :], 
                            image_feats_all[:, :self.num_query_token, :], 
                            word_text_feat, batch_size, word_length, max_l)
            
            word_text_feat_all, word_text_mask_all = concat_all_gather_irr(word_text_feat, max_l)
            if image_feats.shape[1]>self.num_query_token:
                image_feats_r = image_feats[:, self.num_query_token:, :]
                image_feats_all_r = image_feats_all[:, self.num_query_token:, :]

                word_sim_q2t_r = torch.matmul(
                        image_feats_r.unsqueeze(1), word_text_feat_all.unsqueeze(-1)
                    ).squeeze(-1)   
                word_sim_i2t_r, _ = word_sim_q2t_r.max(-1)

                word_sim_t2q_r = torch.matmul(
                    word_text_feat.unsqueeze(1).unsqueeze(1), image_feats_all_r.permute(0, 2, 1)
                ).squeeze(-2)
                word_sim_t2i_r, _ = word_sim_t2q_r.max(-1)
                word_sim_t2i = torch.stack([word_sim_t2i, word_sim_t2i_r])
                word_sim_i2t = torch.stack([word_sim_i2t, word_sim_i2t_r])
                word_sim_i2t, _ = word_sim_i2t.max(0); word_sim_t2i, _ = word_sim_t2i.max(0)
            # word_sim_i2t = word_sim_i2t[0]; word_sim_t2i = word_sim_t2i[0]
            # import pdb; pdb.set_trace()
            
            agg_choice = 1
            #aggregation choice 1: before projection
            #aggregation choice zero, after projection
            if agg_choice==0: 
                #num_vids x num_texts x embed_dim
                # image_feats = image_output
                image_feats = self.vision_proj(image_output)
                image_feats_all = concat_all_gather(
                    image_feats
                )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
                non_word_text_feat = text_feat[non_word_indices]
                non_word_sim_i2t = []; non_word_sim_t2i = []
                for t in range(self.recurrent_steps+1):  
                # for t in range(1):    
                    non_word_sim_i2t_t, non_word_sim_t2i_t = \
                        self.muti_word(image_feats[:, t*self.num_query_token:(t+1)*self.num_query_token, :], 
                                       image_feats_all[:, t*self.num_query_token:(t+1)*self.num_query_token, :], 
                                       non_word_text_feat, batch_size)
                    non_word_sim_i2t.append(non_word_sim_i2t_t); non_word_sim_t2i.append(non_word_sim_t2i_t)
                non_word_sim_i2t = torch.stack(non_word_sim_i2t); non_word_sim_t2i = torch.stack(non_word_sim_t2i)
                non_word_sim_i2t, _ = non_word_sim_i2t.max(0); non_word_sim_t2i, _ = non_word_sim_t2i.max(0)
                # non_word_sim_i2t = non_word_sim_i2t[0]; non_word_sim_t2i = non_word_sim_t2i[0]
                
                all_is_type_word, _ = concat_all_gather_irr(torch.tensor(is_type_word, device=device).unsqueeze(-1), max_l)
                all_is_type_word = all_is_type_word.squeeze()
                all_word_indices = torch.where(all_is_type_word==1)[0]
                all_non_word_indices = torch.where(all_is_type_word==0)[0]
                
                sim_i2t = torch.cat((word_sim_i2t, non_word_sim_i2t), dim=1)
                sim_i2t[:, all_word_indices] = word_sim_i2t
                sim_i2t[:, all_non_word_indices] = non_word_sim_i2t
                sim_t2i = torch.cat((word_sim_t2i, non_word_sim_t2i), dim=0)
                sim_t2i[word_indices] = word_sim_t2i
                sim_t2i[non_word_indices] = non_word_sim_t2i
                
            #aggregation choice one, cross attention
            elif agg_choice==1:
                image_output_all = concat_all_gather(
                    image_output
                )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
                non_word_text_output = text_output[non_word_indices]
                non_word_sim_i2t = []; non_word_sim_t2i = []
                for t in range(self.recurrent_steps+1):  
                # for t in range(1):    
                    non_word_sim_i2t_t, non_word_sim_t2i_t = \
                        self.muti_word(image_output[:, t*self.num_query_token:(t+1)*self.num_query_token, :], 
                                       image_output_all[:, t*self.num_query_token:(t+1)*self.num_query_token, :], 
                                       non_word_text_output, batch_size)
                    non_word_sim_i2t.append(non_word_sim_i2t_t); non_word_sim_t2i.append(non_word_sim_t2i_t)
                non_word_sim_i2t = torch.stack(non_word_sim_i2t); non_word_sim_t2i = torch.stack(non_word_sim_t2i)
                non_word_sim_i2t, _ = non_word_sim_i2t.max(0); non_word_sim_t2i, _ = non_word_sim_t2i.max(0)
                # non_word_sim_i2t = non_word_sim_i2t[0]; non_word_sim_t2i = non_word_sim_t2i[0]
                
                all_is_type_word, _ = concat_all_gather_irr(torch.tensor(is_type_word, device=device).unsqueeze(-1), max_l)
                all_is_type_word = all_is_type_word.squeeze()
                all_word_indices = torch.where(all_is_type_word==1)[0]
                all_non_word_indices = torch.where(all_is_type_word==0)[0]
                
                sim_i2t = torch.cat((word_sim_i2t, non_word_sim_i2t), dim=1)
                sim_i2t[:, all_word_indices] = word_sim_i2t
                sim_i2t[:, all_non_word_indices] = non_word_sim_i2t
                sim_t2i = torch.cat((word_sim_t2i, non_word_sim_t2i), dim=0)
                sim_t2i[word_indices] = word_sim_t2i
                sim_t2i[non_word_indices] = non_word_sim_t2i
                
            
        if self.is_recurrent and self.agg_method=='hungarian':
            fine_image_feats = self.vision_proj(fine_image_outputs)
            fine_image_feats = F.normalize(fine_image_feats, dim=-1)
            fine_image_feats_all = concat_all_gather(
                fine_image_feats
            )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
            fine_sim_q2t = torch.matmul(
                fine_image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
            ).squeeze(-1)    
            fine_sim_i2t, _ = fine_sim_q2t.max(-1)
            fine_sim_i2t = fine_sim_i2t / self.temp
            
            
            fine_sim_t2q = torch.matmul(
                text_feat.unsqueeze(1).unsqueeze(1), fine_image_feats_all.permute(0, 2, 1)
            ).squeeze(-2)
            # text-image similarity: aggregate across all query tokens
            fine_sim_t2i, _ = fine_sim_t2q.max(-1)
            fine_sim_t2i = fine_sim_t2i / self.temp
            
            sim_i2t = torch.max(sim_i2t, fine_sim_i2t)
            sim_t2i = torch.max(sim_t2i, fine_sim_t2i)
            
            
        i2t_pos_idx, t2i_pos_idx = self.get_pos_idx_from_world(text, samples, device) 
        i2t_sim_targets = i2t_pos_idx / i2t_pos_idx.sum(1,keepdim=True) 
        i2t_sim_targets = 0.9 * i2t_sim_targets + 0.1 * torch.ones_like(i2t_sim_targets) / i2t_sim_targets.size(1)
        t2i_sim_targets = t2i_pos_idx / t2i_pos_idx.sum(1,keepdim=True) 
        t2i_sim_targets = 0.9 * t2i_sim_targets + 0.1 * torch.ones_like(t2i_sim_targets) / t2i_sim_targets.size(1)
        
        # import pdb; pdb.set_trace()
        loss_t2i_list = -torch.sum(F.log_softmax(sim_t2i, dim=1)*t2i_sim_targets,dim=1)
        loss_i2t_list = -torch.sum(F.log_softmax(sim_i2t, dim=1)*i2t_sim_targets,dim=1)   

        loss_t2i = loss_t2i_list.mean()
        loss_i2t = loss_i2t_list.mean()
        
        assert not torch.any(torch.isnan(loss_t2i)), 'loss t2i pair is nan'
        assert not torch.any(torch.isnan(loss_i2t)), 'loss i2t pair is nan'
        loss_itc_agg = (loss_t2i*self.t2i_weight+loss_i2t)/2 
        
        ###============== Image-text Matching ===================###
        loss_itm_weight = 1
        if loss_itm_weight:
            text_input_ids_world,_ = concat_all_gather_irr(text_tokens.input_ids, max_l)
            text_attention_mask_world, _ = concat_all_gather_irr(text_tokens.attention_mask, max_l)

            image_embeds_world = all_gather_with_grad(image_embeds)
            query_tokens_world = all_gather_with_grad(query_tokens)

            with torch.no_grad():
                if "image_id" in samples.keys():
                    # mask = torch.eq(image_ids, image_ids_all.t())
                    sim_t2i.masked_fill_(t2i_pos_idx.to(dtype=torch.bool), -10000)
                    sim_i2t.masked_fill_(i2t_pos_idx.to(dtype=torch.bool), -10000)

                else:    
                    sim_t2i[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(-10000)
                    sim_i2t[:, rank * batch_size : rank * batch_size + batch_size].fill_diagonal_(-10000)            
                    
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_i2t = F.softmax(sim_i2t, dim=1)

                #choose difficult text instances
                indexs = torch.zeros(batch_size, dtype=int, device=device)
                prob_all = loss_t2i_list
                for b in range(batch_size):
                    start_idx = sum(text_length[:b])
                    end_idx = sum(text_length[:b+1]) if b<batch_size-1 else sum(text_length)
                    assert end_idx-start_idx>=1, 'assert error, %d/%d, end_index:%d <= start_index%d'%(b, batch_size, start_idx, end_idx)
                    if end_idx-start_idx == 1:
                        indexs[b] = start_idx
                    else:
                        prob = prob_all[start_idx: end_idx]
                        indexs[b] = start_idx + torch.multinomial(prob, 1)
                        # indexs[b] = start_idx

            weights_t2i = weights_t2i[indexs, :]
            input_ids = text_tokens.input_ids[indexs, :]
            attention_mask = text_tokens.attention_mask[indexs, :]
            
            image_embeds_pos = image_embeds

            # select a negative image for each text
            image_embeds_neg = []
            query_token_neg = []
            
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
                query_token_neg.append(query_tokens_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
            query_token_neg = torch.stack(query_token_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_i2t[b], 1)
                text_ids_neg.append(text_input_ids_world[neg_idx])
                text_atts_neg.append(text_attention_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0).squeeze(1)
            text_atts_neg = torch.stack(text_atts_neg, dim=0).squeeze(1)
            
            text_ids_all = torch.cat(
                [input_ids, input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            
            text_atts_all = torch.cat(
                [attention_mask, attention_mask, text_atts_neg],
                dim=0,
            )  # pos, pos, neg

            # query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_tokens_itm = torch.cat((query_tokens, query_token_neg, query_tokens), dim=0)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
                device
            )
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)
            image_embeds_all = torch.cat(
                [image_embeds_pos, image_embeds_neg, image_embeds_pos], dim=0
            )  # pos, neg, pos
            image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
                device
            )

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=image_embeds_all,
                encoder_attention_mask=image_atts_all,
                return_dict=True,
            )
            
            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
            vl_output = self.itm_head(vl_embeddings)

            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(device)
            loss_itm = F.cross_entropy(logits, itm_labels)

        return BlipOutput(
            loss= loss_itm*loss_itm_weight + loss_itc_agg,
            loss_itm=loss_itm*loss_itm_weight,
            loss_itc_agg = loss_itc_agg
        )
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # print(query_tokens.shape)
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "float")
        freeze_vit = cfg.get("freeze_vit", True)
        is_learnable_query = cfg.get("is_learnable_query", False)
        freeze_qformer = cfg.get("freeze_qformer", False)
        pred_num_roi = cfg.get("pred_num_roi", False)
        vision_width = cfg.get("vision_width", 1024)
        
        # is_learnable_query = cfg.get("is_learnable_query", False)

        max_txt_len = cfg.get("max_txt_len", 32)
        num_text_token = cfg.get("num_text_token",32)
        loss_itc_weight = cfg.get("loss_itc_weight",1)
        agg_method = cfg.get("agg_method", 'plain')
        is_recurrent = cfg.get("is_recurrent", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            is_learnable_query=is_learnable_query,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            num_text_token=num_text_token,
            vision_width = vision_width,
            loss_itc_weight=loss_itc_weight,
            agg_method=agg_method,
            is_recurrent=is_recurrent
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
    
