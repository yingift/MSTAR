"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import lavis.common.dist_utils as dist_utils
from lavis.common.dist_utils import download_cached_file
from lavis.common.utils import is_url
from lavis.common.logger import MetricLogger
from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.eva_vit import create_eva_vit_g
from lavis.models.clip_vit import convert_weights_to_fp16, interpolate_pos_embed, interpolate_pos_embed3, interpolate_pos_embed2, interpolate_query_tokens
from transformers import BertTokenizer
from lavis.models.clip_vit import VisionTransformer
from lavis.models.clip_models.model import ModifiedResNet
from transformers import AutoModel, AutoConfig
from transformers import CLIPModel, CLIPImageProcessor
import clip
from safetensors.torch import load_file

from collections import OrderedDict
from itertools import repeat
import collections.abc
import math

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)    

def get_num_layer(visual_encoder, var_name=""):
    if var_name in ("embeddings.patch_embedding", "embeddings.positional_embedding", "conv1", "ln_pre"):
        return 0
    elif var_name.startswith("encoder.layers"):        
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return len(visual_encoder.encoder.layers) 

def create_siglip_base_512(image_size, precision):
    print('image size', image_size)
    if image_size!=512:
        config = AutoConfig.from_pretrained('google/siglip-base-patch16-512')
        config.vision_config.image_size=image_size
        # config.torch_dtype = 'fp16'
        model = AutoModel.from_config(config).train()
    else:
        model = AutoModel.from_pretrained('google/siglip-base-patch16-512')
        
    # if precision == "fp16":
    #     # convert_weights_to_fp16(model)
    #     model.half()
    for name, para in model.vision_model.named_parameters():
        if name.startswith('head') or 'post_layernorm' in name:
            para.requires_grad = False
    return model.vision_model
    

def create_clip_rn50(img_size, precision):
    def interpolate_pos_embed_rn50(image_size, state_dict, interpolation: str = 'bicubic', seq_dim=1):
        # Rescale the grid of position embeddings when loading from state_dict
        old_pos_embed = state_dict.get('visual.attnpool.positional_embedding', None)
        
        grid_size = round(image_size//32)
        if old_pos_embed is None:
            return
        grid_size = to_2tuple(grid_size)
        extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
        new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
        if new_seq_len == old_pos_embed.shape[0]:
            return

        if extra_tokens:
            pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
        else:
            pos_emb_tok, pos_emb_img = None, old_pos_embed
            
        old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

        print('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
        pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
        pos_emb_img = F.interpolate(
            pos_emb_img,
            size=grid_size,
            mode=interpolation,
            align_corners=True,
        )
        pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
        if pos_emb_tok is not None:
            new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
        else:
            new_pos_embed = pos_emb_img
        state_dict['visual.attnpool.positional_embedding'] = new_pos_embed
        
    state_dict = torch.jit.load('~/.cache/clip/RN50.pt', map_location='cpu').state_dict()
        
    interpolate_pos_embed_rn50(image_size=img_size, state_dict=state_dict)
    from lavis.models.clip_models.model import build_model_from_openai_state_dict
    model = build_model_from_openai_state_dict(state_dict)
    model.float()
    # model.load_state_dict(state_dict, strict=False)
    if precision == "fp16":
        convert_weights_to_fp16(model)
    return model.visual


def create_clip_rn50_16(img_size, precision):
    def interpolate_pos_embed_rn50_16(image_size, state_dict, interpolation: str = 'bicubic', seq_dim=1):
        # Rescale the grid of position embeddings when loading from state_dict
        old_pos_embed = state_dict.get('visual.attnpool.positional_embedding', None)
        
        grid_size = round(image_size//32)
        if old_pos_embed is None:
            return
        grid_size = to_2tuple(grid_size)
        extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
        new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
        if new_seq_len == old_pos_embed.shape[0]:
            return

        if extra_tokens:
            pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
        else:
            pos_emb_tok, pos_emb_img = None, old_pos_embed
            
        old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

        print('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
        pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
        pos_emb_img = F.interpolate(
            pos_emb_img,
            size=grid_size,
            mode=interpolation,
            align_corners=True,
        )
        pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
        if pos_emb_tok is not None:
            new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
        else:
            new_pos_embed = pos_emb_img
        state_dict['visual.attnpool.positional_embedding'] = new_pos_embed
        
    # print('image size', img_size)
    clip_model, processor = clip.load('RN50x4')
    state_dict = clip_model.state_dict()

    pos_shape = clip_model.visual.state_dict()['attnpool.positional_embedding'].shape
    print('before inter: ', pos_shape)
    interpolate_pos_embed_rn50_16(img_size, state_dict)

    new_clip_model = clip.model.build_model(state_dict).to('cpu')
    pos_shape = new_clip_model.visual.state_dict()['attnpool.positional_embedding'].shape
    print('after inter: ', pos_shape)
    if precision == "fp16":
        convert_weights_to_fp16(new_clip_model)
    return new_clip_model.visual
    
def create_clip_vit_L(img_size=224, use_checkpoint=False,precision="fp16"):
    model = VisionTransformer(
            input_resolution=img_size,
            patch_size=14,
            width=1024,
            layers=23,
            heads=16,
            use_grad_checkpointing=use_checkpoint,
        )         
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth"
    cached_file = download_cached_file(
        url, check_hash=False, progress=True
    )
    state_dict = torch.load(cached_file, map_location="cpu")    
    interpolate_pos_embed(model,state_dict)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    
    if precision == "fp16":
        convert_weights_to_fp16(model)
    return model



class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2, num_text_token=12):
        # encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        # encoder_config.num_hidden_layers = 6
        # encoder_config.intermediate_size = 512*4
        # encoder_config.num_attention_heads = 8
        # Qformer = BertLMHeadModel.from_pretrained(
        #     "bert-base-uncased", config=encoder_config
        # )
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config, ignore_mismatched_sizes=True
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        
        short_query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token//1, encoder_config.hidden_size)
        )
        short_query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        text_prompt = nn.Parameter(
            torch.zeros(1, num_text_token, encoder_config.hidden_size)
        )
        text_prompt.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens, short_query_tokens, text_prompt

    def init_vision_encoder(
        self, model_name, img_size, drop_path_rate, use_grad_checkpoint, precision
    ):
        assert model_name in [
            "eva_clip_g",
            "eva2_clip_L",
            "clip_L",
            'RN50',
            'RN50x16',
            'siglip-base-512'
        ], "vit model must be eva_clip_g, eva2_clip_L or clip_L"
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                img_size, drop_path_rate, use_grad_checkpoint, precision
            )
#         elif model_name == "eva2_clip_L":
#             visual_encoder = create_eva2_vit_L(
#                 img_size, drop_path_rate, use_grad_checkpoint, precision
#             )
            ln_vision = LayerNorm(visual_encoder.num_features)
        elif model_name == "clip_L":
            visual_encoder = create_clip_vit_L(img_size, use_grad_checkpoint, precision)
            ln_vision = LayerNorm(visual_encoder.num_features)
            self.vis_name = 'vit'
        elif model_name == 'RN50':
            visual_encoder = create_clip_rn50(img_size=img_size, precision=precision)
            self.vis_name = 'rn'
            ln_vision = None
        elif model_name == 'siglip-base-512':
            visual_encoder = create_siglip_base_512(image_size=img_size, precision=precision)
            self.vis_name = 'siglip'
            ln_vision = None
        elif model_name == 'RN50x16':
            visual_encoder = create_clip_rn50_16(img_size=img_size, precision=precision)
            self.vis_name = 'RN50x16'
            ln_vision = None
            
        self.vit_name = model_name
        return visual_encoder, ln_vision

    def load_from_pretrained(self, url_or_filename):
        print(url_or_filename)
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            print(cached_file)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        if self.vis_name == 'siglip':
            interpolate_pos_embed2(self, state_dict=state_dict)
        elif self.vis_name == 'vit':
            interpolate_pos_embed(self.visual_encoder, state_dict=state_dict)
        elif self.vis_name == 'rn':
            interpolate_pos_embed3(self, state_dict=state_dict)
        bs, lens, dim = self.query_tokens.shape
        state_dict['query_tokens'] = interpolate_query_tokens(state_dict=state_dict, new_size=(lens, dim))
        if hasattr(self, 'short_query_tokens'):
            bs, lens, dim = self.short_query_tokens.shape
            if 'short_query_tokens' not in state_dict.keys():
                state_dict['short_query_tokens'] = interpolate_query_tokens(state_dict=state_dict, new_size=(lens, dim))
        
        if hasattr(self, 'word_query'):
            state_dict['word_query'] = state_dict['query_tokens']
        msg = self.load_state_dict(state_dict, strict=False)

        return msg

    def get_optimizer_params(self, weight_decay, lr_scale=1):
        
        if hasattr(self, 'blip2'):
            vit_num_layers = self.blip2.visual_encoder.get_num_layer()
        else:
            try:
                vit_num_layers = self.visual_encoder.get_num_layer()
            except:
                try:
                    vit_num_layers = get_num_layer(self.visual_encoder)
                except:
                    vit_num_layers = 0
        lr_scales = list(lr_scale ** (vit_num_layers + 1 - i) for i in range(vit_num_layers + 2))
      

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                if hasattr(self, 'blip2'):
                    layer_id = self.blip2.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))
                else:
                    try:
                        layer_id = self.visual_encoder.get_num_layer(name.replace('visual_encoder.',''))   
                    except:
                        layer_id = 0
                group_name = "vit_layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None
            
            if 'stn' in name:
                group_name = 'stn'
                
            if 'aggregation' in name:
                group_name = 'aggregation'
                    

            if group_name not in parameter_group_names:
                if layer_id is not None:
                    scale = lr_scales[layer_id]
                else:
                    scale = 1
                if group_name == 'stn':
                    scale = 1
                if group_name == 'aggregation':
                    scale = 2
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        optim_params = list(parameter_group_vars.values())
        return optim_params

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


def compute_sim_matrix(model, data_loader, **kwargs):
    k_test = kwargs.pop("k_test")

    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    logging.info("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = model.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(model.device)
        text_feat = model.forward_text(text_input)
        text_embed = F.normalize(model.text_proj(text_feat))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_ids = torch.cat(text_ids, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    vit_feats = []
    image_embeds = []

    for samples in data_loader:
        image = samples["image"]

        image = image.to(model.device)
        image_feat, vit_feat = model.forward_image(image)
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        vit_feats.append(vit_feat.cpu())
        image_embeds.append(image_embed)

    vit_feats = torch.cat(vit_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = []
    for image_embed in image_embeds:
        sim_q2t = image_embed @ text_embeds.t()
        sim_i2t, _ = sim_q2t.max(0)
        sims_matrix.append(sim_i2t)
    sims_matrix = torch.stack(sims_matrix, dim=0)

    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(model.device)

    num_tasks = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[start + i].repeat(k_test, 1, 1).to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[topk_idx],
            text_atts=text_atts[topk_idx],
        ).float()
        score_matrix_i2t[start + i, topk_idx] = score + topk_sim

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(model.device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        image_inputs = vit_feats[topk_idx.cpu()].to(model.device)
        score = model.compute_itm(
            image_inputs=image_inputs,
            text_ids=text_ids[start + i].repeat(k_test, 1),
            text_atts=text_atts[start + i].repeat(k_test, 1),
        ).float()
        score_matrix_t2i[start + i, topk_idx] = score + topk_sim

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()
