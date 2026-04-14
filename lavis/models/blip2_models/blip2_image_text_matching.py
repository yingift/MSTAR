"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import math
import time

import torch
import torch.nn.functional as F
import numpy as np
import cv2

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer


def get_image_cache_path(image_path):
    cache_dir = os.environ.get("IMAGE_EMBED_CACHE", None)
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    _parent_dir = os.path.basename(os.path.dirname(image_path))
    _base_name = os.path.basename(image_path)
    cache_filename = f"{_parent_dir}_{_base_name}.pt"
    return os.path.join(cache_dir, cache_filename)


def get_mask_batch(maps, threshold):
    '''
    maps: bs, grid_size, grid_size
    '''
    def get_mask(map, threshold):
        img_binary = (map > threshold).astype(np.uint8)
        n_labels, labels, stats, controids = cv2.connectedComponentsWithStats(img_binary)
        mask = np.zeros_like(map)
        for y0, x0, y_bias, x_bias, n in stats[1:]:
            if n > 4:
                mask[x0:x0 + x_bias, y0:y0 + y_bias] = 1
        return mask
    masks = np.stack([get_mask(map, threshold) for map in maps])
    return masks


@registry.register_model("blip2_image_text_matching")
class Blip2ITM(Blip2Qformer):
    """
    BLIP Image-Text Matching (ITM) model.
    Supported model types:
        - pretrained: pretrained model
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_image_text_matching", "pretrained")
        >>> model = load_model("blip2_image_text_matching", "coco")
    """

    def __init__(
        self,
        vit_model="eva_clip_b",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        is_learnable_query=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        kg_loss_weight=0,
        max_txt_len=32,
        num_text_token=12,
        vision_width=1024,
        loss_itc_weight=1,
        agg_method='plain',
        is_recurrent=False,
    ):
        super().__init__(
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
            vision_width=vision_width,
            agg_method=agg_method,
            is_recurrent=is_recurrent,
        )

    def _encode_image_with_cache(self, image, image_path, device):
        """Load image embeddings from cache if available, otherwise encode and save.

        Cache stores both image_embeds and visual_output to support is_recurrent.
        Returns:
            image_embeds, image_atts, visual_output (visual_output may be None if not cached)
        """
        cache_paths = [get_image_cache_path(p) for p in image_path] if image_path else []
        all_cached = cache_paths and all(cp and os.path.exists(cp) for cp in cache_paths)

        if all_cached:
            image_embeds = []
            visual_outputs = []
            for cp in cache_paths:
                cached_data = torch.load(cp, map_location='cpu')
                visual_outputs.append(cached_data['visual_output'])
            visual_outputs = torch.stack(visual_outputs).to(device=device)
            image_atts = torch.ones(visual_outputs.size()[:-1], dtype=torch.long).to(device)
            image_embeds = self.mlp_proj(visual_outputs)

            return image_embeds, image_atts, visual_outputs
        else:
            image_embeds, image_atts, visual_output = self.encode_image(image)
            if image_path:
                for image_embed, vis_out, cp in zip(image_embeds, visual_output, cache_paths):
                    if cp:
                        torch.save({
                            'visual_output': vis_out.detach().cpu(),
                        }, cp)
            return image_embeds, image_atts, visual_output

    def forward(self, samples, match_head="itm", save_feature=False):
        image = samples.get("image", None)
        device = image.device if image is not None else self.Qformer.device

        if match_head == "itm":
            image = samples.get("image", None)
            caption = samples.get("text_input", None)
            image_path = samples.get("image_path", None)

            with self.maybe_autocast():
                try:
                    image_embeds, image_atts, _ = self._encode_image_with_cache(image, image_path, device)
                except Exception as e:
                    print(e)
                    try:
                        image_embeds, image_atts, _ = self.encode_image(image)
                    except Exception as e:
                        print(e)
                        exit(0)

            text = self.tokenizer(
                caption,
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            with self.maybe_autocast():
                output_itm = self.Qformer.bert(
                    text.input_ids,
                    query_embeds=query_tokens,
                    attention_mask=attention_mask,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    output_attentions=False,
                    return_dict=True,
                )
                itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
                itm_logit = self.itm_head(itm_embeddings)

            itm_logit = itm_logit.mean(dim=1)
            return itm_logit

        elif match_head == "itc":
            image = samples["image"]
            caption = samples["text_input"]
            types = samples.get('types', 'word')

            with self.maybe_autocast():
                try:
                    image_embeds, image_atts, visual_output = self.encode_image(image)
                except:
                    try:
                        image_embeds, image_atts, visual_output = self.encode_image1(image)
                    except Exception as e:
                        print(e)
                        exit(0)

            image_embeds = image_embeds.float()

            text_tokens = self.tokenizer(
                caption,
                truncation=True,
                max_length=self.max_txt_len,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            mask = text_tokens.attention_mask
            text_prompt = None
            if self.is_learnable_query:
                if isinstance(caption, list):
                    text_prompt = self.text_prompt.expand(len(caption), -1, -1)
                else:
                    text_prompt = self.text_prompt
                prompt_mask = torch.ones(text_prompt.size()[:-1], dtype=torch.long).to(device)
                mask = torch.cat([prompt_mask, text_tokens.attention_mask], dim=1)

            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                prompt_embedding=text_prompt,
                attention_mask=mask,
                return_dict=True,
            )
            text_output = text_output.last_hidden_state[:, 0, :]
            text_feat = self.text_proj(text_output)

            query_tokens = self.prepare_query_batch(types=[types] * image_embeds.shape[0]).to(image_embeds.device)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=self.is_recurrent,
            )
            image_output = query_output.last_hidden_state

            ####progressive image embedding
            if self.is_recurrent:
                fine_image_output = self.pve(visual_output.clone(), query_output, query_tokens, image_atts)
                image_output = torch.cat((image_output, fine_image_output), dim=1)

            if self.agg_method == 'cross':
                image_output = self.aggregator(text_output, image_output)

            image_feats = self.vision_proj(image_output)
            text_feat = F.normalize(text_feat, dim=-1)
            image_feats = F.normalize(image_feats, dim=-1)
            sims = torch.bmm(image_feats, text_feat.unsqueeze(-1))
            sim, _ = torch.max(sims, dim=1)
            return sim, _

        elif match_head == 'img':
            image = samples["image"]
            image_path = samples.get("image_path", None)

            t1 = time.time()
            with self.maybe_autocast():
                image_embeds, image_atts, visual_output = self._encode_image_with_cache(image, image_path, device)

            query_tokens = self.prepare_query_batch(types=['word'] * image_embeds.shape[0]).to(image_embeds.device)

            t2 = time.time()
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=self.is_recurrent,
            )
            image_output = query_output.last_hidden_state

            ####progressive image embedding
            if self.is_recurrent:
                fine_image_output = self.pve(visual_output.clone(), query_output, query_tokens, image_atts)
                image_output = torch.cat((image_output, fine_image_output), dim=1)

            image_feats = self.vision_proj(image_output)
            if self.agg_method == 'cross' or self.agg_method == 'hug_cross':
                return image_output
            else:
                return image_feats

        elif match_head == 'text':
            caption = samples["text_input"]

            text_tokens = self.tokenizer(
                caption,
                truncation=True,
                max_length=self.max_txt_len,
                padding="max_length",
                return_tensors="pt",
            ).to(device)
            mask = text_tokens.attention_mask
            text_prompt = None
            if self.is_learnable_query:
                if isinstance(caption, list):
                    text_prompt = self.text_prompt.expand(len(caption), -1, -1)
                else:
                    text_prompt = self.text_prompt
                prompt_mask = torch.ones(text_prompt.size()[:-1], dtype=torch.long).to(device)
                mask = torch.cat([prompt_mask, text_tokens.attention_mask], dim=1)

            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                prompt_embedding=text_prompt,
                attention_mask=mask,
                return_dict=True,
            )
            text_feat = self.text_proj(text_output.last_hidden_state[:, 0, :])
            text_output = text_output.last_hidden_state[:, 0, :]
            if self.agg_method == 'cross' or self.agg_method == 'hug_cross':
                return text_output
            else:
                return text_feat

