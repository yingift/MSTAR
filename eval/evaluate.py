import  torch.nn.functional  as F
from omegaconf import OmegaConf
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from lavis.common.registry import registry

import torch
import pickle as pkl
import os.path
from os.path import join
from os import listdir
import time
from sklearn.metrics import average_precision_score
import math
import copy
from tqdm import tqdm
import numpy as np
from lavis.models import load_model_and_preprocess, load_preprocess
from eval.datasets import *

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import multiprocessing
from torch.cuda.amp import autocast
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

def compute_map(texts, gts, pred_scores, caps, is_print = False):
    ap=[]
    pred_scores = np.array(pred_scores)
    gts = np.array(gts)
    caps = np.array(caps)
    for text_q, gt, pred_score in zip(texts, gts, pred_scores):
        p = average_precision_score(gt, pred_score)
        ap.append(p)
        if p<=0.001 and is_print:
            print(f'{text_q} GT: {[caps[gt==1]]}', f'{text_q} ap: {p}')
            print(f'{text_q} ap: {p}')
            top_10_idx = np.argsort(pred_score)[-10:]
            print(f'{text_q} retrieved top10: {[(cap, s) for cap, s in zip(caps[top_10_idx], pred_score[top_10_idx])]}')
    return np.mean(ap)

class vis_dataset(Dataset):
    def __init__(self, images, processor):
        self.images = images
        self.processor = processor
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        image_semantic = self.processor(image)
        # image_perception = self.processor(image, "processor_perception")
        return {
            "image_path": image_path,
            "image": image_semantic,
            # "image_perception": image_perception
        }

class vis_dataset_itm(Dataset):
    def __init__(self, images, processor):
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        return {
            "image_path": self.images[index],
        }
        return 

def my_collate_fn(samples):
    tensor_keys = ['image_perception', 'image']
    tensor_keys = ['image']
    tensor_keys = set(samples[0].keys()) & set(tensor_keys)
    tensor_dict = {
            key:torch.stack([sample[key] for sample in samples])  
            for key in tensor_keys
            }
    non_tensor_keys = ['image_path']
    non_tensor_dict = {
            key:[sample[key] for sample in samples]
            for key in non_tensor_keys
            }
    tensor_dict.update(non_tensor_dict)
    return tensor_dict
    
    
class evaluator(object):
    def __init__(self, 
                 model_name,
                 vitype,
                 dataset,
                 is_conditioned = False,
                 is_grad_crop = False,
                 checkpoint = None,
                 is_reranking = False,
                 is_print = False,
                 is_aug = False,
                 text_prompt = 2,
                 top_k_ratio = 128,
                 batch_size = 128,
                 num_workers = 1,
                 device = 'cpu',
                 fp16 = False,
                 ):
        
        model, vis_processors, text_processors = load_model_and_preprocess(model_name, vitype, device=device, is_eval=True)

        self.model = model
        self.vis_processors = vis_processors
        self.text_processors = text_processors
        images, images_caption, text_queries = load_dataset(dataset, is_readimage=False)
        
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location='cpu')
            model.load_state_dict(state_dict['model'], strict=False)
            
            model.to(device)
        if fp16:
            model.to(dtype=torch.float16)
        model.eval()
        
        self.images = images
        self.images_caption = np.array(images_caption)  #no proper tensor with str dtype 
        self.text_queries = text_queries
        self.is_conditioned = is_conditioned
        self.is_grad_crop = is_grad_crop
        self.is_reranking = is_reranking
        self.is_print = is_print
        self.text_prompt = text_prompt
        self.top_k_ratio = int(top_k_ratio * len(images))
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.pool = multiprocessing.Pool(processes=self.num_workers)
 
    def pipeline(self):
        with torch.no_grad():
            # images_pre = self.vis_process(self.images)
            images_pre = self.images
            text_pre = self.text_process(self.text_queries.keys(), prompt=self.text_prompt)
            t2i_sim = self.cosine_ranking(images_pre, text_pre)
            
            cos_map = compute_map(list(self.text_queries.keys()), list(self.text_queries.values()), t2i_sim, caps=self.images_caption, is_print=self.is_print)
            print('cosine similarity: (%.4f)'%cos_map)
            t2i_sim = t2i_sim.to('cpu')

        if self.is_reranking:
            for i, (t_pre, i_sim) in tqdm(enumerate(zip(text_pre, t2i_sim))):
                _, topk_idx = torch.topk(i_sim, self.top_k_ratio)
                ims = [self.images[int(i)] for i in topk_idx]

                itm_score = self.itm_ranking(t_pre, ims, batch_size=self.batch_size, is_grad_crop=self.is_grad_crop, images=ims)
                i_sim[topk_idx] += torch.tensor(itm_score).to('cpu')
                t2i_sim[i] = i_sim
            itm_map = compute_map(list(self.text_queries.keys()), list(self.text_queries.values()), t2i_sim, caps=self.images_caption, is_print=self.is_print)
            print('itm similarity: (%.4f)'%itm_map)
        
    def init_processors(self):
        model_cls = registry.get_model_class(self.name)
        cfg = OmegaConf.load(model_cls.default_config_path(self.model_type))
        if cfg is not None:
            preprocess_cfg = cfg.preprocess
            vis_processors, txt_processors = load_preprocess(preprocess_cfg)
        return vis_processors, txt_processors
    
    def vis_process(self, images):
        return torch.stack([self.vis_processors["eval"](image) for image in images], 
                           dim=0)

    def text_process(self, text, prompt=2):
        prompt_dict = {
                "default": "a photo of '<value>'",
                "none": "<value>",
                "word": "a photo of the word: '<value>'",
                "phrase" : "a photo of the phrase: '<value>'",
                "combined": "a photo of the text combination: <value>",
                "semantic": "<value>",
                "caption": "<value>"
            }
        prompt = prompt_dict[prompt]
        text_pre =[self.text_processors["eval"](prompt.replace('<value>', text_q)) for text_q in text]
        return text_pre

    def vis_extract(self, images_pre, batch_size=128, device='cpu'):
        d = vis_dataset(images_pre, self.vis_processors["eval"])
        dataloader = DataLoader(d, batch_size=batch_size, collate_fn=my_collate_fn, num_workers=self.num_workers)
        aaa = []
        t = 0
        for sample in dataloader:
            for key in list(sample.keys()):
                if key!='image_path':
                    sample[key] = sample[key].to(device)
            sample["text_input"] = ''
            s1 = time.time()
            aa = self.model(sample, match_head='img').detach().cpu()
            s2 = time.time()
            t += s2-s1 

            aaa.append(aa)
        vis_feature = torch.cat(aaa, dim=0)
        print('speed: %.2f images per sec'%(len(images_pre)/(t)))
        
        return vis_feature
    
    def text_extract(self, text_pre, device='cpu'):
        single_img = torch.zeros((2, 2)).to(device)
        try:
            texts = self.model({"image": single_img, "text_input": text_pre}, match_head='text').detach().cpu().squeeze()
        except:
            text_pre = self.model.tokenizer(text_pre).to(self.device)
            texts = self.model.encode_text(text_pre).detach().cpu().squeeze()
        return texts
    def compute_cos_sim(self, image_embed, text_embed, is_conditioned=False):
        def compute_text_conditioned(model, img_embed, text_embed):
            text_embed = text_embed.to(model.device)
            img_embed = img_embed.to(model.device)
            batch_size = 32
            iters = len(text_embed)//batch_size+1
            out = []
            
            image_feat = img_embed
            text_feat = text_embed
            for i in range(iters):
                if i==iters-1:
                    unit_text_feat = text_feat[i*batch_size:]
                else:
                    unit_text_feat = text_feat[i*batch_size:(i+1)*batch_size]
                output = []
                for i in range(1):
                    out_ = model.vision_proj(model.aggregator(unit_text_feat, image_feat[:,i*model.num_query_token:(i+1)*model.num_query_token,:]))
                    output.append(out_)
                output = torch.stack(output)
                out.append(output)
            out = F.normalize(torch.cat(out, dim=2),dim=-1)
            text_feat = F.normalize(model.text_proj(text_feat), dim=-1)
            #[text_num, vis_num, dim] * [text_num, dim, 1]
            t2i_score = torch.matmul(
                out.permute(0, 2, 1, 3), text_feat.unsqueeze(0).unsqueeze(-1)
                            ).squeeze(-1)      
            t2i_score, _ = t2i_score.max(0)
            return t2i_score.cpu()

        def compute_cos_similarity(imgs_feature, text_feature):
            '''
                text_feat:
                imgs_feat:
                imgs_caption:
                text_gt:
                return: dict{text : tuple(cosine, caption, gt) } ranking according to cos_similarity scores
            '''
            text_feature = F.normalize(text_feature, dim=-1)
            imgs_feature = F.normalize(imgs_feature, dim=-1)
            if len(imgs_feature.shape)==2 and len(text_feature.shape)==2:
                #[num_text, dim]×[num_image, dim] 427, 249
                sim = torch.matmul(text_feature, imgs_feature.T).squeeze()
            else:
                if len(imgs_feature.shape) > 3:
                    imgs_feature = imgs_feature.squeeze()
                if len(text_feature.shape) == 1:
                    text_feature = text_feature.unsqueeze(0)
                img_batch = imgs_feature.shape[0]
                text_batch, dim = text_feature.shape
                
                text_feature = text_feature.unsqueeze(-1).unsqueeze(1).expand(text_batch, img_batch, dim, 1)
                sim = torch.stack([torch.matmul(imgs_feature, text_feat).squeeze().max(-1)[0]  for text_feat in text_feature])
                # sim = torch.stack([torch.matmul(imgs_feature, text_feat).squeeze()  for text_feat in text_feature])
            
            return sim

        if is_conditioned:
            sim = compute_text_conditioned(self.model, image_embed, text_embed)
        else:
            sim = compute_cos_similarity(image_embed, text_embed)
        return sim

    def cosine_ranking(self, images_pre, text_pre):
        with autocast(enabled=True):
            images_feat = self.vis_extract(images_pre, batch_size=self.batch_size, device=self.device)
            text_feat = self.text_extract(text_pre, device=self.device)       
        images_feat = images_feat.float()
        text_feat = text_feat.float()
        t2i_sim = self.compute_cos_sim(image_embed=images_feat, text_embed=text_feat, is_conditioned=self.is_conditioned)
        return t2i_sim
    
    def itm_ranking(self, t_pre, images_pre, batch_size=128, is_grad_crop = False, images=None):
        def rerankingITM(model, t_pre, images_pre, batch_size=14):
            score = []
            tt = t_pre

            
            # d = vis_dataset(images_pre, self.vis_processors["eval"])
            d = vis_dataset_itm(images_pre, self.vis_processors["eval"])
            dataloader = DataLoader(d, batch_size=batch_size, collate_fn=my_collate_fn, num_workers=self.num_workers)
            for sample in dataloader:
                for key in list(sample.keys()):
                    if key!='image_path':
                        sample[key] = sample[key].to(device)
                sample["text_input"] = [tt]*len(sample['image_path'])
                itm_output = model(sample, match_head='itm').detach().cpu()
                itm_output = itm_output.to(float)
                ss = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]
                score = score + ss.tolist()
            return score
        
        return rerankingITM(self.model, t_pre, images_pre, batch_size=batch_size)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='parameter parser')
    parser.add_argument('--model_name', type=str, default='blip2_image_text_matchingpretrain_vitL', 
                        help='support blip_image_text_matching/blip2_image_text_matchingpretrain_vitL/clip_feature_extractor')
    parser.add_argument('--vit', type=str, default='syntext900k_3', help='support syntext900k_3/pretrain_vitL364/syntext100k_words_mlt')
    parser.add_argument('--device', type=str, default='cpu', help='support cpu or single gpu')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')
    parser.add_argument('--dataset', type=str, default='SVT', help='support dataset: SVT/STR/CTR/ICDAR15/total-text/CTW')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--rerank', action='store_true', 
                        help='choose whether rerank with item-item matching, it will lead better performance')
    parser.add_argument('--top_k_ratio', type=float, default=64, help='topk for reranking')
    parser.add_argument('--is_print', action='store_true', help='print ap for each query for further analysis')
    parser.add_argument('--fp16', action='store_true', help='half precision')
    parser.add_argument('--is_conditioned', action='store_true', help='conditional aggregator')
    parser.add_argument('--text_prompt', type=str, default='default', help='text prompt style: 0/1/2')
    
    args = parser.parse_args()
    is_reranking = args.rerank
    top_k_ratio = args.top_k_ratio
    dataset = args.dataset
    device = torch.device(device=args.device) if torch.cuda.is_available() else "cpu"
    is_print = args.is_print
    text_prompt = args.text_prompt
    model_name = args.model_name
    vit = args.vit
    batch_size = args.batch_size
    checkpoint = args.checkpoint
    is_conditioned = args.is_conditioned
    fp16 = args.fp16

    print(top_k_ratio, dataset, model_name, vit, text_prompt)
    
    eval = evaluator(model_name = model_name,
                vitype = vit,
                dataset = dataset,
                is_conditioned = is_conditioned,
                is_reranking = is_reranking,
                is_grad_crop = False,
                checkpoint = checkpoint,
                is_print = is_print,
                is_aug = False,
                text_prompt = text_prompt,
                top_k_ratio = top_k_ratio,
                num_workers = 16,
                batch_size = batch_size,
                device = device,
                fp16=fp16
                )
    eval.pipeline()
                    

    
    