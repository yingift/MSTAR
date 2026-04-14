"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
from typing import Iterable

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate

def my_collate(samples):
    '''
        "image": image,
        "text_input": caption,
        "image_id": self.img_ids[ann["image_id"]],
        "ocr_list": ocr_list,
        "instance_id": ann["instance_id"]
    '''
    if type(samples[0]['image']) == dict:
        image_key = ('flattened_patches', 'attention_mask')  
        image_list = {key : torch.cat(([s['image'][key] for s in samples]))
            for key in image_key
        }
    else:   
        image_list = torch.cat([d['image'] for d in samples])
    text_input_list = []
    for d in samples:
        text_input_list+=d['text_input'] 
    
    instance_list = [d['instance_id'] for d in samples]
    image_id_list = torch.stack([torch.tensor(d['image_id']) for d in samples])
    
    dic = {
         "image": image_list,
        "text_input": text_input_list,
        
        "image_id": image_id_list,
        "instance_id": instance_list
    }
    other_possible_key = ["ocr_list", "vis_caption", "type"]
    for key in other_possible_key:
        if key in samples[0].keys():
            ocr_str_list = [d[key] for d in samples]
            dic[key] = ocr_str_list        
    return dic

        

class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r")))

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # return default_collate(samples)
        return my_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
