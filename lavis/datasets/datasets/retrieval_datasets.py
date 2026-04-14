"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict
import numpy as np
from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import torch
import random

def pad_ocr_list(str_list, padding='[pad]', length=100):
    if len(str_list)<length:
        mask = [1]*len(str_list)+[0]*(length-len(str_list))
        padded = str_list + [padding]*(length-len(str_list))
    else:
        mask = [1]*length
        padded = str_list[:length]
    return padded
class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class RetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1


    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["image"])
        image1 = Image.open(image_path).convert("RGB")

        captions = ann["text"]
        # word = random.choice(captions)
        # i = random.randint(0, 4)
        # word = word.lower()
        # image2_path = os.path.join(self.vis_root, f'{word}_{i}.jpg')
        # image2 = Image.open(image2_path).convert("RGB")
        
        max_words_num = 64
        image = self.vis_processor(image1).unsqueeze(0)
        # image2 = self.vis_processor(image2)
        # image = torch.stack([image1, image2])
        if type(ann['text']) == list:    
            captions = ann["text"]
            caption = [self.text_processor(cap) for cap in captions]
            # word = [self.text_processor(word)]
            caption = [caption]
            # caption = caption[:max_words_num]
        elif type(ann['caption']) == str:
            caption = self.text_processor(ann['caption'])
        dic = {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "instance_id": ann["instance_id"]
        }
        if "ocr_list" in ann.keys():
            ocr_list = [[self.text_processor(t) for t in tt] for tt in ann["ocr_list"] ]
            dic["ocr_list"] = ocr_list
        if "vis_caption" in ann.keys():
            dic["vis_caption"] = ann['vis_caption']
        if "type" in ann.keys():
            dic["type"] = ann['type']

        return dic


class RetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):

        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        return {"image": image, "index": index}


class VideoRetrievalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["video"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])

        video = self.vis_processor(vpath)
        caption = self.text_processor(ann["caption"])

        # return image, caption, self.img_ids[ann['image_id']]
        return {
            "video": video,
            "text_input": caption,
            "image_id": self.img_ids[ann["video"]],
        }


class VideoRetrievalEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of videos.
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["video"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        vpath = os.path.join(self.vis_root, ann["video"])
        video = self.vis_processor(vpath)

        return {"video": video, "index": index}
