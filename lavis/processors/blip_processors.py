"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re

from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import random
# from transformers import  Pix2StructProcessor


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])
        # caption = f"'{caption}'" + " [SEP] "+ " ".join(list(caption))
        # caption = f"{caption}''"
        

        return caption


@registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor):
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)

        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question
class RandomPadding(object):
    def __call__(self, image):
        padding_left = random.randint(0, 30)
        padding_right = random.randint(0, 30)
        padding_top = random.randint(0, 30)
        padding_bottom = random.randint(0, 30)
        return transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

class CustomPadding(object):
    def __call__(self, image):
        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio > 1.:
            new_width = width
            new_height = int(width/1.)
        elif aspect_ratio < 1/1.:
            new_height = height
            new_width = int(height/1.)
        else:
            return image
        # padding_range = (0, 40)
        # ran_pad = random.randint(padding_range[0], padding_range[1])
        padding_width = max(new_width - width, 0)
        padding_height = max(new_height - height, 0)
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        return transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

class CustomResize(object):
    def __init__(self, resize=400):
        self.resize = resize
    def __call__(self, image):
        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_height = self.resize
            new_width = int(new_height*aspect_ratio)
        else:
            new_width = self.resize
            new_height = int(new_width/aspect_ratio)
        return transforms.Resize((new_width, new_height))

            
@registry.register_processor("blip_image_train")
class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                # CustomResize(4),
                # RandomPadding(),
                # CustomPadding(),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                # transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.8)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip2_rn_image_train")
class Blip2ImageRnTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                # CustomResize(4),
                # RandomPadding(),
                CustomPadding(),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.8)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )
        

@registry.register_processor("blip2_rn_image_train_fine")
class Blip2ImageRnFineProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=384, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                # CustomResize(4),
                # RandomPadding(),
                CustomPadding(),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.8)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )

@registry.register_processor("Pixel2structPre")
class Pixel2structPre(BlipImageBaseProcessor):
    def __init__(
        self, 
        preprocess_fn,
        max_patches=2048,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt',
        mean=None, 
        std=None,
    ):
        super().__init__(mean=mean, std=std)
        self.preprocess_fn = preprocess_fn
        self.max_patches = max_patches
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.return_tensors = return_tensors
        self.transform = transforms.Compose(
            [
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                    ],
                ),
            ]
        )

    def __call__(self, item):
        dd = self.transform(item)
        item = self.preprocess_fn(images=dd, 
                                add_special_tokens=self.add_special_tokens,
                                padding=self.padding,
                                return_tensors='pt', 
                                max_patches=self.max_patches,
                            )
        return item

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        preprocessor_name = cfg.get("preprocessor_name", 'preprocessname')
        preprocess_fn = Pix2StructProcessor.from_pretrained(preprocessor_name)    
        preprocess_fn.image_processor.is_vqa=False
        

        max_patches = cfg.get("max_patches", None)
        add_special_tokens = cfg.get("add_special_tokens", None)
        padding = cfg.get("padding", None)
        return_tensors = cfg.get("return_tensors", None)
        
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            preprocess_fn=preprocess_fn,
            max_patches=max_patches,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors=return_tensors,
            mean=mean, 
            std=std,
        )
        

@registry.register_processor("Pixel2structPre_eval")
class Pixel2structPreEval(BlipImageBaseProcessor):
    def __init__(
        self, 
        preprocess_fn,
        image_size,
        max_patches=2048,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt',
        mean=None, 
        std=None,
    ):
        super().__init__(mean=mean, std=std)
        self.preprocess_fn = preprocess_fn
        self.max_patches = max_patches
        self.add_special_tokens = add_special_tokens
        self.padding = padding
        self.return_tensors = return_tensors
        self.transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1]))
        ])
        

    def __call__(self, item):
        item = self.transform(item)
        item = self.preprocess_fn(images=item, 
                                add_special_tokens=self.add_special_tokens,
                                padding=self.padding,
                                return_tensors='pt', 
                                max_patches=self.max_patches,
                            )
        return item

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        preprocessor_name = cfg.get("preprocessor_name", 'preprocessname')
        preprocess_fn = Pix2StructProcessor.from_pretrained(preprocessor_name)    
        preprocess_fn.image_processor.is_vqa=False
        

        max_patches = cfg.get("max_patches", None)
        add_special_tokens = cfg.get("add_special_tokens", None)
        padding = cfg.get("padding", None)
        return_tensors = cfg.get("return_tensors", None)
        image_size = cfg.get("image_size", [400, 600])
        
        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            preprocess_fn=preprocess_fn,
            image_size=image_size,
            max_patches=max_patches,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors=return_tensors,
            mean=mean, 
            std=std,
        )


@registry.register_processor("blip_image_eval")
class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                # CustomPadding(),
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)

@registry.register_processor("blip_image_rn_eval")
class BlipImageRnEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                CustomPadding(),
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 384)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)
    

class RandomPadding(object):
    def __call__(self, image):
        width, height = image.size
        aspect_ratio = width / height
        if aspect_ratio > 1.2:
            new_width = width
            new_height = int(width/1.2)
        elif aspect_ratio < 1/1.2:
            new_height = height
            new_width = int(height/1.2)
        else:
            return image

        padding_width = max(new_width - width, 0)
        padding_height = max(new_height - height, 0)
        padding_left = padding_width // 2
        padding_right = padding_width - padding_left
        padding_top = padding_height // 2
        padding_bottom = padding_height - padding_top
        return transforms.functional.pad(image, (padding_left, padding_top, padding_right, padding_bottom), fill=0)
    

@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(
        self, image_size=364, mean=None, std=None, min_scale=0.5, max_scale=1.0
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                # CustomPadding(),
                # transforms.Resize(420),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                # transforms.RandomRotation(15),
                # transforms.RandomHorizontalFlip(),
                RandomAugment(
                    2,
                    5,
                    isPIL=True,
                    augs=[
                        "Identity",
                        "AutoContrast",
                        "Brightness",
                        "Sharpness",
                        "Equalize",
                        "ShearX",
                        "ShearY",
                        "TranslateX",
                        "TranslateY",
                        "Rotate",
                        # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                    ],
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 364)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.8)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )