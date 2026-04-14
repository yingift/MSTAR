import torch
from PIL import Image
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
import torch.nn.functional as F

# setup device to use
device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")
from PIL import Image, ImageDraw


model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "ft_siglip800_word_hug_rnn", device=device, is_eval=True)
p = sum(p.numel() for p in model.parameters())
print(p/1024/1024)
state_dict = torch.load('mstar_weights/mstar_word1.pth', map_location='cpu')



model.load_state_dict(state_dict['model'], strict=False)
model.to(device)

raw_image = Image.open("09_10.jpg").convert("RGB")
captions = ["LEON's", 'LEON', 'caption', 'new']

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

print('matching score：')
for txt in captions:
    itm_output = model({"image": img, "text_input": txt}, match_head="itm")
    itm_scores = torch.nn.functional.softmax(itm_output, dim=1)
    print(f'{txt}:{itm_scores[:, 1].item():.3%}')

with torch.no_grad():

    print('cosine similarity:')
    for txt in captions:
        txt = f"a photo of '{txt}'"
        
        tokenized_text = model.tokenizer(
                txt,
                truncation=True,
                max_length=model.max_txt_len,
                return_tensors="pt",
            )
        
        itc_score, _ = model({"image": img, "text_input": txt, "types": 'word'}, match_head='itc')
        print('%s: %.4f, at index %d'%(txt, itc_score, _))

        
        