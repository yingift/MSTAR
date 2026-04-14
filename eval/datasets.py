from PIL import Image
import numpy as np
import os
import json
from os.path import join
from os import listdir
import re
from utils import *

def load_SVT_ori(SVT_path, is_readimage=True):
    '''
        SVT_path: dir of SVT, organized as follows:
            SVT_path:
                -img
                -test.xml
                -train.xml
        return (image&title)s(list) and text queries(dict)
    '''
    test_text_queries, test_text_size_que, test_img_text_num = parse_xml(join(SVT_path, 'test.xml'))
    text_queries_ = test_text_queries
    text_size_que = test_text_size_que
    img_text_num = test_img_text_num

    #read images
    img_dir = join(SVT_path, 'test_images')
    images = []
    images_caption = []
    for jpg in listdir(img_dir):
        jpg_name = 'svt_test_'+jpg
        
        if jpg_name in img_text_num.keys():
            jpg_path = join(img_dir, jpg)
            if is_readimage:
                images.append(Image.open(jpg_path).convert("RGB"))
            else:
                images.append(jpg_path)            
            images_caption.append(jpg_name)
    text_queries =text_queries_
    return images, images_caption, text_queries, text_size_que, img_text_num

def load_SVT(SVT_path, is_readimage=True):
    images, images_caption, text_queries, text_size_que, img_text_num = \
                                    load_SVT_ori(SVT_path, is_readimage=is_readimage)
    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if jpg in jpg_list else 0 for jpg in images_caption]
    img_text_num = [img_text_num[jpg] for jpg in images_caption]
    return images, images_caption, text_queries, text_size_que, img_text_num   
    
def load_STR_ori(STR_path, is_readimage=True):
    '''
        STR_path: dir of STR, organized as follows:
            STR_path:
                -imgDatabase
                -data.mat
                -README(optional)
        return (image&title)s(list) and text queries(dict)
    '''
    #read images
    img_dir = join(STR_path, 'imgDatabase')
    images = []
    images_caption = []
    for jpg in listdir(img_dir):
        jpg_path = join(img_dir, jpg)
        if is_readimage:
            images.append(Image.open(jpg_path).convert("RGB"))
        else:
            images.append(jpg_path)

        images_caption.append(jpg)
    
    #read text queries
    mat = scipy.io.loadmat(join(STR_path, 'data.mat'))['data'][0]
    text_queries = {}
    for m in mat:
        text = m[0][0][0][0]
        text = text_filter(text)
        images_titles = [i[0][0] for i in m[1]]
        text_queries[text] = images_titles
    return images, images_caption, text_queries

def load_STR(STR_path, is_readimage=True):
    images, images_caption, text_queries = load_STR_ori(STR_path, is_readimage)
    
    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]    
    return images, images_caption, text_queries

def in_filter_word(text,chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '"):
    text = text.lower()
    char_list = [c for c in text if c in chars]
    text =  "".join(char_list)
    text = re.sub(r'\s+', ' ', text)
    if len(text)>=2:
        text = text[1:] if ' ' == text[0] else text #filter str like ' a' 
        text = text[:-1] if ' ' == text[-1] else text #filter str like 'a ' 
    return text

def load_CTR_ori(CTR_path, is_readimage=True, is_set_optional=True, max_num_jpg=10000, img_idx=None):
    '''
        CTR_path: dir of CTR, organized as follows:
            CTR_path:
                -gts
                -images
                -queries.txt
        return (image&caption)s(list) and text queries(dict)
    '''
    
    img_idx = list(range(max_num_jpg)) if img_idx==None else img_idx
    #read images
    img_dir = join(CTR_path, 'images')
    images = []
    images_caption = []
    for i, jpg in enumerate(sorted(listdir(img_dir))):
        if i not in img_idx:
            continue
        jpg_path = join(img_dir, jpg)
        if is_readimage:
            images.append(Image.open(jpg_path).convert("RGB"))
        else:
            images.append(jpg_path)
        images_caption.append(jpg)
    
    #read text queries
    text_dir = join(CTR_path, 'gts')
    
    text_queries = {}
    img_text_num = {}
    for i, gt in enumerate(sorted(listdir(text_dir))):
        if i not in img_idx:
            continue
        gt_path = join(text_dir, gt)
        jpg_path = os.path.splitext(gt)[0]+'.jpg'
        with open(gt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            img_text_num[jpg_path] = len(lines)
            for line in lines:
                text_q = in_filter_word(line.strip())
                if text_q in text_queries.keys():
                    text_queries[text_q].append(jpg_path)
                else:
                    text_queries[text_q] = [jpg_path]
    return images, images_caption, text_queries, img_text_num
                    
def load_CTR(CTR_path, is_readimage=True, is_set_optional=True):
    images, images_caption, text_queries, img_text_num = \
                                                    load_CTR_ori(CTR_path, is_readimage, is_set_optional)
    img_text_num = [img_text_num[jpg] for jpg in images_caption]

    with open(join(CTR_path, 'queries.txt'), 'r', encoding='utf-8') as f:
        # optional_queries = [line.strip() for line in f]
        optional_queries = [in_filter_word(line.strip()) for line in f]
        
    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]    
    
    text_queries = {key:val for key, val in text_queries.items() if key in optional_queries}
    # for text_q, jpg_list in text_queries.items():
    #     if text_q in optional_queries:
    #         text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]
    if not is_set_optional: 
        tq = sorted(text_queries.items(), key=lambda item:sum(item[1]), reverse=True)
        lens = [sum(c) for t, c in tq]
        print([(t[0], l) for l, t in zip(lens, list(tq))])
        tq = tq[:lens.index(3)]
        text_queries = dict(tq)
    return images, images_caption, text_queries, img_text_num
    
def load_CSVTR(CSVTR_path, is_readimage=True):
    '''
        CSVTR_path: dir of CSVTR, organized as follows:
            CSVTR_path:
                -query1
                    -image1.jpeg
                    -image2.jpeg
                -query2
                    -image1.jpeg
                    -image2.jpeg
        return (image&caption)s(list) and text queries(dict)
    '''
    #read images
    images = []
    images_caption = []
    for sub_dir in listdir(CSVTR_path):
        if os.path.isdir(join(CSVTR_path, sub_dir)):
            for jpeg in listdir(join(CSVTR_path, sub_dir)):
                jpeg_path = join(CSVTR_path, sub_dir, jpeg)
                if is_readimage:
                    images.append(Image.open(jpeg_path).convert("RGB"))
                else:
                    images.append(jpeg_path)
                images_caption.append(sub_dir+jpeg)
                
    #read text queries
    text_queries = {}
    for sub_dir in listdir(CSVTR_path):
       if os.path.isdir(join(CSVTR_path, sub_dir)):
           jpg_list =[sub_dir+jpeg for jpeg in listdir(join(CSVTR_path, sub_dir))]
           text_queries[sub_dir] = [1 if cap in jpg_list else 0 for cap in images_caption]
    return images, images_caption, text_queries

def load_ICDAR15_ori(path, is_readimage=True):
    test_img_dir = join(path, 'ch4_test_images')
    test_ann_dir = join(path, 'Challenge4_Test_Task1_GT')
    images, images_caption = [], []
    for jpg in listdir(test_img_dir):
        jpg_path = join(test_img_dir, jpg)
        jpg = jpg
        if is_readimage:
            images.append(Image.open(jpg_path).convert("RGB"))
        else:
            images.append(jpg_path)
        images_caption.append(jpg)
    text_queries  = {}
    text_size_que = []
    
    img_text_num = {}
    for ann in listdir(test_ann_dir):
        if not ann.endswith(".txt"):
            continue
        ann_path = join(test_ann_dir, ann)
        jpg_name = ann.split('.')[0].split('gt_')[1] + '.jpg'
        jpg_name = jpg_name
        index = images_caption.index(jpg_name)
        # whole_img_size = images[index].size[0]*images[index].size[1]
        with open(ann_path, 'r') as f:
            lines = f.readlines()
            img_text_num[jpg_name] = len(lines)
            for t in lines:
                if len(t) == 1:
                    continue
                poly = t.strip().split(',')[:-1]
                poly = poly[:len(poly)//2*2]
                try:
                    x = np.array([int(a) for i,a in enumerate(poly) if i%2==0])
                    y = np.array([int(a) for i,a in enumerate(poly) if i%2==1])
                except:
                    print(jpg_name, poly)
                    exit(0)
                area = calculate_polygon_area(x, y)
                # ratio = area/whole_img_size
                words = text_filter(t.strip().split(',')[-1])
                for word in words.split(' '):
                    if word == 'ALE':
                        continue
                    if len(word)<=2:
                        continue
                    if word in text_queries.keys():
                        if jpg_name not in text_queries[word]:
                            text_queries[word].append(jpg_name)
                    else:
                        text_queries[word] = [jpg_name]
                    # text_size_que.append([ratio, jpg_name, word])
    return images, images_caption, text_queries, text_size_que, img_text_num
                                  
def load_ICDAR15(path, is_readimage=True):
    images, images_caption, text_queries, text_size_que, img_text_num = load_ICDAR15_ori(path, is_readimage)    
    word_len = [(k,len(v)) for k,v in text_queries.items()]
    sorted_word_len = sorted(word_len, reverse=True, key=lambda x: x[1])[:100]
    filtered_arr = [x[0] for x in sorted_word_len]
    text_queries = dict([(i, text_queries[i]) for i in filtered_arr])
    
    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]
    img_text_num  = [img_text_num[jpg] for jpg in images_caption]
        
    return images, images_caption, text_queries, text_size_que, img_text_num

def load_pstr_ori(path, is_readimage=True):
    test_img_dir = join(path, 'images')
    test_ann_dir = join(path, 'query_image_dict.npy')
    images, images_caption = [], []
    for jpg in listdir(test_img_dir):
        jpg_path = join(test_img_dir, jpg)
        jpg = jpg
        if is_readimage:
            images.append(Image.open(jpg_path).convert("RGB"))
        else:
            images.append(jpg_path)
        images_caption.append(jpg)
    jpg_lines_dict  = {}
    jpg_words_dict  = {}
    text_size_que = []
    img_text_num = []
    text_queries = np.load(test_ann_dir, allow_pickle=True).item()
    return images, images_caption, text_queries, text_size_que, img_text_num

def load_pstr(path, is_readimage=True):
    images, images_caption, text_queries, text_size_que, img_text_num = load_pstr_ori(path, is_readimage)    
    
    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]
    return images, images_caption, text_queries

def load_CTW_ori(path, is_readimage=True):
    test_img_dir = join(path, 'test_images')
    test_ann_dir = join(path, 'gt_ctw1500')
    images, images_caption = [], []
    for jpg in listdir(test_img_dir):
        jpg_path = join(test_img_dir, jpg)
        jpg_name = jpg
        if is_readimage:
            images.append(Image.open(jpg_path).convert("RGB"))
        else:
            images.append(jpg_path)
        images_caption.append(jpg_name)
    text_queries  = {}
    img_text_num = {}
    
    for ann in listdir(test_ann_dir):
        if not ann.endswith(".txt"):
            continue
        ann_path = join(test_ann_dir, ann)
        jpg_name = ann.split('.')[0].split('000')[1] + '.jpg'
        jpg_name = jpg_name
        text_size_que = []
        index = images_caption.index(jpg_name)
        # whole_img_size = images[index].size[0]*images[index].size[1]
        with open(ann_path, 'r') as f:
            lines = f.readlines()
            img_text_num[jpg_name] = len(lines)
            for t in lines:
                if len(t) == 1:
                    continue
                poly = t.strip().split('####')[0].split(',')[:-1]
                poly = poly[:len(poly)//2*2]
                try:
                    x = np.array([int(a) for i,a in enumerate(poly) if i%2==0])
                    y = np.array([int(a) for i,a in enumerate(poly) if i%2==1])
                except:
                    print(jpg_name, poly)
                    exit(0)
                area = calculate_polygon_area(x, y)
                # ratio = area/whole_img_size
                words = text_filter(t.strip().split('####')[1])
                for word in words.split(' '):
                    if len(word)<=2:
                        continue
                    if word in text_queries.keys():
                        if jpg_name not in text_queries[word]:
                            text_queries[word].append(jpg_name)
                    else:
                        text_queries[word] = [jpg_name]
                    # text_size_que.append([ratio, jpg_name, word])
    
    return images, images_caption, text_queries, text_size_que, img_text_num
                    
def load_CTW(path, is_readimage=True):
    images, images_caption, text_queries,text_size_que, img_text_num = \
                            load_CTW_ori(path, is_readimage)

    img_text_num  = [img_text_num[jpg] for jpg in images_caption]
    
    word_len = [(k,len(v)) for k,v in text_queries.items()]
    sorted_word_len = sorted(word_len, reverse=True, key=lambda x: x[1])[:100]
    filtered_arr = [x[0] for x in sorted_word_len]
    
    text_queries = dict([(i, text_queries[i]) for i in filtered_arr])

    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]
    return images, images_caption, text_queries,text_size_que, img_text_num
    
def load_total_text_ori(path, is_readimage=True):
    test_img_dir = join(path, 'Images', 'Test')
    train_img_dir = join(path, 'Images', 'Train')
    train_ann_dir = join(path, 'txt_format', 'Train')
    test_ann_dir = join(path, 'txt_format', 'Test')
    images, images_caption = [], []
    for jpg in listdir(test_img_dir):
        jpg_path = join(test_img_dir, jpg)
        if is_readimage:
            images.append(Image.open(jpg_path).convert("RGB"))
        else:
            images.append(jpg_path)
        jpg = jpg
        images_caption.append(jpg)
    text_queries  = {}
    pattern = r", transcriptions: \[u\'(.*?)\'\]"
    pattern1 = r''', transcriptions: \[u\"(.*?)\"\]'''
    img_text_num = {}
    for ann in listdir(test_ann_dir):
        if not ann.endswith(".txt"):
            continue
        ann_path = join(test_ann_dir, ann)
        jpg_name = ann.split('.')[0].split('poly_gt_')[1] + '.jpg'
        jpg_name = jpg_name
        text_size_que = []
        index = images_caption.index(jpg_name)
        # whole_img_size = images[index].size[0]*images[index].size[1]
        with open(ann_path, 'r') as f:
            lines = f.readlines()
            img_text_num[jpg_name] = len(lines)
            for t in lines:
                if len(t) == 1:
                    continue
                poly = t.strip().split(',')[:-1]
                poly = poly[:2]
                try:
                    x = poly[0].split('[[')[1].split(']]')[0].split(' ')
                    y = poly[1].split('[[')[1].split(']]')[0].split(' ')
                    x = np.array([int(a) for a in x if a!=''])
                    y = np.array([int(a) for a in y if a!=''])                    
                except:
                    print(jpg_name, poly, x, y)
                    exit(0)
                area = calculate_polygon_area(x, y)
                # ratio = area/whole_img_size
                res = re.findall(pattern, t)
                if not res:
                    res = re.findall(pattern1, t)
                words = text_filter(res[0])
                for word in words.split(' '):
                    if len(word)<=2:
                        continue
                    if word in text_queries.keys():
                        if jpg_name not in text_queries[word]:
                            text_queries[word].append(jpg_name)
                    else:
                        text_queries[word] = [jpg_name]
    
    return images, images_caption, text_queries, text_size_que, img_text_num

def load_total_text(path, is_readimage=True):
    images, images_caption, text_queries, text_size_que, img_text_num = \
                        load_total_text_ori(path, is_readimage)
    img_text_num  = [img_text_num[jpg] for jpg in images_caption]
    
    # length = np.array([len(v) for k,v in text_queries.items()])
    word_len = [(k,len(v)) for k,v in text_queries.items()]
    sorted_word_len = sorted(word_len, reverse=True, key=lambda x: x[1])[:60]
    filtered_arr = [x[0] for x in sorted_word_len]
    text_queries = dict([(i, text_queries[i]) for i in filtered_arr])
    
    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]
    return images, images_caption, text_queries, text_size_que, img_text_num
     
def load_MQTR(path, is_readimage=True):
    image_path = join(path, 'images')
    images_caption = []
    images = []
    for jpg in listdir(image_path):
        if jpg.endswith('.jpg') or jpg.endswith('.png'):
            jpg_path = join(image_path, jpg)
        if is_readimage:
            images.append(Image.open(jpg_path).convert("RGB"))
        else:
            images.append(jpg_path)
        images_caption.append(jpg)
    with open(os.path.join(path, 'text_queries.json'), 'r') as f:
        text_queries = json.load(f)
    for text_q, jpg_list in text_queries.items():
        text_queries[text_q] = [1 if cap in jpg_list else 0 for cap in images_caption]
    return images, images_caption, text_queries

def load_dataset(data, is_readimage=True):
    if data == 'CSVTR':
        images, images_caption, text_queries = load_CSVTR('datasets/ChineseRetrievalCollection/', is_readimage=is_readimage)
    elif data == 'STR':
        images, images_caption, text_queries = load_STR('datasets/IIIT_STR_V1.0/', is_readimage=is_readimage)
    elif data == 'CTR':
        images, images_caption, text_queries, _ = load_CTR('datasets/cocotext_top500_retrieval/', is_readimage=is_readimage)
    elif data == 'SVT':
        images, images_caption, text_queries, _, __ = load_SVT('datasets/SVT/',  is_readimage=is_readimage)
    elif data == 'total-text':
        images, images_caption, text_queries, _, __ = load_total_text('datasets/TotalText/',  is_readimage=is_readimage)
    elif data == 'CTW':
        images, images_caption, text_queries, _ , __= load_CTW('datasets/CTW/',  is_readimage=is_readimage)
    elif data == 'ICDAR15':
        images, images_caption, text_queries, _ , __= load_ICDAR15('datasets/ICDAR2015/',  is_readimage=is_readimage)

    elif data == 'pMQTR':
        images, images_caption, text_queries = load_MQTR('datasets/MQTR/phrase', is_readimage=is_readimage)
    elif data == 'sMQTR':
        images, images_caption, text_queries = load_MQTR('datasets/MQTR/semantic', is_readimage=is_readimage)
    elif data == 'cMQTR':
        images, images_caption, text_queries = load_MQTR('datasets/MQTR/combined', is_readimage=is_readimage)
    elif data == 'wMQTR':
        images, images_caption, text_queries = load_MQTR('datasets/MQTR/word', is_readimage=is_readimage)
    elif data == 'pstr':
        images, images_caption, text_queries = load_pstr("datasets/pstr/", is_readimage=is_readimage)
    return images, images_caption, text_queries


if __name__ == "__main__":
    is_readimage=False
    images, images_caption, text_queries = load_dataset('sMSTR', is_readimage=is_readimage)
    import pdb; pdb.set_trace()
