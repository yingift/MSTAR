
import xml.dom.minidom as xmldom
import re



def text_filter(word):
    
    if word.startswith("'") and word.endswith("'"):
        word = word[1:-1]
    word = word.lower()
    no_words = ['.', '#', '!', '&', '$']
    for w in no_words:
        word = word.replace(w, '')
    pattern = [r'&#(169);', r'&(AMP);']
    for p in pattern:
        match = re.search(p, word)
        if match:
            word = match.group(1)
    return word
    
def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    root_node = xml_file.documentElement
    text_queries = {}
    text_size_que = {}
    img_text_num = {}
    for image_node in root_node.getElementsByTagName("image"):
        jpg_name_node = image_node.getElementsByTagName('imageName')[0]
        jpg_name = jpg_name_node.childNodes[0].data.split('/')[1]
        wid = int(image_node.getElementsByTagName('Resolution')[0].getAttribute('x'))
        hei = int(image_node.getElementsByTagName('Resolution')[0].getAttribute('y'))
        taggedRectangles_node = image_node.getElementsByTagName("taggedRectangles")[0]
        text_size = {}
        for taggedRec in taggedRectangles_node.getElementsByTagName("taggedRectangle"):
            t_wid = int(taggedRec.getAttribute('width'))
            t_hei = int(taggedRec.getAttribute('height'))
            size_ratio = t_wid*t_hei/wid/hei
            tq = taggedRec.getElementsByTagName("tag")[0].childNodes[0].data
            tq = text_filter(tq)
            if tq in text_size.keys():
                text_size[tq] = max(text_size[tq], size_ratio)
            else:
                text_size[tq] = size_ratio
        text = list(text_size.keys())
        jpg_name = 'svt_test_'+jpg_name
        img_text_num[jpg_name] = len(text)
        for text_q in text:
            if text_q not in text_queries.keys():
                text_queries[text_q] = [jpg_name]
            else:
                text_queries[text_q].append(jpg_name)
        for text_q, size in text_size.items():
            if text_q not in text_size_que.keys():
                text_size_que[text_q] = {jpg_name:size}
            else:
                text_size_que[text_q][jpg_name] = size
    return text_queries, text_size_que, img_text_num

def update_text_queries(dict1, dict2, mode =  'share'):
    for key, val in dict2.items():
        if mode == 'share':
            if key not in dict1.keys():
                dict1[key] = val
            else:
                dict1[key] = dict1[key]+dict2[key]
        elif mode == 'update':
            if key in dict1.keys():
                dict1[key] = dict1[key]+dict2[key]
    return dict1
def update_text_size_que(dict1, dict2):
    for key, val in dict2.items():
        if key in dict1.keys():
            for k, v in val.items():
                dict1[key][k] = v
        else:
            dict1[key] = val
    return dict1
def update_image_text_num(dict1, dict2):
    for key, val in dict2.items():
        if key in dict1.keys():
             dict1[key] += val
        else:
            dict1[key] = val
    return dict1
def mean(a_list):
    return sum(a_list)/len(a_list)


def load_svt_ann_txt(txt_path, images_caption):
    jpg_dict = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        l = line.strip()
        jpg_name = l.split(':')[0]
        if jpg_name in images_caption:
            text_list = l.split(':')[1].split(',')
            text_list = [text_filter(t) for t in text_list]
            jpg_dict[jpg_name] = text_list
    return jpg_dict

def has_numbers(input_string):
    return bool(re.search(r'\d', input_string))


def calculate_polygon_area(x_coordinates, y_coordinates):
    n = len(x_coordinates)
    area = 0

    for i in range(n):
        j = (i + 1) % n
        area += (x_coordinates[i] * y_coordinates[j]) - (x_coordinates[j] * y_coordinates[i])

    return abs(area) / 2


