import os
import numpy as np
import json
from PIL import Image
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

########sequence text generation###########
path = "./data"
folder_names = ['{:04d}'.format(i) for i in range(1,3810) ]
texts = {}
for folder in tqdm(folder_names):
    texts[folder] = []
    path_to_json_files = os.path.join(path, "annotations", folder)
    json_file_names = [filename for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]
    for i, json_file_name in enumerate(json_file_names):
        sentences = set()
        fp = os.path.join(path_to_json_files, json_file_name)
        with open(fp) as json_file:
            json_text = json.load(json_file)
            nhumans = []
            for nums in json_text['humans']:
                hid = nums['number']
                for p in nums['parts']:
                    if p['obj_class']:
                        sentences.add(' '.join(p['name'].split('_'))+ ' ' +\
                                          ' '.join(p['verb'].split('_'))+ ' ' +\
                                          ' '.join(p['obj_class'].split('_')))
                    else:
                        sentences.add(' '.join(p['name'].split('_'))+ ' ' +\
                                          ' '.join(p['verb'].split('_')))
        texts[folder].append(sentences)


for folder, par in texts.items():
    savepath = os.path.join(path, "descriptions", folder)
    Path(savepath).mkdir(parents=True, exist_ok=True)
    path_to_json_files = os.path.join(path, "annotations", folder)
    json_file_names = [filename[:-5] for filename in os.listdir(path_to_json_files) if filename.endswith('.json')]
    for i, file_name in enumerate(json_file_names):
        with open(os.path.join(savepath, file_name + '.txt'), 'w') as f:
            f.writelines('\n'.join(par[i]))     
