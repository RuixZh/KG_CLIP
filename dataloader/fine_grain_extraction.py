import os
import glob
import numpy as np
import json
import torch
from PIL import Image
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


class FineGrainExtraction(object):
    def __init__(self, annotation_path="./data/annotations/", frame_path="./data/frames/", filelist=["./data/list/body_part.txt", "./data/list/object.txt"], outdir="./data/fine-grain/"):
        self.annotation_path = annotation_path
        self.frame_path = frame_path
        self.outdir = outdir
        self.filelist = filelist

        self._get_text_tensor()
        self._create()
        self._get_extraction()

    def _create(self):
        self.nb_dict = {}
        self.nb_dict['person'] = 0
        self.save_path = {}
        savepath = os.path.join(self.outdir, 'person')
        path = Path(savepath)
        path.mkdir(parents=True, exist_ok=True)
        self.save_path['person'] = savepath
        for f in self.object_list:
            self.nb_dict[f] = 0
            savepath = os.path.join(self.outdir, f)
            path = Path(savepath)
            path.mkdir(parents=True, exist_ok=True)
            self.save_path[f] = savepath

    def _get_text_tensor(self):
        object_list = []
        for filename in self.filelist:
            with open(filename, 'r') as f:
                for l in f.readlines():
                    if len(l.strip()) > 0:
                        object_list.append(l.strip())
        self.object_list = list(set(object_list))
        self.video_list = [os.path.basename(i) for i in glob.glob(self.annotation_path + "*")]

    def _get_extraction(self):
        for video_id in tqdm(self.video_list):

            json_files, frame_files = self._load_annoatation(video_id)

            for frame_id, (json_data, img) in enumerate(zip(json_files, frame_files)):
                for h, humans in json_data.items():
                    for i, hid in enumerate(humans):
                        # # human box
                        x, y, w, h = hid['box']
                        x, y, w, h = int(max(0, x)), int(max(0, y)), int(max(0, w)), int(max(0, h))
                        person_patch = (img[y:h, x:w] * 255.).astype(np.uint8)
                        person_patch = Image.fromarray(person_patch).convert("RGB")
                        person_patch.save(os.path.join(self.save_path['person'], str(self.nb_dict['person'] )+'.jpg'))
                        self.nb_dict['person'] += 1

                        for pid in hid["parts"]:
                            x, y, w, h = pid["box"]
                            x, y, w, h = int(max(0, x)), int(max(0, y)), int(max(0, w)), int(max(0, h))
                            body_patch = (img[y:h, x:w] * 255.).astype(np.uint8)
                            body_patch = Image.fromarray(body_patch).convert("RGB")

                            body_patch.save(os.path.join(self.save_path[pid["name"]], str(self.nb_dict[pid["name"]])+'.jpg'))
                            self.nb_dict[pid["name"]] += 1

                            if pid["obj_class"] is not None:
                                x, y, w, h = pid["obj_bbox"]
                                x, y, w, h = int(max(0, x)), int(max(0, y)), int(max(0, w)), int(max(0, h))
                                obj_patch = (img[y:h, x:w] * 255.).astype(np.uint8)
                                obj_patch = Image.fromarray(obj_patch).convert("RGB")

                                obj_patch.save(os.path.join(self.save_path[pid["obj_class"]], str(self.nb_dict[pid["obj_class"]] )+'.jpg'))
                                self.nb_dict[pid["obj_class"]] += 1

    def _load_annoatation(self, video_id):
        json_file_names = [filename for filename in os.listdir(os.path.join(self.annotation_path, video_id)) if filename.endswith('.json')]
        json_files = []
        frame_files = []
        for jsonf in json_file_names:
            framef = jsonf.split('.')[0]+'.jpg'
            try:
                with open(os.path.join(self.annotation_path, video_id, jsonf)) as f:
                    json_data = json.load(f)
                    json_files.append(json_data)
            except FileNotFoundError :
                print('ERROR: Could not find annotation file of video "{}"'.format( video_id))
                raise
            try:
                img = self._load_frame(os.path.join(self.frame_path, video_id, framef))
                frame_files.append(img)
            except FileNotFoundError :
                print('ERROR: Could not find frame of video "{}"'.format(video_id))
                raise

        return json_files, frame_files

    def _load_frame(self, img_path, resize=None, pil=False):
        image = Image.open(img_path).convert("RGB")
        if resize is not None:
            image = image.resize((resize, resize))
        if pil:
            return image
        image = np.asarray(image).astype(np.float32) / 255.
        return image


def main():
    FineGrainExtraction()

if __name__ == '__main__':
    main()
