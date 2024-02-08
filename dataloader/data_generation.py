import os
import numpy as np
import json
import torch
from PIL import Image
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
# def text_prompt(self):
#     text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
#                 f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
#                 f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
#                 f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
#                 f"The man is {{}}", f"The woman is {{}}"]
#     text_dict = {}
#     num_text_aug = len(text_aug)
#
#     for ii, txt in enumerate(text_aug):
#         text_dict[ii] = torch.cat([clip.tokenize(txt.format(c)) for c in self.class_dict])
#
#     classes = torch.cat([v for k, v in text_dict.items()])
#
#     return classes


class TripleGeneration(object):
    def __init__(self, video_list, annotation_path, frame_path, record=None, exist_folders=None, training=True, outdir="./data/fine-grain/"):
        self.annotation_path = annotation_path
        self.frame_path = frame_path
        self.training = training
        self.outdir = outdir

        self._get_text_tensor()

        if record:
            self._get_triple(record)
        else:
            for r in tqdm(video_list):
                vid = r.video_id
                if vid in exist_folders:
                    continue
                else:
                    savepath = self.outdir + vid + '/'
                    path = Path(savepath)
                    path.mkdir(parents=True, exist_ok=True)
                    self._get_triple(r, savepath)

    def _get_text_tensor(self):
        word_dict = {}
        try:
            with open("./data/word_list.txt", 'r') as f:
                for i, l in enumerate(f.readlines()):
                    word_dict[' '.join(l.strip().split('_'))] = i
                    # word_dict_reverse[i] = l.strip()
        except FileNotFoundError :
            print('ERROR: Could not find word list file')
            raise
        self.word_dict = word_dict
        # text_file = os.path.join(self.outdir, "text_tensor.pth")
        # if not os.path.isfile(text_file):
        #     txt_tensor = clip.tokenize(word_dict)
        #     path = Path(self.outdir)
        #     path.mkdir(parents=True, exist_ok=True)
        #     torch.save(txt_tensor, text_file)

    def _get_triple(self, record, savepath):
        video_id, label = record.video_id, record.label
        img_tensor = []
        kg_index = defaultdict(list)
        nb_dict = defaultdict(lambda: 0)

        json_files, frame_files = self._load_annoatation(video_id)
        img_idx = 0
        frame_dict = {}
        for frame_id, (json_data, img) in enumerate(zip(json_files, frame_files)):
            image = (img * 255.).astype(np.uint8)
            image = Image.fromarray(image).convert("RGB")
            # image = self.transform(image).unsqueeze(0)
            # img_tensor.append(image)
            image.save(os.path.join(savepath, str(img_idx)+'.jpg'))
            frame_dict[frame_id] = img_idx
            img_idx += 1
            if frame_id > 0:
                # (frame1, followed_by, frame2) --  triple['img-text-img']
                kg_index['img-txt-img'].append((frame_dict[frame_id-1], self.word_dict["followed by"], frame_dict[frame_id]))
                nb_dict['nb_triple'] += 1
            if self.training:
                # (frame_img, belong_to, action_text) -- triple['image-text-text']
                kg_index['img-txt-txt'].append((frame_dict[frame_id], self.word_dict["belong to"], self.word_dict[label]))

            for h, humans in json_data.items():
                human_dict = {}
                for i, hid in enumerate(humans):
                    # human box
                    x, y, w, h = hid['box']
                    x, y, w, h = int(max(0, x)), int(max(0, y)), int(max(0, w)), int(max(0, h))
                    person_patch = (img[y:h, x:w] * 255.).astype(np.uint8)
                    person_patch = Image.fromarray(person_patch).convert("RGB")
                    # person_patch = self.transform(person_patch).unsqueeze(0)
                    # img_tensor.append(person_patch)
                    person_patch.save(os.path.join(savepath, str(img_idx)+'.jpg'))
                    human_dict[i] = img_idx
                    img_idx += 1
                    # (person_img, instance_of, frame_img) -- triple['image-text-image']
                    kg_index['img-txt-img'].append((human_dict[i], self.word_dict["instance of"], frame_dict[frame_id]))
                    nb_dict['nb_triple'] += 1

                    for pid in hid["parts"]:
                        x, y, w, h = pid["box"]
                        x, y, w, h = int(max(0, x)), int(max(0, y)), int(max(0, w)), int(max(0, h))
                        body_patch = (img[y:h, x:w] * 255.).astype(np.uint8)
                        body_patch = Image.fromarray(body_patch).convert("RGB")
                        # body_patch = self.transform(body_patch).unsqueeze(0)
                        # img_tensor.append(body_patch)
                        body_patch.save(os.path.join(savepath, str(img_idx)+'.jpg'))
                        body_idx = img_idx
                        img_idx += 1
                        body_name = ' '.join(pid["name"].split('_'))
                        verb = ' '.join(pid["verb"].split('_'))

                        # (body_img, is_about, body_txt) -- triple['image-text-text']
                        kg_index['img-txt-txt'].append((body_idx, self.word_dict["is about"], self.word_dict[body_name]))
                        nb_dict['nb_triple'] += 1

                        # (verb, subject of, body_img) -- triple['text-text-image']
                        kg_index['txt-txt-img'].append((self.word_dict[verb], self.word_dict["subject of"], body_idx))
                        nb_dict['nb_triple'] += 1

                        # (verb, subject of, body_txt) -- triple['text-text-text']
                        kg_index['txt-txt-txt'].append((self.word_dict[verb], self.word_dict["subject of"], self.word_dict[body_name]))
                        nb_dict['nb_triple'] += 1

                        # prompt += pid["verb"] + ' '

                        if pid["obj_class"] is not None:
                            x, y, w, h = pid["obj_bbox"]
                            x, y, w, h = int(max(0, x)), int(max(0, y)), int(max(0, w)), int(max(0, h))
                            obj_patch = (img[y:h, x:w] * 255.).astype(np.uint8)
                            obj_patch = Image.fromarray(obj_patch).convert("RGB")
                            # obj_patch = self.transform(obj_patch).unsqueeze(0)
                            # img_tensor.append(body_patch)
                            obj_patch.save(os.path.join(savepath, str(img_idx)+'.jpg'))
                            obj_idx = img_idx
                            img_idx += 1
                            obj_class = ' '.join(pid["obj_class"].split('_'))
                            # (obj_img, is_about, obj_txt) --  triple['image-text-text']
                            kg_index['img-txt-txt'].append((obj_idx, self.word_dict["is about"], self.word_dict[obj_class]))
                            nb_dict['nb_triple'] += 1

                            # (verb, object of, obj_img) --  triple['text-text-image']
                            kg_index['txt-txt-img'].append((self.word_dict[verb], self.word_dict["object of"], obj_idx))
                            nb_dict['nb_triple'] += 1

                            # (verb, object of, obj_txt) --  triple['text-text-text']
                            kg_index['txt-txt-txt'].append((self.word_dict[verb], self.word_dict["object of"], self.word_dict[obj_class]))
                            nb_dict['nb_triple'] += 1

                            # prompt += pid["obj_class"]

                        # prompt = clip.tokenize(prompt)
                        # prompt_emb = model.encode_text(prompt).float()
                        # # (prompt_text, part_of, person_img) --  triple['text-text-img']
                        # head_embs.append(prompt_emb)
                        # relation_embs.append(common_verb_tensor[[word_dict["part_of"]]])
                        # tail_embs.append(person_emb)

                        # (verb, part_of, person_img) --  triple['text-text-img']
                        kg_index['txt-txt-img'].append((self.word_dict[verb], self.word_dict["part of"], human_dict[i]))
                        nb_dict['nb_triple'] += 1

                    if i > 0:
                        j = 0
                        while (j<i):
                            # (person1, together_with, person2) --  triple['img-text-img']
                            kg_index['img-txt-img'].append((human_dict[j], self.word_dict["together with"], human_dict[i]))
                            nb_dict['nb_triple'] += 1

                            # (person2, together_with, person1) --  triple['img-text-img']
                            kg_index['img-txt-img'].append((human_dict[i], self.word_dict["together with"], human_dict[j]))
                            nb_dict['nb_triple'] += 1
                            j += 1

        # img_tensor = torch.cat(img_tensor, dim=0)
        # # save image tensor of each video
        # saved_path = os.path.join(self.outdir, video_id + "_image_tensor.pth")
        # path = Path(self.outdir)
        # path.mkdir(parents=True, exist_ok=True)
        # torch.save(img_tensor, saved_path)
        # save kg index of each video
        torch.save(kg_index, savepath+"kg_idx.pth")

        # self._print_info(nb_dict, saved_path, video_id)

    def _load_annoatation(self, video_id):
        json_file_names = [filename for filename in os.listdir(os.path.join(self.annotation_path, video_id)) if filename.endswith('.json')]
        json_files = []
        for jsonf in json_file_names:
            try:
                with open(os.path.join(self.annotation_path, video_id, jsonf)) as f:
                    json_data = json.load(f)
                    json_files.append(json_data)
            except FileNotFoundError :
                print('ERROR: Could not find annotation file for frame "{}" of video "{}"'.format(frame_id, video_id))
                raise

        frame_file_names = [filename for filename in os.listdir(os.path.join(self.frame_path, video_id)) if filename.endswith('.jpg')]
        frame_files = []
        for framef in frame_file_names:
            try:
                img = self._load_frame(os.path.join(self.frame_path, video_id, framef))
                frame_files.append(img)
            except FileNotFoundError :
                print('ERROR: Could not find frame "{}" of video "{}"'.format(frame_id, video_id))
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

    def _print_info(self, nb_dict, saved_path, video_id):
        print('-' * 80)
        print('-' * 80)
        print(' ' * 20, "The statistics of event knowledge graph of video {}:".format(video_id))
        print('-' * 80)
        print("{:<8} {:<15} ".format('Key','Number'))
        for k, v in nb_dict.items():
            num = v
            print("{:<8} {:<15} ".format(k, num))
        print('-' * 80)
        print('-' * 80)
        print(' ' * 20, "Triple tensor have been saved in : {}".format(saved_path))
        print('-' * 80)
        print('-' * 80)

    def _draw_graph(self):
        edges = [('body_img', 'body_txt'),
                 ('verb', 'body_img'),
                 ('verb', 'body_txt'),
                 ('verb', 'person_1'),
                 ('person_1', 'person_2'),
                 ('person_2', 'person_1'),
                 ('person_1', 'frame_1'),
                 ('person_2', 'frame_1'),
                 ('frame_1', 'frame_2'),
                 ('verb', 'obj_img'),
                 ('verb', 'obj_txt'),
                 ('obj_img', 'obj_txt')]
        edge_labels = ['is_about',
                       'subject_of',
                       'subject_of',
                       'part_of',
                       'together_with',
                       '',
                       'instance_of',
                       'instance_of',
                       'followed_by',
                       'object_of',
                       'object_of',
                       'is_about']
        G = nx.MultiDiGraph()
        G.add_edges_from(edges)
        pos ={'body_img': np.array([-0.6184773, -0.86804322]),
              'body_txt': np.array([-0.3530498, -0.87327142]),
              'verb': np.array([-0.58021684, -0.13902534]),
              'person_1': np.array([-0.57043059, 0.55390253]),
              'person_2': np.array([-0.16390437, 0.54390253 ]),
              'frame_1': np.array([-0.53489465, 1.3]),
              'frame_2': np.array([0.2052812, 1.3    ]),
              'obj_img': np.array([- 0.1334298, -0.86395495]),
              'obj_txt': np.array([0.170452 , -0.86408848])}
        plt.figure()
        nx.draw(
            G, pos, edge_color='black', width=1, linewidths=1,
            node_size=2000, node_color='skyblue', alpha=0.9,font_size=10,
            labels={node: node for node in G.nodes()},arrows=True
        )

        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={i:j for i,j in zip(edges, edge_labels)},
            font_color='red',font_size=8,
            alpha=0.5
        )
        plt.title("An Illustration of Multi-modal Event Knowledge Graph")
        plt.axis('off')
        plt.savefig(self.outdir+'example_KG.png')
        plt.show()
