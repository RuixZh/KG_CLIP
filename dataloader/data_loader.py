import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from pathlib import Path
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from collections import defaultdict
from .data_transforms import *
from RandAugment import RandAugment
import clip_model as clip
from scipy import sparse
import pickle

class TripleRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def head_id(self):
        return self._data[0]

    @property
    def rel_id(self):
        return self._data[1]

    @property
    def tail_id(self):
        return self._data[2]


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class Kinetics_DataLoader(Dataset):
    def __init__(self,
                 vb_path=None,
                 va_path=None,
                 ab_path=None,
                 kg_dict_path=None,
                 frame_path=None,
                 max_frames=None,
                 class_path=None,
                 n_px=224,
                 isTraining=False
        ):
        self.kg_dict_path = kg_dict_path
        self.frame_path = frame_path
        self.max_frames = max_frames
        self.n_px = 224
        self.isTraining = isTraining
        self.va_graph = va_path
        self.class_path = class_path
        self.transform = self._transform()

        if self.isTraining:
            self.vb_graph, self.ab_graph = vb_path, ab_path
            self.transform.transforms.insert(0, GroupTransform(RandAugment(2, 9)))

        self._kg_dict_load()
        self._parse_list()
        self._gen_label_dict()

    def _gen_label_dict(self):
        label= {}
        self.label_inv = {}
        with open(self.class_path, 'r') as f:
            for l in f.readlines():
                category, lid = l.strip().split('\t')
                category = category.replace('_', ' ')
                label[int(lid)] = category
                self.label_inv[category] = int(lid)
        self.nb_label = len(label)
        self.classes = clip.tokenize(self.label_inv)

    def _kg_dict_load(self):
        with open(self.kg_dict_path, 'r') as fp:
            self.kg_dict = json.load(fp)
        self.kg_dict = {int(k):v for k,v in self.kg_dict.items()}
        self.kg_dict_inv = {v:k for k,v in self.kg_dict.items()}

    def _parse_list(self):
        # load triplet
        if self.isTraining:
            with open(self.vb_graph, "rb") as fp:
                graph = pickle.load(fp)
            self.vb_triple = [TripleRecord(x) for x in graph]
            with open(self.ab_graph, "rb") as fp:
                graph = pickle.load(fp)
            self.ab_triple = [TripleRecord(x) for x in graph]
        with open(self.va_graph, "rb") as fp:
            graph = pickle.load(fp)
        self.va_triple = [TripleRecord(x) for x in graph]

    def _transform(self):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        scale_size = self.n_px * 256 // 224
        if self.isTraining:
            unique = torchvision.transforms.Compose([GroupMultiScaleCrop(self.n_px, [1, .875, .75, .66]),
                                                     GroupRandomHorizontalFlip(is_sth=False),
                                                     GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
                                                                            saturation=0.2, hue=0.1),
                                                     GroupRandomGrayscale(p=0.2),
                                                     GroupGaussianBlur(p=0.0),
                                                     GroupSolarization(p=0.0)]
                                                    )
        else:
            unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                     GroupCenterCrop(self.n_px)])

        common = torchvision.transforms.Compose([Stack(roll=False),
                                                 ToTorchFormatTensor(div=True),
                                                 GroupNormalize(input_mean,
                                                                input_std)])
        return torchvision.transforms.Compose([unique, common])

    def _sample_indices(self, num_frames):
        if num_frames <= self.max_frames:
            offsets = np.concatenate((
                np.arange(num_frames),
                np.random.randint(num_frames,
                        size=self.max_frames - num_frames)))
            return np.sort(offsets)
        offsets = list()
        ticks = [i * num_frames // self.max_frames for i in range(self.max_frames + 1)]

        for i in range(self.max_frames):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= 1:
                tick += np.random.randint(tick_len)
            offsets.extend([j for j in range(tick, tick + 1)])
        return np.array(offsets)

    def _get_val_indices(self, num_frames):
        if self.max_frames == 1:
            return np.array([num_frames //2], dtype=np.int_)

        if num_frames <= self.max_frames:
            return np.array([i * num_frames // self.max_frames
                             for i in range(self.max_frames)], dtype=np.int_)
        offset = (num_frames / self.max_frames - 1) / 2.0
        return np.array([i * num_frames / self.max_frames + offset
                         for i in range(self.max_frames)], dtype=np.int_)

    def _load_image(self, filepath):
        return [Image.open(filepath).convert('RGB')]

    def _load_knowledge(self, vb_record=None, va_record=None, ab_record=None):
        hid, rid, tid = va_record.head_id, va_record.rel_id, va_record.tail_id
        vid = self.kg_dict[hid]
        img_dir = os.path.join(self.frame_path, vid)
        filenames = [i for i in os.listdir(img_dir) if i.endswith('.jpg') & i.startswith('img')]
        filenames.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
        nb_frame = len(filenames)
        try:
            segment_indices = self._sample_indices(nb_frame) if self.isTraining else self._get_val_indices(nb_frame)
        except ValueError:
            print(vid)
        filenames = [filenames[i] for i in segment_indices]
        images = []
        for i, filename in enumerate(filenames):
            try:
                image = self._load_image(os.path.join(img_dir,filename))
                images.extend(image)
            except OSError:
                print('ERROR: Could not load the image!')
                raise
        va_head = self.transform(images)
        va_rel = rid # clip.tokenize(self.kg_dict[rid])[0]
        va_tail = clip.tokenize(self.kg_dict[tid])[0]
        va_label = tid
        if self.isTraining:
            hid, rid, tid = vb_record.head_id, vb_record.rel_id, vb_record.tail_id
            vid = self.kg_dict[hid]
            img_dir = os.path.join(self.frame_path, vid)
            filenames = [i for i in os.listdir(img_dir) if i.endswith('.jpg') & i.startswith('img')]
            filenames.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
            nb_frame = len(filenames)
            try:
                segment_indices = self._sample_indices(nb_frame) if self.isTraining else self._get_val_indices(nb_frame)
            except ValueError:
                print(vid)
            filenames = [filenames[i] for i in segment_indices]
            images = []
            for i, filename in enumerate(filenames):
                try:
                    image = self._load_image(os.path.join(img_dir,filename))
                    images.extend(image)
                except OSError:
                    print('ERROR: Could not load the image!')
                    raise
            vb_head = self.transform(images)
            vb_rel = rid # clip.tokenize(self.kg_dict[rid])[0]
            vb_tail = clip.tokenize(self.kg_dict[tid])[0]
            vb_label = tid

            hid, rid, tid = ab_record.head_id, ab_record.rel_id, ab_record.tail_id
            ab_head = clip.tokenize(self.kg_dict[hid])[0]
            ab_rel = rid # clip.tokenize(self.kg_dict[rid])[0]
            ab_tail = clip.tokenize(self.kg_dict[tid])[0]
            ab_label = tid
            return vb_head, vb_rel, vb_tail, vb_label, ab_head, ab_rel, ab_tail, ab_label, va_head, va_rel, va_tail, va_label
        else:
            label_id = self.label_inv[self.kg_dict[tid]]
            return va_head, va_rel, label_id

    def __getitem__(self, idx):
        if self.isTraining:
            vb_idx = (idx + np.random.randint(len(self.vb_triple))) % len(self.vb_triple)
            vb_record = self.vb_triple[vb_idx]
            va_idx = idx # (idx + np.random.randint(len(self.va_triple))) % len(self.va_triple)
            va_record = self.va_triple[va_idx]
            ab_idx = (idx + np.random.randint(len(self.ab_triple))) % len(self.ab_triple)
            ab_record = self.ab_triple[ab_idx]
            return self._load_knowledge(vb_record=vb_record, va_record=va_record, ab_record=ab_record)
        else:
            va_record = self.va_triple[idx]
            return self._load_knowledge(va_record=va_record)


    def __len__(self):
        if self.isTraining:
            # return max(len(self.vb_triple), len(self.va_triple), len(self.ab_triple))
            return len(self.va_triple)
        else:
            return len(self.va_triple)


class UCF101_DataLoader(Dataset):
    def __init__(self,
                 va_path=None,
                 kg_dict_path=None,
                 frame_path=None,
                 max_frames=None,
                 class_path=None,
                 n_px=224,
        ):
        self.kg_dict_path = kg_dict_path
        self.frame_path = frame_path
        self.max_frames = max_frames
        self.n_px = 224
        self.va_graph = va_path
        self.class_path = class_path
        self.transform = self._transform()

        self._kg_dict_load()
        self._parse_list()
        self._gen_label_dict()

    def _gen_label_dict(self):
        label= {}
        self.label_inv = {}
        with open(self.class_path, 'r') as f:
            for l in f.readlines():
                category, lid = l.strip().split('\t')
                category = category.replace('_', ' ')
                label[int(lid)] = category
                self.label_inv[category] = int(lid)
        self.nb_label = len(label)
        self.classes = clip.tokenize(self.label_inv)

    def _kg_dict_load(self):
        with open(self.kg_dict_path, 'r') as fp:
            self.kg_dict = json.load(fp)
        self.kg_dict = {int(k):v for k,v in self.kg_dict.items()}
        self.kg_dict_inv = {v:k for k,v in self.kg_dict.items()}

    def _parse_list(self):
        # load triplet
        with open(self.va_graph, "rb") as fp:
            graph = pickle.load(fp)
        self.va_triple = [TripleRecord(x) for x in graph]

    def _transform(self):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        scale_size = self.n_px * 256 // 224

        unique = torchvision.transforms.Compose([GroupScale(scale_size),
                                                     GroupCenterCrop(self.n_px)])

        common = torchvision.transforms.Compose([Stack(roll=False),
                                                 ToTorchFormatTensor(div=True),
                                                 GroupNormalize(input_mean,
                                                                input_std)])
        return torchvision.transforms.Compose([unique, common])

    def _sample_indices(self, num_frames):
        if num_frames <= self.max_frames:
            offsets = np.concatenate((
                np.arange(num_frames),
                np.random.randint(num_frames,
                        size=self.max_frames - num_frames)))
            return np.sort(offsets)
        offsets = list()
        ticks = [i * num_frames // self.max_frames for i in range(self.max_frames + 1)]

        for i in range(self.max_frames):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= 1:
                tick += np.random.randint(tick_len)
            offsets.extend([j for j in range(tick, tick + 1)])
        return np.array(offsets)

    def _get_val_indices(self, num_frames):
        if self.max_frames == 1:
            return np.array([num_frames //2], dtype=np.int_)

        if num_frames <= self.max_frames:
            return np.array([i * num_frames // self.max_frames
                             for i in range(self.max_frames)], dtype=np.int_)
        offset = (num_frames / self.max_frames - 1) / 2.0
        return np.array([i * num_frames / self.max_frames + offset
                         for i in range(self.max_frames)], dtype=np.int_)

    def _load_image(self, filepath):
        return [Image.open(filepath).convert('RGB')]

    def _load_knowledge(self, vb_record=None, va_record=None, ab_record=None):
        hid, rid, tid = va_record.head_id, va_record.rel_id, va_record.tail_id
        vid = self.kg_dict[hid]
        img_dir = os.path.join(self.frame_path, vid)
        filenames = [i for i in os.listdir(img_dir) if i.endswith('.jpg') & i.startswith('img')]
        filenames.sort(key=lambda x:int(x.split('.')[0].split('_')[1]))
        nb_frame = len(filenames)
        try:
            segment_indices = self._get_val_indices(nb_frame)
        except ValueError:
            print(vid)
        filenames = [filenames[i] for i in segment_indices]
        images = []
        for i, filename in enumerate(filenames):
            try:
                image = self._load_image(os.path.join(img_dir,filename))
                images.extend(image)
            except OSError:
                print('ERROR: Could not load the image!')
                raise
        va_head = self.transform(images)
        va_rel = rid # clip.tokenize(self.kg_dict[rid])[0]
        va_tail = clip.tokenize(self.kg_dict[tid])[0]
        va_label = tid

        label_id = tid
        return va_head, va_rel, label_id

    def __getitem__(self, idx):
        va_record = self.va_triple[idx]
        return self._load_knowledge(va_record=va_record)


    def __len__(self):
        return len(self.va_triple)
