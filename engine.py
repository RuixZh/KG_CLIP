import torch
import os
import sys
import utils
import numpy as np
from typing import Iterable, Optional
from tqdm import tqdm
import time
import clip_model

def image_encoder(inputs, img_enc, max_frames, device):
    inputs = inputs.view((-1, max_frames, 3) + inputs.size()[-2:])
    bs, nbf, c, h, w = inputs.size()
    inputs = inputs.to(device).view(-1, c, h, w)
    emb = img_enc(inputs)  # (bs*nb_frame, dim)
    emb = emb.view(bs, nbf, -1)  # (bs, nb_frame, dim)
    return emb

def train_epoch(epoch, args, model, cmodel, img_enc, text_enc, data_loader, optimizer, lr_scheduler):
    model.train()
    text_enc.train()
    img_enc.train()
    total_loss = 0.0
    length = len(data_loader)

    with tqdm(total=len(data_loader), desc="Epoch %s"%epoch) as pbar:

        for kkk, batch in enumerate(data_loader):
            if (kkk+1) == 1 or (kkk+1) % 10 == 0:
                lr_scheduler.step(epoch + kkk / length)
            optimizer.zero_grad()

            vb_head, vb_rel, vb_tail, vb_label, ab_head, ab_rel, ab_tail, ab_label, va_head, va_rel, va_tail, va_label = batch
            vb_head_emb = image_encoder(vb_head, img_enc, args.max_frames, args.device)
            vb_rel = vb_rel.to(args.device)
            vb_tail_emb = text_enc(vb_tail.to(args.device))
            vb_label = torch.tensor(utils.generate_label(vb_label), dtype=vb_head_emb.dtype, device=args.device)
            #
            ab_head_emb = text_enc(ab_head.to(args.device))
            ab_rel = ab_rel.to(args.device)
            ab_tail_emb = text_enc(ab_tail.to(args.device))
            ab_label = torch.tensor(utils.generate_label(ab_label), dtype=ab_head_emb.dtype, device=args.device)

            va_head_emb = image_encoder(va_head, img_enc, args.max_frames, args.device)
            va_rel = va_rel.to(args.device)
            va_tail_emb = text_enc(va_tail.to(args.device))
            va_label = torch.tensor(utils.generate_label(va_label), dtype=va_head_emb.dtype, device=args.device)

            logit_scale = cmodel.logit_scale.exp()

            vb = (vb_head_emb, vb_rel, vb_tail_emb, vb_label)
            ab = (ab_head_emb, ab_rel, ab_tail_emb, ab_label)
            va = (va_head_emb, va_rel, va_tail_emb, va_label)

            loss = model(va=va, vb=vb, ab=ab, logit_scale=logit_scale, isTraining=True)
            total_loss += loss
            loss.backward()
            if args.device == "cpu":
                optimizer.step()
            else:
                utils.convert_models_to_fp32(cmodel)
                optimizer.step()
                clip_model.model.convert_weights(cmodel)
            # time.sleep(0.05)
            pbar.set_postfix({"loss":"{0:1.3f}".format(loss)})

            pbar.update(1)

    return total_loss / (kkk+1)


@torch.no_grad()
def evaluate(args, model, cmodel, img_enc, text_enc, classes, data_loader):
    model.eval()
    text_enc.eval()
    img_enc.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0
    total_acc1 = 0.0
    total_acc5 = 0.0

    tail_emb = text_enc(classes.to(args.device))
    with tqdm(total=len(data_loader), desc="Evaluate") as pbar:
        for kkk, batch in enumerate(data_loader):
            va_head, va_rel, label_id = batch
            va_head_emb = image_encoder(va_head, img_enc, args.max_frames, args.device)
            va_rel = va_rel.to(args.device)
            va = (va_head_emb, va_rel, tail_emb)
            b, t, d = va_head_emb.size()
            similarity = model(va=va)

            similarity = similarity.view(b, -1).softmax(dim=-1)
            values_1, indices_1 = similarity.topk(1, dim=-1)
            values_5, indices_5 = similarity.topk(5, dim=-1)
            num += b
            for i in range(b):
                if indices_1[i] == label_id[i]:
                    corr_1 += 1
                if label_id[i] in indices_5[i]:
                    corr_5 += 1

            pbar.update(1)

    top1 = float(corr_1) / num * 100
    top5 = float(corr_5) / num * 100
    # print('Top1: {:.4f}%, Top5: {:.4f}%'.format(top1, top5))
    return top1, top5
