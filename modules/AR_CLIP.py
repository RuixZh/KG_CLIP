import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.sequence_modules import Transformer, TAggregate, LayerNorm
from modules.optimization import KLLoss
from modules.triple_modules import Triple_Transformer
import numpy as np
import math


class ARClip(nn.Module):
    def __init__(self, args, embed_dim, transformer_heads):
        super(ARClip, self).__init__()

        self.args = args

        num_frames = args.max_frames

        self.relation_embeddings = nn.Embedding(num_embeddings=6, embedding_dim=2 * embed_dim)

        self.triple_position_embeddings = nn.Embedding(num_embeddings=4, embedding_dim=embed_dim)
        self.triple_transformer = Triple_Transformer(width=embed_dim, layers=3, heads=transformer_heads)
        self.noise = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)

        self.apply(self.init_weights)
        self.loss_fct = KLLoss()

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_sim(self, x1, x2, logit_scale):
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)
        sim = logit_scale * x1 @ x2.t()
        simt = logit_scale * x2 @ x1.t()

        return sim, simt

    def cross_sim(self, text, frame, logit_scale):
        text = text / text.norm(dim=-1, keepdim=True)
        frame = frame / frame.norm(dim=-1, keepdim=True)
        s = torch.matmul(frame, text.t())
        cross_sim = logit_scale * torch.sum(s * torch.softmax(s / 1e-2, dim=1), dim=1)
        logpt = F.log_softmax(cross_sim, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()

        return sim_loss  # （bs, 1)

    def seq_encoder(self, x, return_hidden=False):
        b, t, c = x.size()
        x = x.contiguous()

        x_original = x
        seq_length = t
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + frame_position_embeddings

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.temporal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original

        if return_hidden:
            return x, x.mean(dim=1, keepdim=False)
        else:
            return x.mean(dim=1, keepdim=False)

    def triple_encoder(self, head, rel, tail, return_hidden=False):
        rel1, rel2 = torch.chunk(rel, 2, dim=-1)
        x = torch.stack([head, rel1,rel2, tail], 1)  # (bs, 3, dim)
        x = x.contiguous()
        x_original = x

        position_ids = torch.arange(4, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(head.size(0), -1)
        triple_position_embeddings = self.triple_position_embeddings(position_ids)
        x = x + triple_position_embeddings

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.triple_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x + x_original    # (bs, 3, dim)

        x_h = x[:, 0, :] * x[:, 1, :] - self.noise
        x_t = x[:, 2, :] * x[:, 3, :]
        if return_hidden:
            return x, x.mean(dim=1, keepdim=False)
        else:
            return x_h.type(head.dtype), x_t.type(head.dtype)

    def E2E_leaner(self, inputs, logit_scale, video=False):
        head_emb, rel, tail_emb, label = inputs
        rel_emb = self.relation_embeddings(rel)
        sim_loss = 0.0
        if video:
            head_emb = head_emb.mean(1)

        sim, simt = self.get_sim(head_emb, tail_emb, logit_scale)
        sim_loss = (self.loss_fct(sim, label) + self.loss_fct(simt, label)) / 2

        b, d = head_emb.size()
        triple_h, triple_t = self.triple_encoder(head_emb, rel_emb, tail_emb)

        rel_emb = self.relation_embeddings(rel + 3)
        triple_inv_h, triple_inv_t = self.triple_encoder(tail_emb, rel_emb, head_emb)

        sim, simt = self.get_sim(triple_h, triple_t, logit_scale)
        loss = (self.loss_fct(sim, label) + self.loss_fct(simt, label)) / 2

        sim, simt = self.get_sim(triple_inv_h, triple_inv_t, logit_scale)
        loss += (self.loss_fct(sim, label) + self.loss_fct(simt, label)) / 2

        return (loss / 2) + (self.args.lam_coef * sim_loss)

    def forward(self, va, vb=None, ab=None, logit_scale=None, isTraining=False):
        if isTraining:
            vb_loss = self.E2E_leaner(vb, logit_scale, video=True)
            ab_loss = self.E2E_leaner(ab, logit_scale, video=False)
            va_loss = self.E2E_leaner(va, logit_scale, video=True)
            total_loss = vb_loss + ab_loss + va_loss
            return total_loss

        else:
            head_emb, va_rel, class_emb = va
            head_emb = head_emb.mean(1) # self.seq_encoder(head_emb)

            vout = head_emb / head_emb.norm(dim=-1, keepdim=True)
            cout = class_emb / class_emb.norm(dim=-1, keepdim=True)
            sim = 0.5*(vout @ cout.t())

            bs, dim = head_emb.size()
            nbc, dim = class_emb.size()

            head_emb = head_emb.unsqueeze(1).repeat(1, nbc, 1).view(-1, dim)
            class_emb = class_emb.unsqueeze(0).repeat(bs, 1, 1).view(-1, dim)

            rel_emb = self.relation_embeddings(va_rel)
            rel_emb = rel_emb.unsqueeze(1).repeat(1, nbc, 1).view(-1, 2*dim)
            triple_h, triple_t = self.triple_encoder(head_emb, rel_emb, class_emb)
            triple_h = triple_h.view(bs, nbc, -1).mean(1)
            triple_t = triple_t.view(bs, nbc, -1).mean(0)

            rel_emb = self.relation_embeddings(va_rel + 3)
            rel_emb = rel_emb.unsqueeze(1).repeat(1, nbc, 1).view(-1, 2*dim)
            triple_inv_h, triple_inv_t = self.triple_encoder(class_emb, rel_emb, head_emb)
            triple_inv_h = triple_inv_h.view(bs, nbc, -1).mean(1)
            triple_inv_t = triple_inv_t.view(bs, nbc, -1).mean(0)

            vout = triple_h / triple_h.norm(dim=-1, keepdim=True)
            cout = triple_t / triple_t.norm(dim=-1, keepdim=True)
            sim += 0.25 * (vout @ cout.t())

            vout = triple_inv_h / triple_inv_h.norm(dim=-1, keepdim=True)
            cout = triple_inv_t / triple_inv_t.norm(dim=-1, keepdim=True)
            sim += 0.25 * (vout @ cout.t())

            return sim
