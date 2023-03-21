import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial, reduce
from itertools import chain
from .utils import *


class SmyrfAttention(nn.Module):
    def __init__(self, n_hashes, q_cluster_size, k_cluster_size,
                 q_attn_size=None, k_attn_size=None,
                 clustering_algo='lsh',
                 dropout=0.,
                 # LSH clustering
                 r=1,
                 # kmeans clustering
                 max_iters=50):
        super(SmyrfAttention, self).__init__()
        self.n_hashes = n_hashes

        if q_attn_size is None:
            self.q_attn_size = q_cluster_size
        else:
            self.q_attn_size = q_attn_size

        if k_attn_size is None:
            self.k_attn_size = k_cluster_size
        else:
            self.k_attn_size = k_attn_size

        self.dropout = nn.Dropout(dropout)
        self.xbox_plus = XBOXPLUS()

        self.clustering_algo =  clustering_algo
        if clustering_algo == 'lsh':
            self.clustering_params = {
                'r': r,
                'n_hashes': self.n_hashes
            }
        else:
            raise NotImplementedError('Uknown clustering algorithm')


    def forward(self, queries, keys, values, attn_mask=None, progress=False,
                norm_factor=1, return_attn_map=False):
        cls_q = queries[:, :1]
        cls_k = keys[:, :1]
        cls_v = values[:, :1]
        queries = queries[:, 1:]
        keys = keys[:, 1:]
        values = values[:, 1:]

        bs, q_seqlen, dim = queries.shape
        bs, k_seqlen, dim = keys.shape
        v_dim = values.shape[-1]
        assert queries.device == keys.device, 'Queries, keys in different devices'
        device = queries.device


        # prepare mask if not None
        if attn_mask is not None:
            # We expect first dimension to be batch_size and second dimension seq. length
            if len(attn_mask.shape) == 1:
                attn_mask = attn_mask.unsqueeze(0)
            # repeat for n_hashes, heads
            attn_mask = attn_mask.unsqueeze(0).repeat(self.n_hashes, queries.shape[0] // attn_mask.shape[0], 1)

        with torch.no_grad():
            # XBOX+ transform
            self.xbox_plus.set_norms(queries, keys)
            Queries = self.xbox_plus.Q(queries)
            Keys = self.xbox_plus.K(keys)

            num_clusters = Queries.shape[1] // self.q_attn_size
            assert num_clusters == (Keys.shape[1] // self.k_attn_size), 'Unequal number of clusters for queries and keys.'


            if self.clustering_algo == 'lsh':
                q_positions, k_positions = lsh_clustering(Queries, Keys, **self.clustering_params, attn_mask=attn_mask)
            else:
                raise NotImplementdError('This algorithm is not supported')

            q_positions = q_positions.reshape(self.n_hashes, bs, -1)
            k_positions = k_positions.reshape(self.n_hashes, bs, -1)

        # free memory
        del Queries
        del Keys


        q_rev_positions = torch.argsort(q_positions, dim=-1)
        q_offset = torch.arange(bs, device=queries.device).unsqueeze(-1) * q_seqlen
        k_offset = torch.arange(bs, device=queries.device).unsqueeze(-1) * k_seqlen


        q_flat = (q_positions + q_offset).reshape(-1)
        k_flat = (k_positions + k_offset).reshape(-1)
        import pdb; pdb.set_trace()

        # sorted queries, keys, values
        s_queries = queries.reshape(-1, dim).index_select(0, q_flat).reshape(-1, self.q_attn_size, dim)
        s_keys = keys.reshape(-1, dim).index_select(0, k_flat).reshape(-1, self.k_attn_size, dim)
        s_values = values.reshape(-1, v_dim).index_select(0, k_flat).reshape(-1, self.k_attn_size, v_dim)

        cls_q = cls_q[None, :, None].repeat(self.n_hashes, 1, q_seqlen // self.q_attn_size, 1, 1).reshape(-1, 1, dim)
        cls_k = cls_k[None, :, None].repeat(self.n_hashes, 1, k_seqlen // self.k_attn_size, 1, 1).reshape(-1, 1, dim) 
        cls_v = cls_v[None, :, None].repeat(self.n_hashes, 1, k_seqlen // self.k_attn_size, 1, 1).reshape(-1, 1, v_dim) 

        s_queries = torch.cat([cls_q, s_queries], dim=1)
        s_keys = torch.cat([cls_k, s_keys], dim=1)
        s_values = torch.cat([cls_v, s_values], dim=1)

        inner = s_queries @ s_keys.transpose(2, 1)
        inner = inner / norm_factor

        # mask out attention to padded tokens
        if attn_mask is not None:
            inner = (attn_mask.reshape(-1)[k_flat].reshape(-1, self.k_attn_size).unsqueeze(1) + inner)

        # free memory
        if not return_attn_map:
            del q_positions, k_positions

        # softmax denominator
        dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
        # softmax
        dots = torch.exp(inner - dots_logsumexp)
        # dropout
        dots = self.dropout(dots)

        # n_hashes outs
        bo = dots @ s_values
        cls_bo = bo[:, :1].reshape(self.n_hashes, bs, q_seqlen // self.q_attn_size, -1)
        bo = bo[:, 1:].reshape(self.n_hashes, bs, q_seqlen, -1)
        # bo = (dots @ s_values).reshape(self.n_hashes, bs, q_seqlen, -1)

        # undo sort
        q_offset = torch.arange(bs * self.n_hashes, device=queries.device).unsqueeze(-1) * q_seqlen
        q_rev_flat = (q_rev_positions.reshape(-1, q_seqlen) + q_offset).reshape(-1)
        o = bo.reshape(-1, v_dim).index_select(0, q_rev_flat).reshape(self.n_hashes, bs, q_seqlen, -1)

        slogits = dots_logsumexp[:, 1:].reshape(self.n_hashes, bs, -1)
        logits = torch.gather(slogits, 2, q_rev_positions)

        # free memory
        del q_rev_positions

        probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
        out = torch.sum(o * probs.unsqueeze(-1), dim=0)
        cls_bo = cls_bo.mean(dim=(0, 2))
        out = torch.cat([cls_bo, out], dim=1)

        if return_attn_map:
            return out, (q_positions, k_positions)
        else:
            return out



def dense(query, key, value):
    return F.softmax(query @ key.permute(0, 2, 1), dim=-1) @ value


if __name__ == '__main__':
    N = 1024
    dim = 30
    bs = 2
    n_hashes = 8
    q_cluster_size = k_cluster_size = 16
    device = 'cuda'

    queries = torch.randn(bs, N, dim, device=device)
    keys = torch.randn(bs, N, dim, device=device)
    values = torch.randn(bs, N, dim, device=device)

    approximator = SmyrfAttention(n_hashes, q_cluster_size, k_cluster_size)
    approximator(queries, keys, values)
