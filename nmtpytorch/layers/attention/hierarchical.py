# -*- coding: utf-8 -*-
import random
import torch
from torch import nn

from ...utils.nn import get_activation_fn


# Libovický, J., & Helcl, J. (2017). Attention Strategies for Multi-Source
# Sequence-to-Sequence Learning. In Proceedings of the 55th Annual Meeting of
# the Association for Computational Linguistics (Volume 2: Short Papers)
# (Vol. 2, pp. 196-202). [Code contributed by @jlibovicky]


class HierarchicalAttention(nn.Module):
    """Hierarchical attention over multiple modalities."""
    def __init__(self, ctx_dims, hid_dim, mid_dim, att_activ='tanh', dropnet=False, \
    dropnet_image_rate=0.15, dropnet_text_rate=0.15):
        super().__init__()

        self.activ = get_activation_fn(att_activ)
        self.ctx_dims = ctx_dims
        self.hid_dim = hid_dim
        self.mid_dim = mid_dim

        self.dropnet = dropnet
        self.dropnet_image_rate = dropnet_image_rate
        self.dropnet_text_rate = dropnet_text_rate

        self.ctx_projs = nn.ModuleList([
            nn.Linear(dim, mid_dim, bias=False) for dim in self.ctx_dims])
        self.dec_proj = nn.Linear(hid_dim, mid_dim, bias=True)
        self.mlp = nn.Linear(self.mid_dim, 1, bias=False)

    def forward(self, contexts, hid):
        # contexts[0] --> textual context
        # contexts[1] --> visual context
        dec_state_proj = self.dec_proj(hid)
        ctx_projected = torch.cat([
            p(ctx).unsqueeze(0) for p, ctx
            in zip(self.ctx_projs, contexts)], dim=0)
        energies = self.mlp(self.activ(dec_state_proj + ctx_projected))
        att_dist = nn.functional.softmax(energies, dim=0)

        ctxs_cat = torch.cat([c.unsqueeze(0) for c in contexts])
        
        # dropnet mechanism
        if self.training == True:
            if self.dropnet == True:
                x = random.choices([0, 1, 2], k=1, weights=[self.dropnet_image_rate, \
                self.dropnet_text_rate, 1-self.dropnet_image_rate-self.dropnet_text_rate])[0]
                # use only textual context
                if x == 0:
                    joint_context = (att_dist * ctxs_cat)[0]
                # use only image context
                elif x == 1:
                    joint_context = (att_dist * ctxs_cat)[1]
                # use both visual and textual contexts (i.e. multimodal context)
                else:
                    joint_context = (att_dist * ctxs_cat).sum(0)
            else: 
                joint_context = (att_dist * ctxs_cat).sum(0)
        else: 
            joint_context = (att_dist * ctxs_cat).sum(0)

        return att_dist, joint_context
