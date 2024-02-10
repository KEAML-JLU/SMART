import math
import torch
import torch.nn as nn
from gin import GraphCNN
from functools import reduce
from operator import mul
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.T = args.t
        self.hid = args.gin_hid
        self.N = args.N_way
        self.K = args.K_shot
        self.Q = args.query_size
        self.device = self.args.device

        self.sample_input_emb_size = args.sample_input_size

        self.gin = GraphCNN(input_dim=args.node_fea_size, use_select_sim=args.use_select_sim, num_layers=args.gin_layer,
                            hidden_dim=args.gin_hid).to(self.device)  # .cuda()

        self.proj_head = nn.Sequential(nn.Linear(self.sample_input_emb_size, self.hid), nn.ReLU(inplace=True),
                                       nn.Linear(self.hid, self.hid))  # whether to use the batchnorm1d?

        if args.baseline_mode == 'relation':
            self.rel_classifier = nn.Linear(self.sample_input_emb_size * 2, args.train_classes_num)

        self.dropout = nn.Dropout(args.dropout)

    def sample_input_GNN(self, tasks, prompt_embs, is_support):
        embs = []
        final_hidds = []
        for task in tasks:
            graphs = []

            if is_support:
                for i in range(len(task['support_set'])):
                    graphs.extend(task['support_set'][i])
            else:
                for i in range(len(task['query_set'])):
                    graphs.extend(task['query_set'][i])

            pooled_h_layers, node_embeds, Adj_block_idx, hidden_rep, final_hidd = self.gin(graphs, mode='test',
                                                                                           promp=prompt_embs)  # [N(K+Q), emb_size]
            embs.append(torch.cat(pooled_h_layers[1:], -1))
            final_hidds.append(final_hidd)
        return torch.cat(embs, 0), [torch.cat([one[layer] for one in final_hidds], 0) for layer in
                                    range(self.gin.num_layers)] if self.args.use_select_sim else []

    def forward(self, batch_graph):
        output_embeds, node_embeds, Adj_block_idx, _, _ = self.gin(batch_graph)
        pooled_h = torch.cat(output_embeds[1:], -1)
        pooled_h = self.proj_head(pooled_h)
        return pooled_h

    def loss_cal(self, x, x_aug):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / self.T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss


class Prompt(nn.Module):
    def __init__(self, args):
        super(Prompt, self).__init__()
        self.token = args.num_token
        self.prompt_dropout = nn.Dropout(args.dropout)
        #         self.prompt_proj = nn.Linear(
        #             args.node_fea_size, args.node_fea_size) #args.gin_hid)
        #         nn.init.kaiming_normal_(
        #             self.prompt_proj.weight, a=0, mode='fan_out')

        val = 0.001  # 0.5  # noqa

        self.prompt_embeddings = nn.Parameter(torch.zeros(
            self.token, args.node_fea_size))
        # xavier_uniform initialization
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)

    def forward(self):
        return self.prompt_dropout(
            self.prompt_embeddings)  # self.prompt_dropout(self.prompt_proj(self.prompt_embeddings))
