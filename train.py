import os
import json
import math
import torch
import random
import argparse
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from model import Model, Prompt
from logreg import LogReg
from aug import aug_fea_mask, aug_drop_node, aug_fea_drop, aug_fea_dropout
from dataset import Dataset, Graph, load_data
from collections import defaultdict
from numpy.random import RandomState
from sklearn.linear_model import LogisticRegression
from ogb.graphproppred import GraphPropPredDataset


class Trainer:
    def __init__(self, args):
        self.args = args
        self.epoch_num = args.epoch_num
        self.K_shot = args.K_shot
        self.patience = args.patience
        self.device = self.args.device
        self.query_size = args.query_size
        self.eval_interval = args.eval_interval

        self.dataset = Dataset(args.dataset_name, args)
        args.train_classes_num = self.dataset.train_classes_num
        args.test_classes_num = self.dataset.test_classes_num
        args.node_fea_size = self.dataset.train_graphs[0].node_features.shape[1]
        args.sample_input_size = (args.gin_layer - 1) * args.gin_hid

        args.N_way = self.dataset.test_classes_num
        self.N_way = self.dataset.test_classes_num

        self.baseline_mode = args.baseline_mode

        self.model = Model(args).to(self.device)  # .cuda()

        self.prompt = Prompt(self.args).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.criterion = nn.CrossEntropyLoss()

        # use torch to implement the linear reg
        self.log = LogReg(self.model.sample_input_emb_size, self.N_way).to(self.device)
        self.opt = optim.SGD([{'params': self.log.parameters()}, {'params': self.prompt.parameters()}], lr=0.01)
        self.xent = nn.CrossEntropyLoss()

    def train(self):
        # best_test_acc = 0
        # best_valid_acc = 0
        best = 1e9
        best_t = 0
        cnt_wait = 0

        train_accs = []
        # graph_copy_1 = deepcopy(self.dataset.train_graphs)
        graph_copy_2 = deepcopy(self.dataset.train_graphs)
        if self.args.aug1 == 'identity':
            graph_aug1 = self.dataset.train_graphs
        elif self.args.aug1 == 'node_drop':
            graph_aug1 = aug_drop_node(self.dataset.train_graphs)
        elif self.args.aug1 == 'feature_mask':
            graph_aug1 = aug_fea_mask(self.dataset.train_graphs)
        elif self.args.aug1 == 'feature_drop':
            graph_aug1 = aug_fea_drop(self.dataset.train_graphs)
        elif self.args.aug1 == 'feature_dropout':
            graph_aug1 = aug_fea_dropout(self.dataset.train_graphs)

        if self.args.aug2 == 'node_drop':
            graph_aug2 = aug_drop_node(graph_copy_2)
        elif self.args.aug2 == 'feature_mask':
            graph_aug2 = aug_fea_mask(graph_copy_2)
        elif self.args.aug2 == 'feature_drop':
            graph_aug2 = aug_fea_drop(self.dataset.train_graphs)
        elif self.args.aug2 == 'feature_dropout':
            graph_aug2 = aug_fea_dropout(self.dataset.train_graphs)

        print("graph augmentation complete!")

        for i in range(self.epoch_num):
            loss = self.train_one_step(mode='train', epoch=i, graph_aug1=graph_aug1, graph_aug2=graph_aug2)

            if loss == None: continue

            if i % 50 == 0:
                print('Epoch {} Loss {:.4f}'.format(i, loss))
                f.write('Epoch {} Loss {:.4f}'.format(i, loss) + '\n')
                if loss < best:
                    best = loss
                    best_t = i
                    cnt_wait = 0
                    torch.save(self.model.state_dict(), './savepoint/' + self.args.dataset_name + '_model.pkl')
                else:
                    cnt_wait += 1
            if cnt_wait > self.patience:
                print("Early Stopping!")
                break

    def test(self):
        best_test_acc = 0
        self.model.load_state_dict(torch.load('./savepoint/' + self.args.dataset_name + '_model.pkl'))
        print("model load success!")
        self.model.eval()

        test_accs = []
        start_test_idx = 0
        while start_test_idx < len(self.dataset.test_graphs) - self.K_shot * self.dataset.test_classes_num:
            test_acc = self.train_one_step(mode='test', epoch=0, test_idx=start_test_idx)
            test_accs.append(test_acc)
            start_test_idx += self.N_way * self.query_size

        # print('test task num', len(test_accs))
        mean_acc = sum(test_accs) / len(test_accs)
        std = np.array(test_accs).std()
        #         if mean_acc > best_test_acc:
        #             best_test_acc = mean_acc

        print('Mean Test Acc {:.4f}  Std {:.4f}'.format(mean_acc, std))
        f.write('Mean Test Acc {:.4f}  Std {:.4f}'.format(mean_acc, std) + '\n')

        return best_test_acc

    def train_one_step(self, mode, epoch, graph_aug1=None, graph_aug2=None, test_idx=None, baseline_mode=None):
        if mode == 'train':
            self.model.train()
            train_embs = self.model(graph_aug1)
            train_embs_aug = self.model(graph_aug2)

            loss = self.model.loss_cal(train_embs, train_embs_aug)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss

        elif mode == 'test':
            self.model.eval()
            self.prompt.train()
            prompt_embeds = self.prompt()

            first_N_class_sample = np.array(list(range(self.dataset.test_classes_num)))
            current_task = self.dataset.sample_one_task(self.dataset.test_tasks, first_N_class_sample,
                                                        K_shot=self.K_shot, query_size=self.query_size,
                                                        test_start_idx=test_idx)
            support_current_sample_input_embs, support_current_sample_input_embs_selected = self.model.sample_input_GNN(
                [current_task], prompt_embeds, True)  # [N(K+Q), emb_size]

            if not self.args.test_mixup:
                support_data = support_current_sample_input_embs.detach().cpu().numpy()  # [NxK, d]

            else:
                # if not use .cpu().numpy(), it illustrates that we use the torch linear reg
                data = support_current_sample_input_embs.reshape(self.N_way, self.K_shot + self.args.gen_test_num,
                                                                 self.model.sample_input_emb_size)
                support_data, support_data_mixup = data[:, :self.K_shot, :].reshape(self.N_way * self.K_shot,
                                                                                    self.model.sample_input_emb_size).detach(), data[
                                                                                                                                :,
                                                                                                                                self.K_shot:self.K_shot + self.args.gen_test_num,
                                                                                                                                :].reshape(
                    self.N_way * self.args.gen_test_num, self.model.sample_input_emb_size).detach()  # .cpu().numpy()

            support_label, support_label_mix_a, weight, support_label_mix_b = [], [], [], []
            for graphs in current_task['support_set']:
                support_label.append(np.array([graph.label for graph in graphs[:self.K_shot]]))
                support_label_mix_a.append(np.array([graph.y_a for graph in graphs[self.K_shot:]]))
                support_label_mix_b.append(np.array([graph.y_b for graph in graphs[self.K_shot:]]))
                weight.append(np.array([graph.lam for graph in graphs[self.K_shot:]]))

            support_label = torch.LongTensor(np.hstack(support_label)).to(self.device)
            support_label_mix_a = torch.LongTensor(np.hstack(support_label_mix_a)).to(self.device)
            support_label_mix_b = torch.LongTensor(np.hstack(support_label_mix_b)).to(self.device)
            weight = torch.FloatTensor(np.hstack(weight)).to(self.device)

            # this is used for linear function based on torch
            self.log.train()
            best_loss = 1e9
            wait = 0
            patience = 10
            for _ in range(500):

                self.opt.zero_grad()
                # original support data
                logits = self.log(support_data)
                loss_ori = self.xent(logits, support_label)

                # mixup data
                logits_mix = self.log(support_data_mixup)  # [Nxgen, class]
                loss_mix = (weight * self.xent(logits_mix, support_label_mix_a) + \
                            (1 - weight) * self.xent(logits_mix, support_label_mix_b)).mean()

                l2_reg = torch.tensor(0.).to(self.device)
                for param in self.log.parameters():
                    l2_reg += torch.norm(param)
                loss_leg = loss_ori + loss_mix + 0.1 * l2_reg

                loss_leg.backward()
                self.opt.step()

                if loss_leg < best_loss:
                    best_loss = loss_leg
                    wait = 0
                    torch.save(self.log.state_dict(), './savepoint/' + self.args.dataset_name + '_lr.pkl')
                else:
                    wait += 1
                if wait > patience:
                    print("Early Stopping!")
                    break

            self.log.load_state_dict(torch.load('./savepoint/' + self.args.dataset_name + '_lr.pkl'))
            self.log.eval()
            self.prompt.eval()
            prompt_embeds = self.prompt()

            query_current_sample_input_embs, _ = self.model.sample_input_GNN(
                [current_task], prompt_embeds, False)  # [N(K+Q), emb_size]
            query_label = []

            if not self.args.test_mixup:
                query_data = query_current_sample_input_embs.reshape(self.N_way, self.query_size,
                                                                     self.model.sample_input_emb_size).detach().cpu().numpy()  # [NxQ, d]
            else:
                query_data = query_current_sample_input_embs.detach()  # .cpu().numpy()

            for graphs in current_task['query_set']:
                query_label.append(np.array([graph.label for graph in graphs]))

            query_label = torch.LongTensor(np.hstack(query_label)).to(self.device)

            query_len = query_label.shape[0]
            if current_task['append_count'] != 0:
                query_data = query_data[: query_len - current_task['append_count'], :]
                query_label = query_label[: query_len - current_task['append_count']]

            logits = self.log(query_data)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == query_label).float() / query_label.shape[0]

            test_acc = acc.cpu().numpy()

            return test_acc


def parse_arguments():
    parser = argparse.ArgumentParser()

    # GIN parameters
    parser.add_argument('--dataset_name', type=str, default="TRIANGLES",
                        help='name of dataset')

    parser.add_argument('--baseline_mode', type=str, default=None,
                        help='baseline')

    parser.add_argument('--N_way', type=int, default=3)
    parser.add_argument('--K_shot', type=int, default=5)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch', type=float, default=128)

    parser.add_argument('--gin_layer', type=int, default=3)
    parser.add_argument('--gin_hid', type=int, default=128)
    parser.add_argument('--aug1', type=str, default='node_drop')
    parser.add_argument('--aug2', type=str, default='feature_mask')
    parser.add_argument('--t', type=float, default=0.2)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-7)

    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--epoch_num', type=int, default=3000)

    parser.add_argument('--use_select_sim', type=bool, default=False)
    parser.add_argument('--gen_train_num', type=int, default=500)
    parser.add_argument('--gen_test_num', type=int, default=20)

    parser.add_argument('--save_test_emb', type=bool, default=True)
    parser.add_argument('--test_mixup', type=bool, default=True)
    parser.add_argument('--num_token', type=int, default=1)

    args = parser.parse_args()
    return args


args = parse_arguments()

args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
datasets = []  # ['TRIANGLES']
datasets.append(args.dataset_name)
res = {}

for dataset in datasets:
    for k in [5]:  # 10
        args.K_shot = k
        accs = []
        for seed_value in range(72, 73):
            # for seed_value in range(5):
            os.environ['PYTHONHASHSEED'] = str(seed_value)
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed(seed_value)

            text_file_name = './our_results/{}-{}-shot.txt'.format(dataset, k)
            f = open(text_file_name, 'w')
            file_name = './our_results/{}-{}-shot-params.txt'.format(dataset, k)

            print(file_name)
            args.dataset_name = dataset
            trainer = Trainer(args)

            trainer.train()
            test_acc = trainer.test()
            accs.append(test_acc)

            res[test_acc] = str(args)
            del trainer
            del test_acc

            json.dump(res, open(file_name, 'a'), indent=4)
        # print("acc: ", np.array(accs).mean(), "std: ", np.array(accs).std())