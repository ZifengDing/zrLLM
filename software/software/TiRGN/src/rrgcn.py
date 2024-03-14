import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import *
class PathPredictor(nn.Module):
    # def __init__(self, args, num_rel, num_ent, orig_score_func):
    def __init__(self, args, num_rel, num_ent, input_dropout, hidden_dropout, feat_dropout):
        super(PathPredictor, self).__init__()
        self.loss = nn.BCELoss(reduction='mean')
        self.match_loss_func = nn.MSELoss(reduction='mean')
        self.path_loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.args = args
        self.gru_cell = torch.nn.GRUCell(input_size=args.n_hidden, hidden_size=args.n_hidden)
        self.rel_mapping1 = nn.Linear(args.n_hidden, 2 * args.n_hidden)
        self.rel_mapping2 = nn.Linear(2 * args.n_hidden, args.n_hidden)
        # torch.nn.init.xavier_normal_(self.rel_mapping)
        self.rel_prob = nn.Linear(args.n_hidden, num_rel)
        self.rel2hist = nn.Linear(args.n_hidden, args.n_hidden)
        self.rel2hist2 = nn.Linear(args.n_hidden, args.n_hidden)
        # self.orig_score_func = orig_score_func
        # self.orig_score_func = TimeConvTransE(num_ent, args.n_hidden, input_dropout, hidden_dropout, feat_dropout)
        self.num_ent = num_ent
        self.linear_align = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear_gru_hidden = nn.Linear(args.n_hidden, args.n_hidden)
        self.linear_attn = nn.Linear(args.n_hidden, args.n_hidden)

        # self.time_encode = TimeEncode(args.n_hidden, 'cuda:'+ str(args.gpu))
        self.linear_time = nn.Linear(args.n_hidden, args.n_hidden)
        if self.args.path:
            if self.args.path_method.lower() == 'tucker':
                self.W_tk = torch.nn.Parameter(
                    torch.tensor(np.random.uniform(-1, 1, (args.n_hidden, args.n_hidden, args.n_hidden)),
                                 dtype=torch.float, requires_grad=True))
                self.input_dropout = torch.nn.Dropout(0.1)
                self.hidden_dropout1 = torch.nn.Dropout(0.1)
                self.hidden_dropout2 = torch.nn.Dropout(0.1)

                self.bn0 = torch.nn.BatchNorm1d(args.n_hidden)
                self.bn1 = torch.nn.BatchNorm1d(args.n_hidden)

    def forward_back(self, pre_emb, r_emb, time_embs, all_triples, num_rel, partial_embeding=None, cur_ts=None, orig_score_func=None):
        mapped_rel_ = self.rel_mapping1(r_emb)
        mapped_rel = self.rel_mapping2(mapped_rel_)
        # mapped_rel = r_emb
        q_rel = mapped_rel[all_triples[:, 1]]
        history_rel_path = []
        gru_hidden = torch.zeros(partial_embeding.shape[0], self.args.n_hidden)
        gru_hidden = gru_hidden.to('cuda')

        if cur_ts > self.args.path_len:
            start_ts = cur_ts - self.args.path_len
        else:
            start_ts = 0

        if self.args.path_len == -1:
            start_ts = 0

        for ts in range(start_ts, cur_ts):
            cur = partial_embeding[:, ts * num_rel: (ts + 1) * num_rel]
            attn = cur * torch.matmul(self.linear_attn(q_rel), mapped_rel.transpose(1,0))
            attn = F.softmax(torch.where(attn == 0, torch.tensor([-1000000000.0]).cuda(), attn), dim=1)
            rel_path_emb = torch.matmul(attn, mapped_rel)
            gru_hidden = self.gru_cell(rel_path_emb, gru_hidden)
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]
        # predicted_hist = self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]]))

        match_loss = self.match_loss_func(predicted_hist, gru_hidden)
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, time_embs, all_triples, predicted_hist,
                                                   orig_score_func)

        return match_loss, path_loss, path_score

    def forward_forth(self, pre_emb, mapped_rel, time_embs, all_triples, predicted_hist, orig_score_func=None):
        gru_hidden = self.gru_cell(mapped_rel[all_triples[:, 1]], predicted_hist) # complete reasoning path
        sub_emb, obj_emb = pre_emb[all_triples[:, 0]], pre_emb[all_triples[:, 2]]
        score_type = self.args.path_method
        if score_type == 'distmult':
            score_ = torch.matmul(self.linear_align(sub_emb) * gru_hidden, self.linear_align(pre_emb).transpose(1,0))
            score = F.softmax(score_, dim=1)
        elif score_type == 'complex':
            rank = pre_emb.shape[1] // 2
            lhs = self.linear_align(sub_emb)[:, :rank], self.linear_align(sub_emb)[:, rank:]
            rel = gru_hidden[:, :rank], gru_hidden[:, rank:]

            right = self.linear_align(pre_emb)
            right = right[:, :rank], right[:, rank:]
            score_ = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(1, 0) + \
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(1, 0)
        elif score_type == 'tucker':
            x = self.linear_align(sub_emb).view(-1, 1, self.args.n_hidden)
            W_mat = torch.mm(self.linear_gru_hidden(gru_hidden), self.W_tk.view(self.args.n_hidden, -1))
            W_mat = W_mat.view(-1, self.args.n_hidden, self.args.n_hidden)
            x = torch.bmm(x, W_mat)
            x = x.view(-1, self.args.n_hidden)
            score_ = x @ self.linear_align(pre_emb).transpose(1, 0)
        elif score_type == 'orig':
            scores_ob = self.orig_score_func.forward(pre_emb, gru_hidden, time_embs, all_triples).view(-1, self.num_ent)
            score = F.softmax(scores_ob, dim=1)

        labels = torch.zeros_like(score_)
        labels[torch.tensor([i for i in range(all_triples.shape[0])]), all_triples[:, 2]] = 1
        score = torch.sigmoid(score_)
        path_loss = self.loss(score, labels)

        return path_loss, score

    def forward_eval(self, pre_emb, r_emb, time_embs, all_triples, num_rel, partial_embeding=None, cur_ts=None, orig_score_func=None):
        mapped_rel_ = self.rel_mapping1(r_emb)
        mapped_rel = self.rel_mapping2(mapped_rel_)
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]
        # predicted_hist = self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]]))
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, time_embs, all_triples, predicted_hist, orig_score_func)
        return path_loss, path_score

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "convgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "convgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g.ndata.pop('h')
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')



class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, num_times, time_interval, h_dim, opn, history_rate, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False, args=None):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.history_rate = history_rate
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.num_times = num_times
        self.time_interval = time_interval
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu
        self.sin = torch.sin
        self.linear_0 = nn.Linear(num_times, 1)
        self.linear_1 = nn.Linear(num_times, self.h_dim - 1)
        self.tanh = nn.Tanh()
        self.use_cuda = None
        self.args = args

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        # self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        # torch.nn.init.xavier_normal_(self.emb_rel)
        self.LLM_init = self.args.LLM
        self.pure_LLM = self.args.pure_LLM
        self.rel_path = self.args.path
        self.gru_init = True
        self.LLM_path = self.args.LLM_path
        self.with_explanation = False
        if self.LLM_init or self.LLM_path:
            self.linear_rel2inv = nn.Linear(self.h_dim, self.h_dim) # transform relation to inverse
            self.linear_LLM2kg1 = nn.Linear(1024, 1024)
            self.linear_LLM2kg2 = nn.Linear(1024, self.h_dim)
            self.linear_LLM2kg3 = nn.Linear(self.h_dim * self.h_dim, self.h_dim)
            if args.gamma_fix == 1:
                self.gamma = torch.nn.Parameter(torch.Tensor([args.gamma_init]), requires_grad=False).float()
            elif args.gamma_fix == 0:
                self.gamma = torch.nn.Parameter(torch.Tensor([args.gamma_init]), requires_grad=True).float()
            init_path = '../data/{}/'.format(self.args.dataset)
            if args.dataset.lower() == 'icews22':
                raw_emb = []
                for i in range(5):
                    emb_batch = np.load(init_path + 'Rel_middleExpl_combined_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
                print(self.emb_rel_LLM.shape)
            elif args.dataset.lower() == 'icews21':
                raw_emb = []
                for i in range(17):
                    emb_batch = np.load(init_path + 'Rel_Expl_Embedding_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
                print(self.emb_rel_LLM.shape)
            else:
                emb_batch = np.load(init_path + 'Rel_Expl_combined_batch1.npy')
                emb_batch_torch = torch.from_numpy(emb_batch)
                self.emb_rel_LLM = torch.nn.Parameter(emb_batch_torch, requires_grad=False)
                print(self.emb_rel_LLM.shape)
            if self.with_explanation:
                    self.linear_rel2inv2 = nn.Linear(self.h_dim, self.h_dim)  # transform relation to inverse
                    if args.dataset.lower() == 'icews21':
                        exp_emb = []
                        for i in range(17):
                            exp_emb.append(torch.load(init_path + 'Rel_Expl_Embedding_batch' + str(i + 1) + '.pt'))
                        exp_emb_ = torch.cat(exp_emb, dim=0)
                        self.emb_rel_LLM_short = torch.nn.Parameter(exp_emb_, requires_grad=False)
                        print(self.emb_rel_LLM_short.shape)
                    else:
                        self.emb_rel_LLM_short = torch.nn.Parameter(torch.load(init_path + 'Rel_longExpl_both_batch1.pt'), requires_grad=False)
                    self.combine_LLM = nn.Linear(2 * self.h_dim, self.h_dim)
                    self.combine_LLM2 = nn.Linear(self.h_dim, self.h_dim)
                    self.weight = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, 1), requires_grad=True).float().cuda()
            else:
                self.emb_rel_LLM_short = None
            self.act = nn.ReLU()
            self.gru_cell_LLM = torch.nn.GRUCell(input_size=args.n_hidden, hidden_size=args.n_hidden) # another way to learn from LLM init
            if self.with_explanation:
                self.gru_cell_LLM_short = torch.nn.GRUCell(input_size=args.n_hidden, hidden_size=args.n_hidden)
            self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.emb_rel)
        else:
            self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)
        

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, h_dim))


        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))    
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        # add
        self.global_weight = nn.Parameter(torch.Tensor(self.num_ents, 1))
        nn.init.xavier_uniform_(self.global_weight , gain=nn.init.calculate_gain('relu'))
        self.global_bias = nn.Parameter(torch.Tensor(1))
        nn.init.zeros_(self.global_bias)

        # GRU cell for relation evolving
        self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)
        self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # decoder
        if decoder_name == "timeconvtranse":
            self.decoder_ob1 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob2 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.decoder_ob3 = TimeConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re1 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder_re2 = TimeConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
            # self.pathdecoder = PathPredictor(self.args, 2 * self.num_rels, self.num_ents, self.decoder_ob3)
            self.pathdecoder = PathPredictor(self.args, 2 * self.num_rels, self.num_ents, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def init_rel_LLM(self, LLM_rel_emb, LLM_rel_emb_short=None):
        if self.gru_init:
            LLM_rel_emb = self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb))
            gru_hidden = torch.zeros(LLM_rel_emb.shape[0], self.args.n_hidden)
            gru_hidden = gru_hidden.to('cuda')
            for w in range(LLM_rel_emb.shape[1]):
                word_emb = LLM_rel_emb[:, w, :]
                gru_hidden = self.gru_cell_LLM(word_emb, gru_hidden)
            LLM_rel_emb = gru_hidden
            if self.with_explanation:
                LLM_rel_emb_short = self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb_short))
                gru_hidden_short = torch.zeros(LLM_rel_emb_short.shape[0], self.args.n_hidden)
                gru_hidden_short = gru_hidden_short.to('cuda')
                gru_hidden_short = gru_hidden
                for w in range(LLM_rel_emb_short.shape[1]):
                    word_emb = LLM_rel_emb_short[:, w, :]
                    gru_hidden_short = self.gru_cell_LLM(word_emb, gru_hidden_short)
                    # gru_hidden_short = self.gru_cell_LLM(word_emb, gru_hidden_short)
                LLM_rel_emb_short = gru_hidden_short
        else:
            LLM_rel_emb = self.linear_LLM2kg3(self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb)).view(self.num_rels, -1))
        # create inverse mapping
        LLM_rel_emb_inv = self.linear_rel2inv(LLM_rel_emb)
        if self.with_explanation:
            LLM_rel_emb_short_inv = self.linear_rel2inv(LLM_rel_emb_short)
            LLM_emb = torch.cat([LLM_rel_emb, LLM_rel_emb_inv], dim=0)
            LLM_emb_short = torch.cat([LLM_rel_emb_short, LLM_rel_emb_short_inv], dim=0)
            return LLM_emb_short
        else:
            return torch.cat([LLM_rel_emb, LLM_rel_emb_inv], dim=0)

    def forward(self, g_list, static_graph, use_cuda):
        gate_list = []
        degree_list = []

        if self.LLM_init:
            if self.pure_LLM:
                self.emb_rel_ = self.init_rel_LLM(self.emb_rel_LLM, self.emb_rel_LLM_short)
            else:
                self.emb_rel_ = self.emb_rel + 0.0001 * self.init_rel_LLM(self.emb_rel_LLM)
        else:
            self.emb_rel_ = self.emb_rel

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        history_embs = []

        for i, g in enumerate(g_list):
            g = g.to(self.gpu)
            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1],:]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel_, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel_)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel_, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            current_h = self.rgcn.forward(g, self.h, [self.h_0, self.h_0])
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h) if self.layer_norm else self.h
            history_embs.append(self.h)
        return history_embs, static_emb, self.h_0, gate_list, degree_list


    def predict(self, test_graph, num_rels, static_graph, test_triplets, entity_history_vocabulary, rel_history_vocabulary, rel_path_vocabulary, use_cuda, cur_ts, sparse_rel=None, ent2sec=None, rel2obj=None):
        self.use_cuda = use_cuda
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))
            # print(all_triples)
            
            evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda)
            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
            time_embs = self.get_init_time(all_triples)

            score_rel_r = self.rel_raw_mode(embedding, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(embedding, r_emb, time_embs, all_triples, rel_history_vocabulary)
            if self.rel_path:
                if self.LLM_path:
                    _, score_path, _ = self.reasoning_path(embedding, self.init_rel_LLM(self.emb_rel_LLM, self.emb_rel_LLM_short), time_embs, all_triples, rel_path_vocabulary, cur_ts, 'eval')
                else:
                    _, score_path, _ = self.reasoning_path(embedding, r_emb, time_embs, all_triples, rel_path_vocabulary,
                                                         cur_ts, 'eval')
            score_r = self.raw_mode(embedding, r_emb, time_embs, all_triples)
            score_h = self.history_mode(embedding, r_emb, time_embs, all_triples, entity_history_vocabulary)

            # concept
            if self.args.concept:
                score_concept = self.concept_predictor(sparse_rel, ent2sec, rel2obj, all_triples)
            score_rel = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r
            score_rel = torch.log(score_rel)
            if self.rel_path:
                score = self.history_rate * score_h + (1 - self.history_rate) * score_r + self.gamma * score_path

                if self.args.concept:
                    score = score + 0.01 * score_concept
            else:
                score = self.history_rate * score_h + (1 - self.history_rate) * score_r
                if self.args.concept:
                    score = score + 0.01 * score_concept
            score = torch.log(score)

            return all_triples, score, score_rel


    def get_loss(self, glist, triples, static_graph, entity_history_vocabulary, rel_history_vocabulary, rel_path_vocabulary, use_cuda, cur_ts):
        self.use_cuda = use_cuda
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda)
        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]
        time_embs = self.get_init_time(all_triples)

        if self.entity_prediction:
            score_r = self.raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_h = self.history_mode(pre_emb, r_emb, time_embs, all_triples, entity_history_vocabulary)
            if self.rel_path:
                match_loss, path_loss, score_path = self.reasoning_path(pre_emb, self.init_rel_LLM(self.emb_rel_LLM, self.emb_rel_LLM_short), time_embs,
                                                                all_triples, rel_path_vocabulary, cur_ts, 'train')
                score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r + self.gamma * score_path
            else:
                score_en = self.history_rate * score_h + (1 - self.history_rate) * score_r

            scores_en = torch.log(score_en)
            loss_ent += F.nll_loss(scores_en, all_triples[:, 2])
            if self.rel_path:
                loss_ent = loss_ent + match_loss + path_loss
     
        if self.relation_prediction:
            score_rel_r = self.rel_raw_mode(pre_emb, r_emb, time_embs, all_triples)
            score_rel_h = self.rel_history_mode(pre_emb, r_emb, time_embs, all_triples, rel_history_vocabulary)
            if self.LLM_path:
                loss_path, score_rel_path = self.reasoning_path(pre_emb, self.init_rel_LLM(self.emb_rel_LLM), time_embs,
                                                          all_triples, rel_path_vocabulary, cur_ts)
            else:
                loss_path, score_rel_path = self.reasoning_path(pre_emb, r_emb, time_embs,
                                                          all_triples, rel_path_vocabulary, cur_ts)
            if self.rel_path:
                score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r + self.gamma * score_rel_path
            else:
                score_re = self.history_rate * score_rel_h + (1 - self.history_rate) * score_rel_r

            scores_re = torch.log(score_re)
            loss_rel += F.nll_loss(scores_re, all_triples[:, 1])
            if self.rel_path:
                loss_rel += loss_path

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    angle = 90 // len(evolve_embs)
                    # step = (self.angle * math.pi / 180) * (time_step + 1)
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] // self.time_interval
        T_idx = T_idx.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = self.sin(self.weight_t2 * T_idx + self.bias_t2)
        return t1, t2

    def raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_ob = self.decoder_ob1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, self.num_ents)
        score = F.softmax(scores_ob, dim=1)
        return score

    def history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.decoder_ob2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding = global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def rel_raw_mode(self, pre_emb, r_emb, time_embs, all_triples):
        scores_re = self.rdecoder_re1.forward(pre_emb, r_emb, time_embs, all_triples).view(-1, 2 * self.num_rels)
        score = F.softmax(scores_re, dim=1)
        return score

    def rel_history_mode(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        score_global = self.rdecoder_re2.forward(pre_emb, r_emb, time_embs, all_triples, partial_embeding=global_index)
        score_h = score_global
        score_h = F.softmax(score_h, dim=1)
        return score_h

    def reasoning_path(self, pre_emb, r_emb, time_embs, all_triples, history_vocabulary, cur_ts, mode):
        if self.use_cuda:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
            global_index = global_index.to('cuda')
        else:
            global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        # print(global_index.shape)
        # assert 0
        if mode == 'train':
            match_loss, path_loss, score_path = self.pathdecoder.forward_back(pre_emb, r_emb, time_embs, all_triples, 2 * self.num_rels, partial_embeding=global_index, cur_ts=cur_ts)
            return match_loss, path_loss, score_path
        elif mode == 'eval':
            path_loss, score_path = self.pathdecoder.forward_eval(pre_emb, r_emb, time_embs, all_triples, 2 * self.num_rels,
                                                             partial_embeding=global_index, cur_ts=cur_ts)
            return path_loss, score_path, None

    def concept_predictor(self, sparse_rel, ent2sec, rel2obj, all_triples):
        sparse_rel = sparse_rel + [r + self.num_rels for r in sparse_rel]
        freq_rel = torch.tensor(np.delete(np.arange(2 * self.num_rels), sparse_rel)).cuda()
        sparse_rel = torch.tensor(sparse_rel).cuda()
        # print(freq_rel, sparse_rel)
        # assert 0
        freq_rel_emb = self.emb_rel_[freq_rel]
        sparse_rel_emb = self.emb_rel_[sparse_rel]
        sim = F.softmax(torch.matmul(self.emb_rel_, freq_rel_emb.transpose(1, 0)), dim=1)
        rel2obj = rel2obj.cuda()
        freq_rel2obj = rel2obj[freq_rel]
        ent2sec = torch.from_numpy(ent2sec).float().cuda()
        count = torch.matmul(freq_rel2obj, ent2sec)
        freq_rel2concept_prob = F.softmax(count, dim=1)
        rel2concept_prob = torch.matmul(sim, freq_rel2concept_prob)
        rel2concept_score = torch.matmul(rel2concept_prob, ent2sec.transpose(1, 0))
        return rel2concept_score[all_triples[:, 1]]






