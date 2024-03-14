import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.init import xavier_normal_
from .odeblock import ODEBlock
from .MGCN import *
from .MGCNLayer import *

class HistoryEncoderGRU(nn.Module):
    def __init__(self, args):
        super(HistoryEncoderGRU, self).__init__()
        # self.config = config
        self.gru_cell = torch.nn.GRUCell(input_size=args.embsize,
                                           hidden_size=args.embsize)

    def set_hiddenx(self, batch_size):
        """Set hidden layer parameters. Initialize to 0"""
        if self.config['cuda']:
            self.hx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
            # self.cx = torch.zeros(batch_size, self.config['state_dim'], device='cuda')
        else:
            self.hx = torch.zeros(batch_size, self.config['state_dim'])
            # self.cx = torch.zeros(batch_size, self.config['state_dim'])

    def forward(self, prev_action, mask):
        """mask: True if NO_OP. ON_OP does not affect history coding results"""
        self.hx_ = self.gru_cell(prev_action, self.hx)
        self.hx = torch.where(mask, self.hx, self.hx_)
        return self.hx

class PathPredictor(nn.Module):
    def __init__(self, args, num_rel, num_ent, input_dropout, hidden_dropout, feat_dropout):
        super(PathPredictor, self).__init__()
        self.loss = nn.BCELoss(reduction='mean')
        self.match_loss_func = nn.MSELoss(reduction='mean')
        self.path_loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.args = args
        self.gru_cell = torch.nn.GRUCell(input_size=args.embsize, hidden_size=args.embsize)
        self.rel_mapping1 = nn.Linear(args.embsize, 2 * args.embsize)
        self.rel_mapping2 = nn.Linear(2 * args.embsize, args.embsize)
        self.rel_prob = nn.Linear(args.embsize, num_rel)
        self.rel2hist = nn.Linear(args.embsize, args.embsize)
        self.rel2hist2 = nn.Linear(args.embsize, args.embsize)
        self.num_ent = num_ent
        self.linear_align = nn.Linear(args.embsize, args.embsize)
        self.linear_gru_hidden = nn.Linear(args.embsize, args.embsize)
        self.linear_attn = nn.Linear(args.embsize, args.embsize)

        self.linear_time = nn.Linear(args.embsize, args.embsize)
        if self.args.path:
            if self.args.path_method.lower() == 'tucker':
                self.W_tk = torch.nn.Parameter(
                    torch.tensor(np.random.uniform(-1, 1, (args.embsize, args.embsize, args.embsize)),
                                 dtype=torch.float, requires_grad=True))
                self.input_dropout = torch.nn.Dropout(0.1)
                self.hidden_dropout1 = torch.nn.Dropout(0.1)
                self.hidden_dropout2 = torch.nn.Dropout(0.1)

                self.bn0 = torch.nn.BatchNorm1d(args.embsize)
                self.bn1 = torch.nn.BatchNorm1d(args.embsize)

    def forward_back(self, pre_emb, r_emb, all_triples, num_rel, partial_embeding=None, cur_ts=None, orig_score_func=None):
        mapped_rel_ = self.rel_mapping1(r_emb)
        mapped_rel = self.rel_mapping2(mapped_rel_)
        # mapped_rel = r_emb
        q_rel = mapped_rel[all_triples[:, 1]]
        history_rel_path = []
        gru_hidden = torch.zeros(partial_embeding.shape[0], self.args.embsize)
        gru_hidden = gru_hidden.to(self.args.device)
        for ts in range(cur_ts):
            cur = partial_embeding[:, ts * num_rel: (ts + 1) * num_rel]
            attn = cur * torch.matmul(self.linear_attn(q_rel), mapped_rel.transpose(1,0))
            attn = F.softmax(torch.where(attn == 0, torch.tensor([-1000000000.0]).to(self.args.device), attn), dim=1)
            rel_path_emb = torch.matmul(attn, mapped_rel)
            gru_hidden = self.gru_cell(rel_path_emb, gru_hidden)
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]
        # predicted_hist = self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]]))

        match_loss = self.match_loss_func(predicted_hist, gru_hidden)
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, all_triples, predicted_hist,
                                                   orig_score_func)
        return match_loss, path_loss, path_score

    def forward_forth(self, pre_emb, mapped_rel, all_triples, predicted_hist, orig_score_func=None):
        gru_hidden = self.gru_cell(mapped_rel[all_triples[:, 1]], predicted_hist) # complete reasoning path
        sub_emb, obj_emb = pre_emb[all_triples[:, 0]], pre_emb[all_triples[:, 2]]
        score_type = self.args.path_method
        if score_type == 'distmult':
            # score_ = torch.matmul(self.linear_distmult(sub_emb) * self.linear_gru_hidden(gru_hidden), self.linear_distmult(pre_emb).transpose(1,0))
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
            x = self.linear_align(sub_emb).view(-1, 1, self.args.embsize)
            W_mat = torch.mm(self.linear_gru_hidden(gru_hidden), self.W_tk.view(self.args.embsize, -1))
            W_mat = W_mat.view(-1, self.args.embsize, self.args.embsize)
            x = torch.bmm(x, W_mat)
            x = x.view(-1, self.args.embsize)
            score_ = x @ self.linear_align(pre_emb).transpose(1, 0)

        labels = torch.zeros_like(score_)
        labels[torch.tensor([i for i in range(all_triples.shape[0])]), all_triples[:, 2]] = 1
        score = torch.sigmoid(score_)
        path_loss = self.loss(score, labels)

        return path_loss, score

    def forward_eval(self, pre_emb, r_emb, all_triples, num_rel, partial_embeding=None, cur_ts=None, orig_score_func=None):
        mapped_rel_ = self.rel_mapping1(r_emb)
        mapped_rel = self.rel_mapping2(mapped_rel_)
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]
        # predicted_hist = self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]]))
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, all_triples, predicted_hist, orig_score_func)
        return path_loss, path_score

class TANGO(nn.Module):
    def __init__(self, num_e, num_rel, params, device, logger):
        super().__init__()

        self.num_e = num_e
        self.num_rel = num_rel
        self.p = params
        self.core = self.p.gde_core
        self.core_layer = self.p.core_layer
        self.score_func = self.p.score_func
        self.solver = self.p.solver
        self.rtol = self.p.rtol
        self.atol = self.p.atol
        self.device = self.p.device
        self.initsize = self.p.initsize
        self.adjoint_flag = self.p.adjoint_flag
        self.drop = self.p.dropout
        self.hidsize = self.p.hidsize
        self.embsize = self.p.embsize

        self.device = device
        self.logger = logger
        if self.p.activation.lower() == 'tanh':
            self.act = torch.tanh
        elif self.p.activation.lower() == 'relu':
            self.act = F.relu
        elif self.p.activation.lower() == 'leakyrelu':
            self.act = F.leaky_relu

        # define loss
        self.loss = torch.nn.CrossEntropyLoss()

        self.LLM_init = self.p.LLM
        self.pure_LLM = self.p.pure_LLM
        self.rel_path = self.p.path
        self.gru_init = True
        self.LLM_path = self.p.LLM_path
        self.with_explanation = False
        if self.LLM_init or self.LLM_path:
            self.linear_rel2inv = nn.Linear(self.embsize, self.embsize)  # transform relation to inverse
            self.linear_LLM2kg1 = nn.Linear(1024, 1024)
            self.linear_LLM2kg2 = nn.Linear(1024, self.embsize)
            self.linear_LLM2kg3 = nn.Linear(self.embsize * self.embsize, self.embsize)
            if self.p.gamma_fix == 1:
                self.gamma = torch.nn.Parameter(torch.Tensor([self.p.gamma_init]), requires_grad=False).float()
            elif self.p.gamma_fix == 0:
                self.gamma = torch.nn.Parameter(torch.Tensor([self.p.gamma_init]), requires_grad=True).float()
            # self.eta = torch.nn.Parameter(torch.Tensor([0.01]), requires_grad=True).float()
            init_path = '{}/'.format(self.p.dataset)
            if self.p.dataset.lower() == 'icews22':
                raw_emb = []
                for i in range(5):
                    emb_batch = np.load(init_path + 'Rel_middleExpl_combined_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
            elif self.p.dataset.lower() == 'icews21':
                raw_emb = []
                for i in range(17):
                    emb_batch = np.load(init_path + 'Rel_Expl_Embedding_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
            else:
                emb_batch = np.load(init_path + 'Rel_longExpl_both_batch1.npy')
                emb_batch_torch = torch.from_numpy(emb_batch)
                self.emb_rel_LLM = torch.nn.Parameter(emb_batch_torch,
                                                      requires_grad=False)
            self.emb_rel_LLM_short = None
            self.gru_cell_LLM = torch.nn.GRUCell(input_size=self.embsize,
                                                 hidden_size=self.embsize)  # another way to learn from LLM init

            self.emb_r = self.get_param((self.num_rel * 2, self.initsize))
        else:
            self.emb_r = self.get_param((self.num_rel * 2, self.initsize))

        self.pathdecoder = PathPredictor(self.p, 2 * self.num_rel, self.num_e, self.drop, self.drop,
                                         self.drop)

        # define entity and relation embeddings
        self.emb_e = self.get_param((self.num_e, self.initsize))
        # self.emb_r = self.get_param((self.num_rel * 2, self.initsize))

        # define graph ode core
        self.gde_func = self.construct_gde_func()

        # define ode block
        self.odeblock = self.construct_GDEBlock(self.gde_func)

        # #######
        # self.MGCN = self.odeblock.odefunc
        # #######

        # define jump modules
        if self.p.jump:
            self.jump, self.jump_weight = self.Jump()
            self.gde_func.jump = self.jump
            self.gde_func.jump_weight = self.jump_weight

        # score function TuckER
        if self.score_func.lower() == "tucker":
            self.W_tk, self.input_dropout, self.hidden_dropout1, self.hidden_dropout2, self.bn0, self.bn1 = self.TuckER()

    def get_param(self, shape):
        # a function to initialize embedding
        param = Parameter(torch.empty(shape, requires_grad=True, device=self.device))
        torch.nn.init.xavier_normal_(param.data)
        return param

    def add_base(self):
        model = MGCNLayerWrapper(None, None, self.num_e, self.num_rel, self.act, drop1=self.drop, drop2=self.drop,
                                       sub=None, rel=None, params=self.p)
        model.to(self.device)
        return model

    def construct_gde_func(self):
        gdefunc = self.add_base()
        return gdefunc

    def construct_GDEBlock(self, gdefunc):
        gde = ODEBlock(odefunc=gdefunc, method=self.solver, atol=self.atol, rtol=self.rtol, adjoint=self.adjoint_flag).to(self.device)
        return gde

    def TuckER(self):
        W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.hidsize, self.hidsize, self.hidsize)),
                                    dtype=torch.float, device=self.device, requires_grad=True))
        input_dropout = torch.nn.Dropout(self.drop)
        hidden_dropout1 = torch.nn.Dropout(self.drop)
        hidden_dropout2 = torch.nn.Dropout(self.drop)

        bn0 = torch.nn.BatchNorm1d(self.hidsize)
        bn1 = torch.nn.BatchNorm1d(self.hidsize)

        input_dropout.to(self.device)
        hidden_dropout1.to(self.device)
        hidden_dropout2.to(self.device)
        bn0.to(self.device)
        bn1.to(self.device)

        return W, input_dropout, hidden_dropout1, hidden_dropout2, bn0, bn1

    def Jump(self):
        if self.p.rel_jump:
            jump = MGCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p, isjump=True, diag=True)
        else:
            jump = GCNConvLayer(self.hidsize, self.hidsize, act=self.act, params=self.p)

        jump.to(self.device)
        jump_weight = torch.FloatTensor([self.p.jump_init]).to(self.device)
        return jump, jump_weight

    def loss_comp(self, sub, rel, emb, label, core, obj=None, score_path=None):
        score = self.score_comp(sub, rel, emb, core)
        if self.rel_path:
            score = score + self.gamma * score_path
            # print(score.shape)
        return self.loss(score, obj)


    def score_comp(self, sub, rel, emb, core):
        sub_emb, rel_emb, all_emb = self.find_related(sub, rel, emb)
        if self.score_func.lower() == 'distmult':
            obj_emb = torch.cat([torch.index_select(self.emb_e, 0, sub), sub_emb], dim=1) * rel_emb.repeat(1,2)
            s = torch.mm(obj_emb, torch.cat([self.emb_e, all_emb], dim=1).transpose(1,0))

        if self.score_func.lower() == 'tucker':
            x = self.bn0(sub_emb)
            x = self.input_dropout(x)
            x = x.view(-1, 1, sub_emb.size(1))

            W_mat = torch.mm(rel_emb, self.W_tk.view(rel_emb.size(1), -1))
            W_mat = W_mat.view(-1, sub_emb.size(1), sub_emb.size(1))
            W_mat = self.hidden_dropout1(W_mat)

            x = torch.bmm(x, W_mat)
            x = x.view(-1, sub_emb.size(1))
            x = self.bn1(x)
            x = self.hidden_dropout2(x)
            s = torch.mm(x, all_emb.transpose(1, 0))

        if self.score_func.lower() == 'complex':
            rank = sub_emb.shape[1] // 2
            lhs = sub_emb[:, :rank], sub_emb[:, rank:]
            rel = rel_emb[:, :rank], rel_emb[:, rank:]

            right = all_emb
            right = right[:, :rank], right[:, rank:]
            s = (lhs[0] * rel[0] - lhs[1] * rel[1]) @ right[0].transpose(0, 1) +\
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ right[1].transpose(0, 1)

        return s

    def find_related(self, sub, rel, emb):
        x = emb[:self.num_e,:]
        r = emb[self.num_e:,:]
        assert x.shape[0] == self.num_e
        assert r.shape[0] == self.num_rel * 2
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, rel_emb, x

    def push_data(self, *args):
        out_args = []
        for arg in args:
            arg = [_arg.to(self.device) for _arg in arg]
            out_args.append(arg)
        return out_args

    def forward(self, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list,
                edge_type_list, edge_id_jump, edge_w_jump, rel_jump, rel_path_vocabulary=None, cur_ts=None, triples=None):
        # self.test_flag = 0

        # push data onto gpu
        if self.p.jump:
            [sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                        self.push_data(sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list] = \
                        self.push_data(sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list)

        if self.LLM_init:
            if self.pure_LLM:
                self.emb_r_ = self.init_rel_LLM(self.emb_rel_LLM, self.emb_rel_LLM_short)
            else:
                self.emb_r_ = self.emb_r + 0.0001 * self.init_rel_LLM(self.emb_rel_LLM)
        else:
            self.emb_r_ = self.emb_r

        emb = torch.cat([self.emb_e, self.emb_r_], dim=0)

        for i in range(len(times)):
            self.odeblock.odefunc.set_graph(edge_index_list[i], edge_type_list[i])
            if i != (len(times) - 1):
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight, False)

                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=times[i+1], cheby_grid=self.p.cheby_grid)
            else:
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                    
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)

                if self.rel_path:
                    match_loss, path_loss, score_path = self.reasoning_path(emb[:self.num_e,:],
                                                                            self.init_rel_LLM(self.emb_rel_LLM),
                                                                            torch.cat([sub_tar[0].unsqueeze(1), rel_tar[0].unsqueeze(1), obj_tar[0].unsqueeze(1)], dim=1), rel_path_vocabulary, cur_ts,
                                                                            'train')
                else: score_path = None

                loss = self.loss_comp(sub_tar[0], rel_tar[0], emb, lab_tar[0], self.odeblock.odefunc,
                                      obj=obj_tar[0], score_path=score_path)

        if self.rel_path:
            return loss + match_loss + 1.2 * path_loss
        else:
            return loss

    def forward_eval(self, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump, rel_path_vocabulary=None, cur_ts=None, triples=None):
        # push data onto gpu

        if self.p.jump:
            [times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [times, tar_times, edge_index_list, edge_type_list, edge_index_list] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_index_list)

        if self.LLM_init:
            if self.pure_LLM:
                self.emb_r_ = self.init_rel_LLM(self.emb_rel_LLM, self.emb_rel_LLM_short)
            else:
                self.emb_r_ = self.emb_r + 0.0001 * self.init_rel_LLM(self.emb_rel_LLM)
        else:
            self.emb_r_ = self.emb_r

        emb = torch.cat([self.emb_e, self.emb_r_], dim=0)
        for i in range(len(times)):
            self.odeblock.odefunc.set_graph(edge_index_list[i], edge_type_list[i])
            if i != (len(times) - 1):
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=times[i + 1], cheby_grid=self.p.cheby_grid)
            else:
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)

        time_back = tar_times[0] / 0.1 * torch.tensor(5808, dtype=torch.float)
        return time_back, emb

    def init_rel_LLM(self, LLM_rel_emb, LLM_rel_emb_short=None):
        if self.gru_init:
            LLM_rel_emb = self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb))
            gru_hidden = torch.zeros(LLM_rel_emb.shape[0], self.p.embsize)
            gru_hidden = gru_hidden.to(self.device)
            for w in range(LLM_rel_emb.shape[1]):
                word_emb = LLM_rel_emb[:, w, :]
                gru_hidden = self.gru_cell_LLM(word_emb, gru_hidden)
            LLM_rel_emb = gru_hidden
        else:
            LLM_rel_emb = self.linear_LLM2kg3(self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb)).view(self.num_rels, -1))
        # create inverse mapping
        LLM_rel_emb_inv = self.linear_rel2inv(LLM_rel_emb)

        return torch.cat([LLM_rel_emb, LLM_rel_emb_inv], dim=0)

    def reasoning_path(self, pre_emb, r_emb, all_triples, history_vocabulary, cur_ts, mode):
        global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        global_index = global_index.to(self.device)
        if mode == 'train':
            match_loss, path_loss, score_path = self.pathdecoder.forward_back(pre_emb, r_emb, all_triples, 2 * self.num_rel, partial_embeding=global_index, cur_ts=cur_ts)
            return match_loss, path_loss, score_path
        elif mode == 'eval':
            path_loss, score_path = self.pathdecoder.forward_eval(pre_emb, r_emb, all_triples, 2 * self.num_rel,
                                                             partial_embeding=global_index, cur_ts=cur_ts)
            return path_loss, score_path, None

    def forward_foul(self, sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list,
                edge_type_list, edge_id_jump, edge_w_jump, rel_jump):
        # self.test_flag = 0

        # push data onto gpu
        if self.p.jump:
            [sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list, edge_id_jump,
             edge_w_jump, rel_jump] = \
                self.push_data(sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list,
                               edge_id_jump, edge_w_jump, rel_jump)
        else:
            [sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list] = \
                self.push_data(sub_tar, rel_tar, obj_tar, lab_tar, times, tar_times, edge_index_list, edge_type_list)

        emb = torch.cat([self.emb_e, self.emb_r], dim=0)

        for i in range(len(times)):
            self.odeblock.odefunc.set_graph(edge_index_list[i], edge_type_list[i])
            if i != (len(times) - 1):
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)

                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=times[i + 1], cheby_grid=self.p.cheby_grid)
            else:
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)

                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)

        self.odeblock.odefunc.set_graph(edge_index_list[-1], edge_type_list[-1])
        emb = self.odeblock.forward_nobatch(emb, start=times[-1], end=tar_times[0], cheby_grid=self.p.cheby_grid)
        # emb = self.odeblock.odefunc(None, emb)
        loss = self.loss_comp(sub_tar[0], rel_tar[0], emb, lab_tar[0], self.odeblock.odefunc,
                                  obj=obj_tar[0])

        return loss

    def forward_eval_foul(self, times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump):
        # push data onto gpu

        if self.p.jump:
            [times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_id_jump, edge_w_jump, rel_jump)
        else:
            [times, tar_times, edge_index_list, edge_type_list, edge_index_list] = \
                self.push_data(times, tar_times, edge_index_list, edge_type_list, edge_index_list)

        emb = torch.cat([self.emb_e, self.emb_r], dim=0)
        for i in range(len(times)):
            self.odeblock.odefunc.set_graph(edge_index_list[i], edge_type_list[i])
            if i != (len(times) - 1):
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=times[i + 1], cheby_grid=self.p.cheby_grid)
            else:
                # ODE
                if self.p.jump:
                    if self.p.rel_jump:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump,
                                                           jumpw=self.jump_weight,
                                                           skip=False, rel_jump=rel_jump[i])
                    else:
                        self.odeblock.odefunc.set_jumpfunc(edge_id_jump[i], edge_w_jump[i], self.jump, self.jump_weight,
                                                           False)
                emb = self.odeblock.forward_nobatch(emb, start=times[i], end=tar_times[0], cheby_grid=self.p.cheby_grid)

        self.odeblock.odefunc.set_graph(edge_index_list[-1], edge_type_list[-1])
        emb = self.odeblock.forward_nobatch(emb, start=times[-1], end=tar_times[0], cheby_grid=self.p.cheby_grid)

        time_back = tar_times[0] / 0.1 * torch.tensor(5808, dtype=torch.float)
        return time_back, emb