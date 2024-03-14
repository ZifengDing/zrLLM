import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
import math
import copy

"""
class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, 2 * input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(2 * input_dim, 2 * input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(2 * input_dim, 2 * input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(2 * input_dim, input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(input_dim, out_dim),
                                    )

    def forward(self, x):
        return self.linear(x)
"""
class PathPredictor(nn.Module):
    def __init__(self, args, num_rel, num_ent):
        super(PathPredictor, self).__init__()
        self.loss = nn.BCELoss(reduction='mean')
        self.match_loss_func = nn.MSELoss(reduction='mean')
        self.path_loss_func = nn.CrossEntropyLoss(reduction='mean')
        self.args = args
        self.gru_cell = torch.nn.GRUCell(input_size=args.embedding_dim, hidden_size=args.embedding_dim)
        self.rel_mapping1 = nn.Linear(args.embedding_dim, 2 * args.embedding_dim)
        self.rel_mapping2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        # torch.nn.init.xavier_normal_(self.rel_mapping)
        self.rel_prob = nn.Linear(args.embedding_dim, num_rel)
        self.rel2hist = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.rel2hist2 = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.num_ent = num_ent
        self.linear_align = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.linear_gru_hidden = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.linear_attn = nn.Linear(args.embedding_dim, args.embedding_dim)

        # self.time_encode = TimeEncode(args.n_hidden, 'cuda:'+ str(args.gpu))
        self.linear_time = nn.Linear(args.embedding_dim, args.embedding_dim)
        if self.args.path:
            if self.args.path_method.lower() == 'tucker':
                self.W_tk = torch.nn.Parameter(
                    torch.tensor(np.random.uniform(-1, 1, (args.embedding_dim, args.embedding_dim, args.embedding_dim)),
                                 dtype=torch.float, requires_grad=True))
                self.input_dropout = torch.nn.Dropout(0.1)
                self.hidden_dropout1 = torch.nn.Dropout(0.1)
                self.hidden_dropout2 = torch.nn.Dropout(0.1)

                self.bn0 = torch.nn.BatchNorm1d(args.embedding_dim)
                self.bn1 = torch.nn.BatchNorm1d(args.embedding_dim)

    def forward_back(self, pre_emb, r_emb, all_triples, num_rel, partial_embeding=None, cur_ts=None, orig_score_func=None):
        mapped_rel_ = self.rel_mapping1(r_emb)
        mapped_rel = self.rel_mapping2(mapped_rel_)
        # mapped_rel = r_emb
        q_rel = mapped_rel[all_triples[:, 1]]
        history_rel_path = []
        gru_hidden = torch.zeros(partial_embeding.shape[0], self.args.embedding_dim)
        gru_hidden = gru_hidden.to('cuda')
        cur_ts_sorted, ts_idx = torch.sort(cur_ts)
        partial_embeding = partial_embeding[ts_idx]
        q_rel = q_rel[ts_idx]
        # print(cur_ts, cur_ts.shape)
        ts_all, example_group = torch.unique_consecutive(cur_ts_sorted, return_counts=True)
        cur_position = 0
        last_ts = 0
        gru_hidden_list = []
        for i, cur_ts in enumerate(ts_all):
            partial_embeding_ = partial_embeding[cur_position:, :]
            q_rel_ = q_rel[cur_position:, :]
            # t_diff =
            for ts in range(last_ts, cur_ts):
                cur = partial_embeding_[:, ts * num_rel: (ts + 1) * num_rel]
                attn = cur * torch.matmul(self.linear_attn(q_rel_), mapped_rel.transpose(1,0))
                attn = F.softmax(torch.where(attn == 0, torch.tensor([-1000000000.0]).cuda(), attn), dim=1)
                rel_path_emb = torch.matmul(attn, mapped_rel)
                gru_hidden = self.gru_cell(rel_path_emb, gru_hidden)
            gru_hidden_list.append(gru_hidden[:example_group[i]])
            gru_hidden = gru_hidden[example_group[i]:]
            last_ts = cur_ts
            cur_position += example_group[i]
        gru_hidden_ = torch.cat(gru_hidden_list, dim=0)
        gru_hidden_all = torch.zeros_like(gru_hidden_).to("cuda")
        gru_hidden_all[ts_idx] = gru_hidden_
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]
        # predicted_hist = self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]]))

        match_loss = self.match_loss_func(predicted_hist, gru_hidden_all)
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, all_triples, predicted_hist,
                                                   orig_score_func)
        return match_loss, path_loss, path_score

    def forward_forth(self, pre_emb, mapped_rel, all_triples, predicted_hist, orig_score_func=None):
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
            x = self.linear_align(sub_emb).view(-1, 1, self.args.embedding_dim)
            W_mat = torch.mm(self.linear_gru_hidden(gru_hidden), self.W_tk.view(self.args.embedding_dim, -1))
            W_mat = W_mat.view(-1, self.args.embedding_dim, self.args.embedding_dim)
            x = torch.bmm(x, W_mat)
            x = x.view(-1, self.args.embedding_dim)
            score_ = x @ self.linear_align(pre_emb).transpose(1, 0)

        labels = torch.zeros_like(score_)
        labels[torch.tensor([i for i in range(all_triples.shape[0])]), all_triples[:, 2]] = 1
        score = torch.sigmoid(score_)
        path_loss = self.loss(score, labels)
        # path_loss = self.path_loss_func(score_, all_triples[:, 2])
        return path_loss, score

    def forward_eval(self, pre_emb, r_emb, all_triples, num_rel, partial_embeding=None, cur_ts=None, orig_score_func=None):
        mapped_rel_ = self.rel_mapping1(r_emb)
        mapped_rel = self.rel_mapping2(mapped_rel_)
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]
        # predicted_hist = self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]]))
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, all_triples, predicted_hist, orig_score_func)
        return path_loss, path_score

class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim),
                                    nn.Dropout(0.4),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(input_dim, out_dim),
                                    )

    def forward(self, x):
        return self.linear(x)


class CENET(nn.Module):
    def __init__(self, num_e, num_rel, num_t, args):
        super(CENET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args
        self.mode_lk = None

        self.LLM_init = args.LLM
        self.pure_LLM = args.pure_LLM
        self.rel_path = args.path
        self.gru_init = True
        self.LLM_path = args.LLM_path
        self.with_explanation = False
        if self.LLM_init or self.LLM_path:
            self.linear_rel2inv = nn.Linear(args.embedding_dim, args.embedding_dim)  # transform relation to inverse
            self.linear_LLM2kg1 = nn.Linear(1024, 1024)
            self.linear_LLM2kg2 = nn.Linear(1024, args.embedding_dim)
            self.linear_LLM2kg3 = nn.Linear(args.embedding_dim * args.embedding_dim, args.embedding_dim)
            self.gamma = torch.nn.Parameter(torch.Tensor([args.gamma_init]), requires_grad=True).float()

            # self.weights_init(self.linear_rel2inv)
            # self.weights_init(self.linear_LLM2kg1)
            # self.weights_init(self.linear_LLM2kg2)
            # self.weights_init(self.linear_LLM2kg3)

            init_path = 'data/{}/'.format(args.dataset)
            if self.args.dataset.lower() == 'icews22':
                raw_emb = []
                for i in range(5):
                    emb_batch = np.load(init_path + 'Rel_middleExpl_combined_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
                print(self.emb_rel_LLM.shape)
            elif self.args.dataset.lower() == 'icews21':
                raw_emb = []
                for i in range(17):
                    emb_batch = np.load(init_path + 'Rel_Expl_Embedding_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                    # raw_emb.append(torch.load(init_path + 'Rel_Expl_combined_batch' + str(i + 1) + '.pt'))
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
                print(self.emb_rel_LLM.shape)
            else:
                emb_batch = np.load(init_path + 'Rel_longExpl_combined_batch1.npy')
                emb_batch_torch = torch.from_numpy(emb_batch)
                self.emb_rel_LLM = torch.nn.Parameter(emb_batch_torch,
                                                      requires_grad=False)
                print(self.emb_rel_LLM.shape)
            self.emb_rel_LLM_short = None
            self.gru_cell_LLM = torch.nn.GRUCell(input_size=args.embedding_dim,
                                                 hidden_size=args.embedding_dim)  # another way to learn from LLM init
            # for param in [self.gru_cell_LLM.weight_ih, self.gru_cell_LLM.weight_hh, self.gru_cell_LLM.bias_ih, self.gru_cell_LLM.bias_hh]:
            #     self.weights_init(param)

            self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel, args.embedding_dim))
            # nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        else:
            self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel, args.embedding_dim))
            # nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))

        if self.args.path:
            self.pathdecoder = PathPredictor(self.args, 2 * self.num_rel, self.num_e)

        # entity relation embedding
        # self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel, args.embedding_dim))
        # nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, args.embedding_dim))
        # nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))

        self.linear_frequency = nn.Linear(self.num_e, args.embedding_dim)

        self.contrastive_hidden_layer = nn.Linear(3 * args.embedding_dim, args.embedding_dim)
        self.contrastive_output_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.oracle_layer = Oracle(3 * args.embedding_dim, 1)
        # self.oracle_layer.apply(self.weights_init)

        self.linear_pred_layer_s1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        # self.weights_init(self.linear_frequency)
        # self.weights_init(self.linear_pred_layer_s1)
        # self.weights_init(self.linear_pred_layer_o1)
        # self.weights_init(self.linear_pred_layer_s2)
        # self.weights_init(self.linear_pred_layer_o2)

        """
        pe = torch.zeros(400, 3 * args.embedding_dim)
        position = torch.arange(0, 400, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, 3 * args.embedding_dim, 2).float() * (-math.log(10000.0) / (3 * args.embedding_dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        """

        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()
        self.oracle_mode = args.oracle_mode
        self.rel_path_vocabulary = None

        print('CENET Initiated')

    def init_rel_LLM(self, LLM_rel_emb, LLM_rel_emb_short=None):
        if self.gru_init:
            LLM_rel_emb = self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb))
            gru_hidden = torch.zeros(LLM_rel_emb.shape[0], self.args.embedding_dim)
            gru_hidden = gru_hidden.to("cuda")
            for w in range(LLM_rel_emb.shape[1]):
                word_emb = LLM_rel_emb[:, w, :]
                gru_hidden = self.gru_cell_LLM(word_emb, gru_hidden)
            LLM_rel_emb = gru_hidden
        else:
            LLM_rel_emb = self.linear_LLM2kg3(self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb)).view(self.num_rel, -1))
        # create inverse mapping
        LLM_rel_emb_inv = self.linear_rel2inv(LLM_rel_emb)

        return torch.cat([LLM_rel_emb, LLM_rel_emb_inv], dim=0)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_block, mode_lk, total_data=None, rel_path_vocabulary=None):
        quadruples, s_history_event_o, o_history_event_s, \
        s_history_label_true, o_history_label_true, s_frequency, o_frequency = batch_block

        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            sub_rank, obj_rank, batch_loss = [None] * 3
            if mode_lk == 'Training':
                return batch_loss
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, batch_loss
            else:
                return None
        self.mode_lk = mode_lk
        self.rel_path_vocabulary = rel_path_vocabulary
        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]
        t = quadruples[:, 3]

        """
        t = (quadruples[:, 3] / 24.0).long()
        time_embedding = self.pe[t]
        """

        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = self.args.lambdax
        o_history_tag[o_history_tag != 0] = self.args.lambdax

        s_non_history_tag[s_history_tag == 1] = -self.args.lambdax
        s_non_history_tag[s_history_tag == 0] = self.args.lambdax

        o_non_history_tag[o_history_tag == 1] = -self.args.lambdax
        o_non_history_tag[o_history_tag == 0] = self.args.lambdax

        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax

        s_frequency = F.softmax(s_frequency, dim=1)
        o_frequency = F.softmax(o_frequency, dim=1)
        s_frequency_hidden = self.tanh(self.linear_frequency(s_frequency))
        o_frequency_hidden = self.tanh(self.linear_frequency(o_frequency))

        if self.LLM_init:
            if self.pure_LLM:
                self.rel_embeds_ = self.init_rel_LLM(self.emb_rel_LLM, self.emb_rel_LLM_short)
            else:
                self.rel_embeds_ = self.rel_embeds + 0.0001 * self.init_rel_LLM(self.emb_rel_LLM)
        else:
            self.rel_embeds_ = self.rel_embeds

        if mode_lk == 'Training':
            s_nce_loss, _, match_loss, path_loss = self.calculate_nce_loss(s, o, r, t, quadruples, self.rel_embeds_[:self.num_rel],
                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                    s_history_tag, s_non_history_tag)
            o_nce_loss, _, match_loss, path_loss = self.calculate_nce_loss(o, s, r, t, quadruples, self.rel_embeds_[self.num_rel:],
                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                    o_history_tag, o_non_history_tag)
            # calculate_spc_loss(self, hidden_lk, actor1, r, rel_embeds, targets):
            s_spc_loss = self.calculate_spc_loss(s, r, self.rel_embeds_[:self.num_rel],
                                                 s_history_label_true, s_frequency_hidden)
            o_spc_loss = self.calculate_spc_loss(o, r, self.rel_embeds_[self.num_rel:],
                                                 o_history_label_true, o_frequency_hidden)
            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            spc_loss = (s_spc_loss + o_spc_loss) / 2.0
            # print('nce loss', nce_loss.item(), ' spc loss', spc_loss.item())

            if self.args.path:
                return self.args.alpha * nce_loss + (1 - self.args.alpha) * spc_loss + match_loss + path_loss
            else:
                return self.args.alpha * nce_loss + (1 - self.args.alpha) * spc_loss

        elif mode_lk in ['Valid']:
            s_nce_loss, s_preds,_ ,_ = self.calculate_nce_loss(s, o, r, t, quadruples, self.rel_embeds_[:self.num_rel],
                                                          self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                          s_history_tag, s_non_history_tag)
            o_nce_loss, o_preds,_ ,_ = self.calculate_nce_loss(o, s, r, t, quadruples, self.rel_embeds_[self.num_rel:],
                                                          self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                          o_history_tag, o_non_history_tag)

            s_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))

            s_ce_loss, s_pred_history_label, s_ce_all_acc = self.oracle_loss(s, r, self.rel_embeds_[:self.num_rel],
                                                                             s_history_label_true, s_frequency_hidden)
            o_ce_loss, o_pred_history_label, o_ce_all_acc = self.oracle_loss(o, r, self.rel_embeds_[self.num_rel:],
                                                                             o_history_label_true, o_frequency_hidden)
            s_total_loss2, sub_rank2 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r, t,
                                                         s_mask, total_data, 's', False)
            o_total_loss2, obj_rank2 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r, t,
                                                         o_mask, total_data, 'o', False)
            batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0

            return sub_rank2, obj_rank2, batch_loss2, \
                sub_rank2, obj_rank2, batch_loss2, \
                sub_rank2, obj_rank2, batch_loss2, \
                (s_ce_all_acc + o_ce_all_acc) / 2

        elif mode_lk in ['Test']:
            s_history_oid = []
            o_history_sid = []

            for i in range(quadruples.shape[0]):
                s_history_oid.append([])
                o_history_sid.append([])
                for con_events in s_history_event_o[i]:
                    s_history_oid[-1] += con_events[:, 1].tolist()
                for con_events in o_history_event_s[i]:
                    o_history_sid[-1] += con_events[:, 1].tolist()

            s_nce_loss, s_preds, _, _ = self.calculate_nce_loss(s, o, r, t, quadruples, self.rel_embeds_[:self.num_rel],
                                                          self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                          s_history_tag, s_non_history_tag)
            o_nce_loss, o_preds, _, _ = self.calculate_nce_loss(o, s, r, r, quadruples, self.rel_embeds_[self.num_rel:],
                                                          self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                          o_history_tag, o_non_history_tag)

            s_ce_loss, s_pred_history_label, s_ce_all_acc = self.oracle_loss(s, r, self.rel_embeds_[:self.num_rel],
                                                                             s_history_label_true, s_frequency_hidden)
            o_ce_loss, o_pred_history_label, o_ce_all_acc = self.oracle_loss(o, r, self.rel_embeds_[self.num_rel:],
                                                                             o_history_label_true, o_frequency_hidden)

            s_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))

            for i in range(quadruples.shape[0]):
                if s_pred_history_label[i].item() > 0.5:
                    s_mask[i, s_history_oid[i]] = 1
                else:
                    s_mask[i, :] = 1
                    s_mask[i, s_history_oid[i]] = 0

                if o_pred_history_label[i].item() > 0.5:
                    o_mask[i, o_history_sid[i]] = 1
                else:
                    o_mask[i, :] = 1
                    o_mask[i, o_history_sid[i]] = 0

            if self.oracle_mode == 'soft':
                s_mask = F.softmax(s_mask, dim=1)
                o_mask = F.softmax(o_mask, dim=1)


            s_total_loss1, sub_rank1 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r, t,
                                                         s_mask, total_data, 's', True)
            o_total_loss1, obj_rank1 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r, t,
                                                         o_mask, total_data, 'o', True)
            batch_loss1 = (s_total_loss1 + o_total_loss1) / 2.0

            s_total_loss2, sub_rank2 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r, t,
                                                         s_mask, total_data, 's', False)
            o_total_loss2, obj_rank2 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r, t,
                                                         o_mask, total_data, 'o', False)
            batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0

            # Ground Truth
            s_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))


            for i in range(quadruples.shape[0]):
                if o[i] in s_history_oid[i]:
                    s_mask_gt[i, s_history_oid[i]] = 1
                else:
                    s_mask_gt[i, :] = 1
                    s_mask_gt[i, s_history_oid[i]] = 0

                if s[i] in o_history_sid[i]:
                    o_mask_gt[i, o_history_sid[i]] = 1
                else:
                    o_mask_gt[i, :] = 1
                    o_mask_gt[i, o_history_sid[i]] = 0

            s_total_loss3, sub_rank3 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r, t,
                                                         s_mask_gt, total_data, 's', True)
            o_total_loss3, obj_rank3 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r, t,
                                                         o_mask_gt, total_data, 'o', True)
            batch_loss3 = (s_total_loss3 + o_total_loss3) / 2.0

            return sub_rank1, obj_rank1, batch_loss1, \
                   sub_rank2, obj_rank2, batch_loss2, \
                   sub_rank3, obj_rank3, batch_loss3, \
                   (s_ce_all_acc + o_ce_all_acc) / 2

        elif mode_lk == 'Oracle':
            print('Oracle Training')
            s_ce_loss, _, _ = self.oracle_loss(s, r, self.rel_embeds_[:self.num_rel],
                                               s_history_label_true, s_frequency_hidden)
            o_ce_loss, _, _ = self.oracle_loss(o, r, self.rel_embeds_[self.num_rel:],
                                               o_history_label_true, o_frequency_hidden)
            return (s_ce_loss + o_ce_loss) / 2.0 + self.oracle_l1(0.01)

    def oracle_loss(self, actor1, r, rel_embeds, history_label, frequency_hidden):
        history_label_pred = F.sigmoid(
            self.oracle_layer(torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1)))
        tmp_label = torch.squeeze(history_label_pred).clone().detach()
        tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        # print('# Bias Ratio', torch.sum(tmp_label).item() / tmp_label.shape[0])
        ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        # print('# CE Accuracy', ce_accuracy)
        ce_loss = self.crossEntropy(torch.squeeze(history_label_pred), torch.squeeze(history_label))
        return ce_loss, history_label_pred, ce_accuracy * tmp_label.shape[0]

    def calculate_nce_loss(self, actor1, actor2, r, t, quadruples, rel_embeds, linear1, linear2, history_tag, non_history_tag):
        preds_raw1 = self.tanh(linear1(
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds1 = F.softmax(preds_raw1.mm(self.entity_embeds.transpose(0, 1)) + history_tag, dim=1)

        preds_raw2 = self.tanh(linear2(
            self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds2 = F.softmax(preds_raw2.mm(self.entity_embeds.transpose(0, 1)) + non_history_tag, dim=1)

        if self.args.path:
            if self.mode_lk == 'Training':
                match_loss, path_loss, score_path = self.reasoning_path(self.entity_embeds,
                                                                            self.init_rel_LLM(self.emb_rel_LLM),
                                                                            quadruples, self.rel_path_vocabulary, t,
                                                                            'train')
            else:
                _, score_path,_ = self.reasoning_path(self.entity_embeds,
                                                                        self.init_rel_LLM(self.emb_rel_LLM),
                                                                        quadruples, self.rel_path_vocabulary, t,
                                                                        'eval')
                match_loss = None
                path_loss = None
            preds = preds1 + preds2 + self.gamma * score_path
        else:
            preds = preds1 + preds2
            match_loss = None
            path_loss = None

        # cro_entr_loss = self.criterion_link(preds1 + preds2, actor2)

        nce = torch.sum(torch.gather(torch.log(preds), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]

        pred_actor2 = torch.argmax(preds, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        # print('# Batch accuracy', accuracy)

        return nce, preds, match_loss, path_loss

    def link_predict(self, nce_loss, preds, ce_loss, actor1, actor2, r, t, trust_musk, all_triples, pred_known, oracle,
                     history_tag=None, case_study=False):
        if case_study:
            # f = open("case_study.txt", "a+")
            # entity2id, relation2id = get_entity_relation_set(self.args.dataset)
            pass

        if oracle:
            preds = torch.mul(preds, trust_musk)
            # print('$Batch After Oracle accuracy:', end=' ')
        else:
            # print('$Batch No Oracle accuracy:', end=' ')
            pass
        # compute the correct triples
        pred_actor2 = torch.argmax(preds, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        # print(accuracy)
        # print('Batch Error', 1 - accuracy)

        total_loss = nce_loss + ce_loss

        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]
            cur_ts = t[i]
            if case_study:
                in_history = torch.where(history_tag[i] > 0)[0]
                not_in_history = torch.where(history_tag[i] < 0)[0]
                print('---------------------------', file=f)
                for hh in range(in_history.shape[0]):
                    print('his:', entity2id[in_history[hh].item()], file=f)

                print(pred_known,
                      'Truth:', entity2id[cur_s.item()], '--', relation2id[cur_r.item()], '--', entity2id[cur_o.item()],
                      'Prediction:', entity2id[pred_actor2[i].item()], file=f)

            o_label = cur_o
            ground = preds[i, cur_o].clone().item()
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]

                    idx_ = torch.nonzero(all_triples[idx, 3] == cur_ts).view(-1)
                    idx_ = idx[idx_]

                    idx = all_triples[idx_, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]

                    idx_ = torch.nonzero(all_triples[idx, 3] == cur_ts).view(-1)
                    idx_ = idx[idx_]

                    idx = all_triples[idx_, 0]

                preds[i, idx] = 0
                preds[i, o_label] = ground

            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
        return total_loss, ranks

    def regularization_loss(self, reg_param):
        # if self.LLM_init:
        #     if self.pure_LLM:
        #         self.rel_embeds_ = self.init_rel_LLM(self.emb_rel_LLM, self.emb_rel_LLM_short)
        #     else:
        #         self.rel_embeds_ = self.rel_embeds + 0.0001 * self.init_rel_LLM(self.emb_rel_LLM)
        # else:
        #     self.rel_embeds_ = self.rel_embeds
        regularization_loss = torch.mean(self.rel_embeds_.pow(2)) + torch.mean(self.entity_embeds.pow(2))
        return regularization_loss * reg_param

    def oracle_l1(self, reg_param):
        reg = 0
        for param in self.oracle_layer.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param

    # contrastive
    def freeze_parameter(self):
        self.rel_embeds_.requires_grad_(False)
        if self.LLM_init:
            self.linear_rel2inv.requires_grad_(False)
            self.linear_LLM2kg1.requires_grad_(False)
            self.linear_LLM2kg2.requires_grad_(False)
            self.linear_LLM2kg3.requires_grad_(False)
            self.gamma.requires_grad_(False)
            self.gru_cell_LLM.requires_grad_(False)
        self.entity_embeds.requires_grad_(False)
        self.linear_pred_layer_s1.requires_grad_(False)
        self.linear_pred_layer_o1.requires_grad_(False)
        self.linear_pred_layer_s2.requires_grad_(False)
        self.linear_pred_layer_o2.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.contrastive_hidden_layer.requires_grad_(False)
        self.contrastive_output_layer.requires_grad_(False)

    def contrastive_layer(self, x):
        # Implement from the encoder E to the projection network P
        # x = F.normalize(x, dim=1)
        x = self.contrastive_hidden_layer(x)
        # x = F.relu(x)
        # x = self.contrastive_output_layer(x)
        # Normalize to unit hypersphere
        # x = F.normalize(x, dim=1)
        return x

    def calculate_spc_loss(self, actor1, r, rel_embeds, targets, frequency_hidden):
        projections = self.contrastive_layer(
            torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1))
        targets = torch.squeeze(targets)
        """if np.random.randint(0, 10) < 1 and torch.sum(targets) / targets.shape[0] < 0.65 and torch.sum(targets) / targets.shape[0] > 0.35:
            np.savetxt("xx.tsv", projections.detach().cpu().numpy(), delimiter="\t")
            np.savetxt("yy.tsv", targets.detach().cpu().numpy(), delimiter="\t")
        """
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = to_device(targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets)
        mask_anchor_out = to_device(1 - torch.eye(exp_dot_tempered.shape[0]))
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return 0
        return supervised_contrastive_loss

    def reasoning_path(self, pre_emb, r_emb, all_triples, history_vocabulary, cur_ts, mode):
        global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        global_index = global_index.to('cuda')

        if mode == 'train':
            match_loss, path_loss, score_path = self.pathdecoder.forward_back(pre_emb, r_emb, all_triples, 2 * self.num_rel, partial_embeding=global_index, cur_ts=cur_ts)
            return match_loss, path_loss, score_path
        elif mode == 'eval':
            path_loss, score_path = self.pathdecoder.forward_eval(pre_emb, r_emb, all_triples, 2 * self.num_rel,
                                                             partial_embeding=global_index, cur_ts=cur_ts)
            return path_loss, score_path, None
