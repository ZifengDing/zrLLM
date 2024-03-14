import torch
from torch.nn import Parameter, init
from torch import nn
import torch.nn.functional as F
import numpy as np
from config import args


class link_prediction(nn.Module):
    def __init__(self, i_dim, h_dim, num_rels, num_times, use_cuda=False):
        super(link_prediction, self).__init__()

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_times = num_times
        self.use_cuda = use_cuda
        self.drop = 0.3

        self.ent_init_embeds = nn.Parameter(torch.Tensor(i_dim, h_dim))
        self.w_relation = nn.Parameter(torch.Tensor(2 * num_rels, h_dim))
        self.tim_init_embeds = nn.Parameter(torch.Tensor(1, h_dim))

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.args=args
        self.LLM_init = self.args.LLM_init
        self.pure_LLM = self.args.pure_LLM
        self.rel_path = self.args.path
        self.gru_init = True
        self.LLM_path = self.args.LLM_path
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # self.W_tk, self.input_dropout, self.hidden_dropout1, self.hidden_dropout2, self.bn0, self.bn1 = self.TuckER()
        self.pathdecoder = PathPredictor(self.args, 2 * self.num_rels, self.i_dim)
        self.match_loss=None
        self.path_loss=None

        if self.LLM_init or self.LLM_path:
            self.linear_rel2inv = nn.Linear(args.embedding_dim, args.embedding_dim)  # transform relation to inverse
            self.linear_LLM2kg1 = nn.Linear(1024, 1024)
            self.linear_LLM2kg2 = nn.Linear(1024, args.embedding_dim)
            self.linear_LLM2kg3 = nn.Linear(args.embedding_dim * args.embedding_dim, args.embedding_dim)
            self.linear_for_path = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
            if args.gamma_fix == 1:
                self.gamma = torch.nn.Parameter(torch.Tensor([args.gamma_init]), requires_grad=False).float()
            elif args.gamma_fix == 0:
                self.gamma = torch.nn.Parameter(torch.Tensor([args.gamma_init]), requires_grad=True).float()
            # self.gamma = torch.nn.Parameter(torch.Tensor([args.gamma_init]), requires_grad=True).float()
            # self.gamma = torch.nn.Parameter(torch.Tensor([args.gamma_init]), requires_grad=False).float()
            # self.eta = torch.nn.Parameter(torch.Tensor([0.01]), requires_grad=True).float()

            self.gru_cell_LLM = torch.nn.GRUCell(input_size=args.embedding_dim,
                                        hidden_size=self.h_dim)  # another way to learn from LLM init

            init_path = 'data/{}/'.format(args.dataset)
            if args.dataset.lower() == 'icews22':
                raw_emb = []
                for i in range(5):
                    emb_batch = np.load(init_path + 'Rel_middleExpl_combined_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                    # raw_emb.append(torch.load(init_path+ 'Rel_Expl_Embedding_batch' + str(i + 1) + '.pt'))
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
                print("embedding shape:" + str(self.emb_rel_LLM.shape))
            elif args.dataset.lower() == 'icews21':
                raw_emb = []
                for i in range(17):
                    emb_batch = np.load(init_path + 'Rel_Expl_Embedding_batch' + str(i + 1) + '.npy')
                    emb_batch_torch = torch.from_numpy(emb_batch)
                    raw_emb.append(emb_batch_torch)
                    # raw_emb.append(torch.load(init_path + 'Rel_Expl_Embedding_batch' + str(i + 1) + '.pt'))
                raw_emb_ = torch.cat(raw_emb, dim=0)
                self.emb_rel_LLM = torch.nn.Parameter(raw_emb_, requires_grad=False)
               
                print(self.emb_rel_LLM.shape) 
            else:
                emb_batch = np.load(init_path + 'Rel_longExpl_both_batch1.npy')
                emb_batch_torch = torch.from_numpy(emb_batch)
                self.emb_rel_LLM = torch.nn.Parameter(emb_batch_torch, requires_grad=False)
                # self.emb_rel_LLM = torch.nn.Parameter(torch.load(init_path + 'Rel_Expl_Embedding_batch1.pt'),
                #                                         requires_grad=False)
                
                print(self.emb_rel_LLM.shape)
        self.generate_mode = Generate_mode(h_dim, h_dim, self.i_dim)
        self.copy_mode = Copy_mode(self.h_dim, self.i_dim, use_cuda)
        self.reset_parameters()

    def init_rel_LLM(self, LLM_rel_emb):
        LLM_rel_emb = self.linear_LLM2kg2(self.linear_LLM2kg1(LLM_rel_emb))
        gru_hidden = torch.zeros(LLM_rel_emb.shape[0], self.h_dim).to("cuda")
        for w in range(LLM_rel_emb.shape[1]):
            word_emb = LLM_rel_emb[:, w, :]
            gru_hidden = self.gru_cell_LLM(word_emb, gru_hidden)
        LLM_rel_emb = gru_hidden
        LLM_rel_emb_inv = self.linear_rel2inv(LLM_rel_emb)
    

        return torch.cat([LLM_rel_emb, LLM_rel_emb_inv], dim=0)
       
           
    
    def get_param(self, shape):
        # a function to initialize embedding
        param = Parameter(torch.empty(shape, requires_grad=True, device=self.device))
        torch.nn.init.xavier_normal_(param.data)
        return param

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_init_embeds,
                                gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.tim_init_embeds,
                                gain=nn.init.calculate_gain('relu'))
    
    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList[:, 3] / args.time_stamp
        init_tim = torch.Tensor(self.num_times, self.h_dim)
        for i in range(self.num_times):
            init_tim[i] = torch.Tensor(self.tim_init_embeds.cpu().detach().numpy().reshape(self.h_dim)) * (i + 1)
        init_tim = init_tim.to('cuda')
        T = init_tim[T_idx]
        return T
    
    def reasoning_path(self, pre_emb, r_emb, all_triples, history_vocabulary, cur_ts, mode):
        global_index = torch.Tensor(np.array(history_vocabulary.cpu(), dtype=float))
        global_index = global_index.to(self.device)
        if mode == 'train':
            match_loss, path_loss, score_path = self.pathdecoder.forward_back(pre_emb, r_emb, all_triples, 2 * self.num_rels, partial_embeding=global_index, cur_ts=cur_ts)
            return match_loss, path_loss, score_path
        elif mode == 'eval':
            path_loss, score_path = self.pathdecoder.forward_eval(pre_emb, r_emb, all_triples, 2 * self.num_rels,
                                                             partial_embeding=global_index, cur_ts=cur_ts)
            return path_loss, score_path, None

    
    def get_raw_m_t(self, quadrupleList):
        h_idx = quadrupleList[:, 0]
        r_idx = quadrupleList[:, 1]
        t_idx = quadrupleList[:, 2]

        h = self.ent_init_embeds[h_idx]
        if self.LLM_init:
            r=self.init_rel_LLM(self.emb_rel_LLM)[r_idx]
        else:
            r = self.w_relation[r_idx]
        return h, r
    
    def get_raw_m_t_sub(self, quadrupleList):
        h_idx = quadrupleList[:, 0]
        r_idx = quadrupleList[:, 1]
        t_idx = quadrupleList[:, 2]

        t = self.ent_init_embeds[t_idx]
        if self.LLM_init:
            r= self.init_rel_LLM(self.emb_rel_LLM)[r_idx]
        else:
            r = self.w_relation[r_idx]
        return t, r


    def forward(self, quadruple, copy_vocabulary, rel_path_vocabulary, cur_ts, entity):
        if self.LLM_init:
            if self.args.pure_LLM:
                emb_r_ = self.init_rel_LLM(self.emb_rel_LLM)
            else:
                emb_r_ = self.w_relation + 0.0001 * self.init_rel_LLM(self.emb_rel_LLM)
        else:
            emb_r_ = self.w_relation
        emb = torch.cat([self.ent_init_embeds,emb_r_], dim=0)
        if entity == 'object':
            h, r = self.get_raw_m_t(quadruple)
            T = self.get_init_time(quadruple)
            score_g = self.generate_mode(h, r, T, entity)
            score_c = self.copy_mode(h, r, T, copy_vocabulary, entity)
            if self.args.path:
                self.match_loss, self.path_loss, score_path = self.reasoning_path(emb[:self.i_dim,:], 
                                                                    self.init_rel_LLM(self.emb_rel_LLM),
                                                                    quadruple, rel_path_vocabulary, cur_ts,
                                                                    'train')
    

        if entity == 'subject':
            t, r = self.get_raw_m_t_sub(quadruple)
            T = self.get_init_time(quadruple)
            score_g = self.generate_mode(t, r, T, entity)
            score_c = self.copy_mode(t, r, T, copy_vocabulary, entity)
            if self.args.path:
                self.match_loss, self.path_loss, score_path = self.reasoning_path(emb[:self.i_dim,:],
                                                                    self.init_rel_LLM(self.emb_rel_LLM),
                                                                    quadruple, rel_path_vocabulary, cur_ts,
                                                                    'train')
    

       
        a = args.alpha
        score = score_c * a + score_g * (1-a)
        if self.rel_path:
            score = score + self.gamma * score_path
        score = torch.log(score)
        return score
    
    def forward_eval(self, quadruple, copy_vocabulary, rel_path_vocabulary, cur_ts, entity):
        if self.LLM_init:
            if self.args.pure_LLM:
                emb_r_ = self.init_rel_LLM(self.emb_rel_LLM)
            else:
                emb_r_ = self.w_relation + 0.0001 * self.init_rel_LLM(self.emb_rel_LLM)
        else:
            emb_r_ = self.w_relation
        emb = torch.cat([self.ent_init_embeds,emb_r_], dim=0)
        if entity == 'object':
            h, r = self.get_raw_m_t(quadruple)
            T = self.get_init_time(quadruple)
            score_g = self.generate_mode(h, r, T, entity)
            score_c = self.copy_mode(h, r, T, copy_vocabulary, entity)
            if self.args.path:
                _, score_path, _ = self.reasoning_path(emb[:self.i_dim,:], 
                                                                    self.init_rel_LLM(self.emb_rel_LLM),
                                                                    quadruple, rel_path_vocabulary, cur_ts,
                                                                    'eval')


       
        a = args.alpha
        score = score_c * a + score_g * (1-a)
        if self.rel_path:
            score = score + self.gamma * score_path
        score = torch.log(score)
        return score

    def regularization_loss(self, reg_param):
        if self.LLM_init:
            r=self.init_rel_LLM(self.emb_rel_LLM)
            regularization_loss = torch.mean(r.pow(2)) + torch.mean(self.ent_init_embeds.pow(2)) + torch.mean(self.tim_init_embeds.pow(2))
        else:
            regularization_loss = torch.mean(self.w_relation.pow(2)) + torch.mean(self.ent_init_embeds.pow(2)) + torch.mean(self.tim_init_embeds.pow(2))
        return regularization_loss * reg_param
    
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
    
    def find_related(self, sub, rel, emb):
        x = emb[:self.num_e,:]
        r = emb[self.num_e:,:]
        assert x.shape[0] == self.num_e
        assert r.shape[0] == self.num_rel * 2
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, rel_emb, x
    
    def loss_comp(self, sub, rel, emb, obj=None, score_path=None):
        score = self.score_comp(sub, rel, emb)
        if self.rel_path:
            score = score + self.gamma * score_path
        return self.loss(score, obj)


    def score_comp(self, sub, rel, emb):
        sub_emb, rel_emb, all_emb = self.find_related(sub, rel, emb)
       
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

        return s


class Copy_mode(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda):
        super(Copy_mode, self).__init__()
        self.hidden_dim = hidden_dim

        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 3, output_dim)
        self.use_cuda = use_cuda

    def forward(self, ent_embed, rel_embed, time_embed, copy_vocabulary, entity):
        if entity == 'object':
            m_t = torch.cat((ent_embed, rel_embed, time_embed), dim=1)
        if entity == 'subject':
            m_t = torch.cat((rel_embed, ent_embed, time_embed), dim=1)

        q_s = self.tanh(self.W_s(m_t))
        if self.use_cuda:
            encoded_mask = torch.Tensor(np.array(copy_vocabulary.cpu() == 0, dtype=float) * (-100))
            encoded_mask = encoded_mask.to('cuda')
        else:
            encoded_mask = torch.Tensor(np.array(copy_vocabulary == 0, dtype=float) * (-100))

        score_c = q_s + encoded_mask

        return F.softmax(score_c, dim=1)
    
class Generate_mode(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(Generate_mode, self).__init__()
        self.W_mlp = nn.Linear(hidden_size * 3, output_dim)

    def forward(self, ent_embed, rel_embed, tim_embed, entity):
        if entity == 'object':
            m_t = torch.cat((ent_embed, rel_embed, tim_embed), dim=1)
        if entity == 'subject':
            m_t = torch.cat((rel_embed, ent_embed, tim_embed), dim=1)

        score_g = self.W_mlp(m_t)

        return F.softmax(score_g, dim=1)
    

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
        self.rel_prob = nn.Linear(args.embedding_dim, num_rel)
        self.rel2hist = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.rel2hist2 = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.num_ent = num_ent
        self.linear_align = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.linear_gru_hidden = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.linear_attn = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.device = torch.device("cuda")
        
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
        q_rel = mapped_rel[all_triples[:, 1]]

        history_rel_path = []
        gru_hidden = torch.zeros(partial_embeding.shape[0], self.args.embedding_dim)
        gru_hidden = gru_hidden.to(self.device)
        for ts in range(cur_ts):
            cur = partial_embeding[:, ts * num_rel: (ts + 1) * num_rel]
            attn = cur * torch.matmul(self.linear_attn(q_rel), mapped_rel.transpose(1,0))
            attn = F.softmax(torch.where(attn == 0, torch.tensor([-1000000000.0]).to(self.device), attn), dim=1)
            rel_path_emb = torch.matmul(attn, mapped_rel)
            gru_hidden = self.gru_cell(rel_path_emb, gru_hidden)
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]


        match_loss = self.match_loss_func(predicted_hist, gru_hidden)
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, all_triples, predicted_hist,
                                                   orig_score_func)
        return match_loss, path_loss, path_score

    def forward_forth(self, pre_emb, mapped_rel, all_triples, predicted_hist, orig_score_func=None):
        gru_hidden = self.gru_cell(mapped_rel[all_triples[:, 1]], predicted_hist) # complete reasoning path
        sub_emb, obj_emb = pre_emb[all_triples[:, 0]], pre_emb[all_triples[:, 2]]
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
        return path_loss, score

    def forward_eval(self, pre_emb, r_emb, all_triples, num_rel, partial_embeding=None, cur_ts=None, orig_score_func=None):
        mapped_rel_ = self.rel_mapping1(r_emb)
        mapped_rel = self.rel_mapping2(mapped_rel_)
        predicted_hist = 0.1 * self.rel2hist2(self.rel2hist(mapped_rel[all_triples[:, 1]])) + mapped_rel[all_triples[:, 1]]
        path_loss, path_score = self.forward_forth(pre_emb, mapped_rel, all_triples, predicted_hist, orig_score_func)
        return path_loss, path_score


