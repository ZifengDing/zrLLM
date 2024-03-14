import argparse
import numpy as np
import torch
import pickle
import time
import datetime
import os
import random
import utils
from cenet_model import CENET
import scipy.sparse as sp

def execute_valid(args, total_data, model,
                  data,
                  s_history, o_history,
                  s_label, o_label,
                  s_frequency, o_frequency, num_nodes, max_train_t):
    s_ranks2 = []
    o_ranks2 = []
    all_ranks2 = []

    s_ranks3 = []
    o_ranks3 = []
    all_ranks3 = []
    total_data = utils.to_device(torch.from_numpy(total_data))

    for batch_data in utils.make_batch(data,
                                       s_history,
                                       o_history,
                                       s_label,
                                       o_label,
                                       s_frequency,
                                       o_frequency,
                                       args.batch_size):
        if args.path:
            if args.dataset.lower() == 'icews21':
                cur_file = (max_train_t // 6)
                all_rel_path = sp.load_npz(
                    'data/{}/history/rel_path_all_{}.npz'.format(args.dataset, cur_file + 1))
                rel_path_idx = (max_train_t - (max_train_t // 6) * 6) * batch_data[0][:, 0] * num_nodes + batch_data[0][:,
                                                                                                      2]
                rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
                one_hot_rel_path = utils.to_device(rel_path.masked_fill(rel_path != 0, 1))
            else:
                all_rel_path = sp.load_npz(
                    'data/{}/history/rel_path_all.npz'.format(args.dataset))
                rel_path_idx = max_train_t * batch_data[0][:, 0] * num_nodes + batch_data[0][:, 2]
                rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
                one_hot_rel_path = utils.to_device(rel_path.masked_fill(rel_path != 0, 1))
        else:
            one_hot_rel_path = None

        batch_data[0] = utils.to_device(torch.from_numpy(batch_data[0]))
        batch_data[3] = utils.to_device(torch.from_numpy(batch_data[3])).float()
        batch_data[4] = utils.to_device(torch.from_numpy(batch_data[4])).float()
        batch_data[5] = utils.to_device(torch.from_numpy(batch_data[5])).float()
        batch_data[6] = utils.to_device(torch.from_numpy(batch_data[6])).float()

        with torch.no_grad():
            _, _, _, \
            sub_rank2, obj_rank2, cur_loss2, \
            sub_rank3, obj_rank3, cur_loss3, ce_all_acc = model(batch_data, 'Valid', total_data, rel_path_vocabulary=one_hot_rel_path)

            s_ranks2 += sub_rank2
            o_ranks2 += obj_rank2
            tmp2 = sub_rank2 + obj_rank2
            all_ranks2 += tmp2

            s_ranks3 += sub_rank3
            o_ranks3 += obj_rank3
            tmp3 = sub_rank3 + obj_rank3
            all_ranks3 += tmp3
    return s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3
