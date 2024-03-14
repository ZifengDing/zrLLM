import torch
import time
from utils import *
import scipy.sparse as sp

def push_data(*args, device=None):
    out_args = []
    for arg in args:
        arg = [_arg.to(device) for _arg in arg]
        out_args.append(arg)
    return out_args


def push_data2(*args, device=None):
    out_args = []
    for arg in args:
        arg = arg.to(device)
        out_args.append(arg)
    return out_args


def predict(loader, model, params, num_e, num_rel, test_adjmtx, logger, ts2emb, ts_max, history_time_nogt, triple_list):
    model.eval()
    p = params
    rank_group_num = 2000

    with torch.no_grad():
        results = {}
        iter = loader

        print("Start evaluation")
        t1 = time.time()

        for step, (
        sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, tar_ts, in_ts, edge_idlist, edge_typelist,
        indep_lab, adj_mtx, edge_jump_w, edge_jump_id, rel_jump) in enumerate(iter):
            if len(sub_tar) == 0:
                continue

            cur_ts = int(tar_ts[0].item() / params.scale * ts_max)
            # print(cur_ts)
            if tar_ts[0].item() / params.scale * ts_max - cur_ts > 0.9:
                cur_ts += 1

            # forward
            back_time, emb = model.forward_eval(in_ts, tar_ts, edge_idlist, edge_typelist, edge_jump_id, edge_jump_w, rel_jump)
            ts2emb = None
            rank_count = 0
            while rank_count < sub_tar[0].shape[0]:
                l, r = rank_count, (rank_count + rank_group_num) if (rank_count + rank_group_num) <= sub_tar[0].shape[
                    0] else sub_tar[0].shape[0]

                if params.path:
                    all_rel_path = sp.load_npz(
                        '{}/history/rel_path_{}.npz'.format(params.dataset, history_time_nogt))
                    rel_path_idx = sub_tar[0][l:r] * num_e + obj_tar[0][l:r]
                    rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
                    one_hot_rel_path = rel_path.masked_fill(rel_path != 0, 1)

                    one_hot_rel_path = one_hot_rel_path.to(params.device)
                    # print(one_hot_rel_path.shape)

                # push data onto gpu
                [sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_] = \
                    push_data2(sub_tar[0][l:r], rel_tar[0][l:r], obj_tar[0][l:r], lab_tar[0][l:r, :],
                               indep_lab[0][l:r, :], device=p.device)

                # compute scores for corresponding triples
                score = model.score_comp(sub_tar_, rel_tar_, emb, model.odeblock.odefunc)
                if params.path:
                    # print(torch.cat(
                    #     [sub_tar_.unsqueeze(1), rel_tar_.unsqueeze(1), obj_tar_.unsqueeze(1)], dim=1).shape, emb.shape)
                    _, score_path, _ = model.reasoning_path(emb[:num_e,:], model.init_rel_LLM(model.emb_rel_LLM), torch.cat(
                        [sub_tar_.unsqueeze(1), rel_tar_.unsqueeze(1), obj_tar_.unsqueeze(1)], dim=1),
                                                            one_hot_rel_path, history_time_nogt, 'eval')
                    score = score + model.gamma * score_path
                b_range = torch.arange(score.shape[0], device=p.device)

                # raw ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]

                ranks = ranks.float()
                results['count_raw'] = torch.numel(ranks) + results.get('count_raw', 0.0)
                results['mar_raw'] = torch.sum(ranks).item() + results.get('mar_raw', 0.0)
                results['mrr_raw'] = torch.sum(1.0 / ranks).item() + results.get('mrr_raw', 0.0)
                for k in range(10):
                    results['hits@{}_raw'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}_raw'.format(k + 1), 0.0)

                # time aware filtering
                target_score = score[b_range, obj_tar_]
                score = torch.where(lab_tar_.byte(), -torch.ones_like(score) * 10000000, score)
                score[b_range, obj_tar_] = target_score

                # time aware filtered ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mar'] = torch.sum(ranks).item() + results.get('mar', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                # time unaware filtering
                score = torch.where(indep_lab_.byte(), -torch.ones_like(score) * 10000000, score)
                score[b_range, obj_tar_] = target_score

                # time unaware filtered ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]
                ranks = ranks.float()
                results['count_ind'] = torch.numel(ranks) + results.get('count_ind', 0.0)
                results['mar_ind'] = torch.sum(ranks).item() + results.get('mar_ind', 0.0)
                results['mrr_ind'] = torch.sum(1.0 / ranks).item() + results.get('mrr_ind', 0.0)
                for k in range(10):
                    results['hits@{}_ind'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}_ind'.format(k + 1), 0.0)

                rank_count += rank_group_num
            del sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_

        results['mar'] = round(results['mar'] / results['count'], 5)
        results['mrr'] = round(results['mrr'] / results['count'], 5)
        results['mar_raw'] = round(results['mar_raw'] / results['count_raw'], 5)
        results['mrr_raw'] = round(results['mrr_raw'] / results['count_raw'], 5)
        results['mar_ind'] = round(results['mar_ind'] / results['count_ind'], 5)
        results['mrr_ind'] = round(results['mrr_ind'] / results['count_ind'], 5)
        for k in range(10):
            results['hits@{}'.format(k + 1)] = round(results['hits@{}'.format(k + 1)] / results['count'], 5)
            results['hits@{}_raw'.format(k + 1)] = round(results['hits@{}_raw'.format(k + 1)] / results['count_raw'], 5)
            results['hits@{}_ind'.format(k + 1)] = round(results['hits@{}_ind'.format(k + 1)] / results['count_ind'], 5)

        t2 = time.time()
        print("evaluation time: ", t2 - t1)
        logger.info("evaluation time: {}".format(t2 - t1))

    return results, ts2emb