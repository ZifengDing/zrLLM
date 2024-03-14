"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""
import numpy as np
import torch
import dgl
from tqdm import tqdm
import rgcn.knowledge_graph as knwlgrh
from collections import defaultdict
from scipy import sparse


#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def sort_and_rank(score, target):
    '''
    :param score: (batch_size, num_ents)
    :param target: (batch_size, 1)
    '''
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


#TODO filer by groud truth in the same time snapshot not all ground truth
def sort_and_rank_time_filter(batch_a, batch_r, score, target, total_triplets):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    for i in range(len(batch_a)):
        ground = indices[i]
    indices = indices[:, 1].view(-1)
    return indices


def sort_and_rank_filter(batch_a, batch_r, score, target, all_ans):
    for i in range(len(batch_a)):
        ans = target[i]
        b_multi = list(all_ans[batch_a[i].item()][batch_r[i].item()])
        ground = score[i][ans]
        score[i][b_multi] = 0
        score[i][ans] = ground
    _, indices = torch.sort(score, dim=1, descending=True)  # indices : [B, number entity]
    indices = torch.nonzero(indices == target.view(-1, 1))  # indices : [B, 2]
    indices = indices[:, 1].view(-1)
    return indices


def filter_score(test_triples, score, all_ans):
    '''
    :param test_triples: (batch_size, 3)
    :param score: (batch_size, num_ents)
    :param all_ans: {e1: {rel: set(e2)}, e2: {rel+num_rels: set(e1)}, ...}
    '''
    if all_ans is None:
        return score
    test_triples = test_triples.cpu() # (batch_size, 3)
    for _, triple in enumerate(test_triples):
        h, r, t, nouse = triple
        ans = list(all_ans[h.item()][r.item()])
        ans.remove(t.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000
    return score

def filter_score_r(test_triples, score, all_ans):
    if all_ans is None:
        return score
    test_triples = test_triples.cpu()
    for _, triple in enumerate(test_triples):
        h, r, t, nouse = triple
        ans = list(all_ans[h.item()][t.item()])
        # print(h, r, t)
        # print(ans)
        ans.remove(r.item())
        ans = torch.LongTensor(ans)
        score[_][ans] = -10000000  #
    return score

def get_total_rank(test_triples, score, all_ans, eval_bz, rel_predict):
    '''
    :param test_triples: (num_triples_time*2, 3) num_triples_time*2
    :param score: (num_triples_time*2, num_ents)
    :param all_ans: {e1: {rel: set(e2)}, e2: {rel+num_rels: set(e1)}, ...}
    :param eval_bz: 1000
    :param rel_predict:
    :return:
    '''
    num_triples = len(test_triples)
    n_batch = (num_triples + eval_bz - 1) // eval_bz
    rank = []
    filter_rank = []
    for idx in range(n_batch):
        batch_start = idx * eval_bz
        batch_end = min(num_triples, (idx + 1) * eval_bz)
        triples_batch = test_triples[batch_start:batch_end, :] # (tensor)(batch_size, 3)
        score_batch = score[batch_start:batch_end, :] # (batch_size, num_ents)
        if rel_predict == 1:
            target = test_triples[batch_start:batch_end, 1]
        elif rel_predict == 0:
            target = test_triples[batch_start:batch_end, 2] # ground truth (batch_size, 1)
        rank.append(sort_and_rank(score_batch, target)) # (batch_size)

        if rel_predict:
            filter_score_batch = filter_score_r(triples_batch, score_batch, all_ans)
        else:
            filter_score_batch = filter_score(triples_batch, score_batch, all_ans)
        filter_rank.append(sort_and_rank(filter_score_batch, target))

    rank = torch.cat(rank)
    filter_rank = torch.cat(filter_rank)
    rank += 1 # change to 1-indexed
    filter_rank += 1
    mrr = torch.mean(1.0 / rank.float())
    filter_mrr = torch.mean(1.0 / filter_rank.float())
    return filter_mrr.item(), mrr.item(), rank, filter_rank


def stat_ranks(rank_list, method, zero_log):
    hits = [1, 3, 10]
    total_rank = torch.cat(rank_list)

    mrr = torch.mean(1.0 / total_rank.float())
    print("MRR ({}): {:.6f}".format(method, mrr.item()))
    zero_log.info("MRR " + str(method) + ': ' + str(mrr.item()))
    for hit in hits:
        avg_count = torch.mean((total_rank <= hit).float())
        print("Hits ({}) @ {}: {:.6f}".format(method, hit, avg_count.item()))
        zero_log.info("Hits @ " + str(hit) + ' ' + str(method) + " : " + str(avg_count.item()))
    return mrr


def flatten(l):
    flatten_l = []
    for c in l:
        if type(c) is list or type(c) is tuple:
            flatten_l.extend(flatten(c))
        else:
            flatten_l.append(c)
    return flatten_l

def UnionFindSet(m, edges):
    """

    :param m:
    :param edges:
    :return: union number in a graph
    """
    roots = [i for i in range(m)]
    rank = [0 for i in range(m)]
    count = m

    def find(member):
        tmp = []
        while member != roots[member]:
            tmp.append(member)
            member = roots[member]
        for root in tmp:
            roots[root] = member
        return member

    for i in range(m):
        roots[i] = i
    # print ufs.roots
    for edge in edges:
        print(edge)
        start, end = edge[0], edge[1]
        parentP = find(start)
        parentQ = find(end)
        if parentP != parentQ:
            if rank[parentP] > rank[parentQ]:
                roots[parentQ] = parentP
            elif rank[parentP] < rank[parentQ]:
                roots[parentP] = parentQ
            else:
                roots[parentQ] = parentP
                rank[parentP] -= 1
            count -= 1
    return count

def append_object(e1, e2, r, d):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(e1, e2, r, d, num_rel):
    if not e2 in d:
        d[e2] = {}
    if not r+num_rel in d[e2]: # r: sub->obj; r+num+rel: obj->sub
        d[e2][r+num_rel] = set()
    d[e2][r+num_rel].add(e1)


def add_object(e1, e2, r, d, num_rel):
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def load_all_answers(total_data, num_rel):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[: 3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {} # {e1: {e2: set()}，...}
    for line in total_data:
        s, r, o = line[: 3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans # {e1: {rel: set(e2)}, e2: {rel+num_rels: set(e1)}, ...}


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)
    return all_ans_list

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)): # np.array([[s, r, o, time], []...])
        t = data[i][3]
        train = data[i]

        if latest_t != t:
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:])
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list

def add_inverse_rel(data, num_rel): # [[s, r, o], [], ...]
    inverse_triples = np.array([[o, r+num_rel, s] for s, r, o in data])
    triples_w_inverse = np.concatenate((data, inverse_triples))
    return triples_w_inverse

def slide_list(snapshots, k=1):
    """
    :param k: padding K history for sequence stat
    :param snapshots: all snapshot
    :return:
    """
    k = k
    if k > len(snapshots):
        print("ERROR: history length exceed the length of snapshot: {}>{}".format(k, len(snapshots)))
    for _ in tqdm(range(len(snapshots)-k+1)):
        yield snapshots[_: _+k]



def load_data(dataset, bfs_level=3, relabel=False):
    if dataset in ['aifb', 'mutag', 'bgs', 'am']:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ['FB15k', 'wn18', 'FB15k-237']:
        return knwlgrh.load_link(dataset)
    elif dataset in ['ICEWS18', 'ICEWS14', "GDELT", "SMALL", "ICEWS14s", "ICEWS05-15","YAGO",
                     "WIKI", "ACLED", "ICEWS21", "ICEWS22"]:
        return knwlgrh.load_from_local("../data", dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))

def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0].item(), r.item(), index.item(), test_triples[_][3].item()])
            else:
                predict_triples.append([index.item(), r.item()-num_rels, test_triples[_][0].item(), test_triples[_][3].item()])


    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    # for _ in range(len(test_triples)):
    #     h, r = test_triples[_][0], test_triples[_][1]
    #     if (sorted_score[_][0]-sorted_score[_][1])/sorted_score[_][0] > 0.3:
    #         if r < num_rels:
    #             predict_triples.append([h, r, indices[_][0]])

    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                #predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index-num_rels, h])
                #predict_triples.append([t, index-num_rels, h])

    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def dilate_input(input_list, dilate_len):
    dilate_temp = []
    dilate_input_list = []
    for i in range(len(input_list)):
        if i % dilate_len == 0 and i:
            if len(dilate_temp):
                dilate_input_list.append(dilate_temp)
                dilate_temp = []
        if len(dilate_temp):
            dilate_temp = np.concatenate((dilate_temp, input_list[i]))
        else:
            dilate_temp = input_list[i]
    dilate_input_list.append(dilate_temp)
    dilate_input_list = [np.unique(_, axis=0) for _ in dilate_input_list]
    return dilate_input_list

def emb_norm(emb, epo=0.00001):
    x_norm = torch.sqrt(torch.sum(emb.pow(2), dim=1))+epo
    emb = emb/x_norm.view(-1,1)
    return emb

def shuffle(data, labels):
    shuffle_idx = np.arange(len(data))
    np.random.shuffle(shuffle_idx)
    relabel_output = data[shuffle_idx]
    labels = labels[shuffle_idx]
    return relabel_output, labels


def cuda(tensor):
    if tensor.device == torch.device('cpu'):
        return tensor.cuda()
    else:
        return tensor


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t)
    return a

def r2e(triplets, num_rels): # triplets(array): [[s, r, o], [s, r, o], ...]
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel+num_rels].add(src)
        r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

def build_sub_graph(num_nodes, num_rels, triples, use_cuda, segnn, gpu):
    '''
    :param num_nodes:
    :param num_rels:
    :param triples:
    :param use_cuda:
    :param segnn:
    :param gpu:
    :return:
    '''
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    triples = triples[:, :3]
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    if segnn:
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g.edata['rel_id'] = torch.LongTensor(rel)
    else:
        g = dgl.DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(src, dst)
        norm = comp_deg_norm(g)
        node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1) # [0, num_nodes)
        g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
        g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g

def r2e_super(triplets, num_rels): # triplets(array): [[s, r, o], [s, r, o], ...]
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_super_r = np.unique(rel)
    # uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        # r_to_e[rel+num_rels].add(src)
        # r_to_e[rel+num_rels].add(dst)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_super_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_super_r, r_len, e_idx

def get_relhead_reltal(tris, num_nodes, num_rels):
    '''
    :param tris:
    :param num_nodes:
    :param num_rels:
    :return:
    '''
    tris = tris[:, :-1]
    inverse_triplets = tris[:, [2, 1, 0]]
    inverse_triplets[:, 1] = inverse_triplets[:, 1] + num_rels
    all_tris = np.concatenate((tris, inverse_triplets), axis=0)

    rel_head = torch.zeros((num_rels*2, num_nodes), dtype=torch.int)
    rel_tail = torch.zeros((num_rels*2, num_nodes), dtype=torch.int)

    for tri in all_tris:
        h, r, t = tri

        rel_head[r, h] += 1
        rel_tail[r, t] += 1

    return rel_head, rel_tail

def build_super_g(num_rels, rel_head, rel_tail, use_cuda, segnn, gpu):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    tail_head = torch.matmul(rel_tail, rel_head.T) # (num_rels*2, num_rels*2)
    head_tail = torch.matmul(rel_head, rel_tail.T) # (num_rels*2, num_rels*2)

    tail_tail = torch.matmul(rel_tail, rel_tail.T) - torch.diag(torch.sum(rel_tail, axis=1)) # (num_rels*2, num_rels*2)

    head_head = torch.matmul(rel_head, rel_head.T) - torch.diag(torch.sum(rel_head, axis=1)) # (num_rels*2, num_rels*2)

    # construct super relation graph from adjacency matrix
    src = torch.LongTensor([])
    dst = torch.LongTensor([])
    p_rel = torch.LongTensor([])
    for p_rel_idx, mat in enumerate([tail_head, head_tail, tail_tail, head_head]): # p_rel_idx: 0, 1, 2, 3
        sp_mat = sparse.coo_matrix(mat)
        src = torch.cat([src, torch.from_numpy(sp_mat.row)])
        dst = torch.cat([dst, torch.from_numpy(sp_mat.col)])
        p_rel = torch.cat([p_rel, torch.LongTensor([p_rel_idx] * len(sp_mat.data))])

    src_tris = src.unsqueeze(1)
    dst_tris = src.unsqueeze(1)
    p_rel_tris = p_rel.unsqueeze(1)
    super_triples = torch.cat((src_tris, p_rel_tris, dst_tris), dim=1).numpy()
    src = src.numpy()
    dst = dst.numpy()
    p_rel = p_rel.numpy()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    p_rel = np.concatenate((p_rel, p_rel + 4))

    if segnn:
        super_g = dgl.graph((src, dst), num_nodes=num_rels*2)
        super_g.edata['rel_id'] = torch.LongTensor(p_rel)
    else:
        super_g = dgl.DGLGraph()
        super_g.add_nodes(num_rels*2)
        super_g.add_edges(src, dst)
        norm = comp_deg_norm(super_g)
        rel_node_id = torch.arange(0, num_rels*2, dtype=torch.long).view(-1, 1) # [0, num_rels*2)
        super_g.ndata.update({'id': rel_node_id, 'norm': norm.view(-1, 1)})
        super_g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        super_g.edata['type'] = torch.LongTensor(p_rel)

    uniq_super_r, r_len, r_to_e = r2e_super(super_triples, num_rels)
    super_g.uniq_super_r = uniq_super_r
    super_g.r_to_e = r_to_e
    super_g.r_len = r_len

    if use_cuda:
        super_g.to(gpu)
        super_g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return super_g