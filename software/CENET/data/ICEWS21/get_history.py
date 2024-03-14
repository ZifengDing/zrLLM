import os
import torch
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from tqdm import tqdm
import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ICEWS21')
args = args.parse_args()
print(args)

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def load_all_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    quadrupleList = []
    times = set()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName2), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName3), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    times = list(times)
    times.sort()
    ts = times

    return np.asarray(quadrupleList), np.asarray(times), ts

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

all_data, all_times, ts = load_all_quadruples('', 'train.txt', 'valid.txt', "test.txt")
# all_data, all_times = load_all_quadruples('../data/{}'.format(args.dataset), 'train.txt', "test.txt")
train_data, train_times = load_quadruples('', 'train.txt')
num_e, num_r = get_total_number('','stat.txt')

save_dir_obj = 'history/'

def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

mkdirs(save_dir_obj)
raw_num_r = num_r
num_r = num_r * 2
num_ts = len(all_times)
# print(num_ts)
# assert 0

# train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if (quad[3] < tim)])
group_num = 6
k = 0
while k * group_num < len(all_times):
    rel_dim1 = []
    rel_dim2 = []
    if (k+1) * group_num > len(all_times):
        end = -1
        size_cur = len(all_times) - (k * group_num)
    else:
        end = (k+1) * group_num
        size_cur = group_num
    for i, tim in enumerate(tqdm(all_times[(group_num * k):end])):
        ts_cur = ts[(group_num * k):end]
        ts_cur = [item -  ts[(group_num * k)] for item in ts_cur]
        # print(ts[i])
        train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if (quad[3] < tim)])
        if tim != all_times[0]:
            rel_dim1.extend(ts_cur[i] * train_new_data[:, 0] * num_e + train_new_data[:, 2])
            rel_dim2.extend(train_new_data[:, 3] * num_r + train_new_data[:, 1])
    rel_path_d = np.ones(len(rel_dim1))
    rel_path = sp.csr_matrix((rel_path_d, (rel_dim1, rel_dim2)), shape=(size_cur * num_e * num_e, num_ts * num_r), dtype=np.int8)
    sp.save_npz('history/rel_path_all_{}.npz'.format(k+1), rel_path)
    print("saved group {}".format(k+1))
    k += 1
assert 0
rel_dim1 = []
rel_dim2 = []
for i, tim in enumerate(tqdm(all_times[int(len(all_times)/2):])):
    # print(ts[i])
    train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if (quad[3] < tim)])
    if tim != all_times[0]:
        rel_dim1.extend(ts[i + int(len(all_times)/2)] * train_new_data[:, 0] * num_e + train_new_data[:, 2])
        rel_dim2.extend(train_new_data[:, 3] * num_r + train_new_data[:, 1])
rel_path_d = np.ones(len(rel_dim1))
rel_path = sp.csr_matrix((rel_path_d, (rel_dim1, rel_dim2)), shape=((num_ts-int(len(all_times)/2)) * num_e * num_e, num_ts * num_r))
sp.save_npz('history/rel_path_all_2.npz', rel_path)
# for tim in tqdm(all_times):
#     train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if (quad[3] < tim)])
#     if tim != all_times[0]:
#         train_new_data = torch.from_numpy(train_new_data)
#         inverse_train_data = train_new_data[:, [2, 1, 0, 3]] # object, relation, subject, timestamp
#         inverse_train_data[:, 1] = inverse_train_data[:, 1] + raw_num_r # create inverse relation
#         train_new_data = torch.cat([train_new_data, inverse_train_data])
#         train_new_data_ = train_new_data
#         # entity history
#         train_new_data = torch.unique(train_new_data[:, :3], sorted=False, dim=0)
#         train_new_data = train_new_data.numpy()
#         row = train_new_data[:, 0] * num_r + train_new_data[:, 1]
#         col = train_new_data[:, 2]
#         # print(train_new_data[:, 0])
#         # print(num_r, num_e)
#         # print(row, col)
#         # print(num_e * num_r)
#         # print(np.array([a * num_r for a in train_new_data[:, 0]]))
#         d = np.ones(len(row))
#         tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r, num_e))
#
#         # relation history
#         rel_row = train_new_data[:, 0] * num_e + train_new_data[:, 2]
#         rel_col = train_new_data[:, 1]
#         rel_d = np.ones(len(rel_row))
#         rel_seq = sp.csr_matrix((rel_d, (rel_row, rel_col)), shape=(num_e * num_e, num_r))
#
#         # relation path
#         rel_dim1 = train_new_data_[:, 0] * num_e + train_new_data_[:, 2]
#         rel_dim2 = train_new_data_[:, 3] * num_r + train_new_data_[:, 1]
#         # rel_dim3 = train_new_data[:, 1]
#         rel_path_d = np.ones(len(rel_dim1))
#         rel_path = sp.csr_matrix((rel_path_d, (rel_dim1, rel_dim2)), shape=(num_e * num_e, num_ts * num_r))
#     else:
#         tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_r, num_e))
#         rel_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_e, num_r))
#         rel_path = sp.csr_matrix(([], ([], [])), shape=(num_e * num_e, num_ts * num_r))
#     sp.save_npz('../data/{}/history/tail_history_{}.npz'.format(args.dataset, tim), tail_seq)
#     sp.save_npz('../data/{}/history/rel_history_{}.npz'.format(args.dataset, tim), rel_seq)
#     sp.save_npz('../data/{}/history/rel_path_{}.npz'.format(args.dataset, tim), rel_path)








