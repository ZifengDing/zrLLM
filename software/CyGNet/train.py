#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
from utils import *
import torch.nn.functional as F
from torch import nn
from torch import optim
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from config import args
from link_prediction import link_prediction
from evaluation import *
from tensorboardX import SummaryWriter



if args.LLM_init and args.path:
	output_file = "results/_{}_LLM_with_RHL.txt".format(args.dataset)
elif args.LLM_init:
	output_file = "results/_{}_LLM_without_RHL.txt".format(args.dataset)
else:
	output_file = "results/_{}_baseline.txt".format(args.dataset)

torch.set_num_threads(2)

import warnings
warnings.filterwarnings(action='ignore')


def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)


use_cuda = args.gpu >= 0 and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

if args.dataset == 'ICEWS14':
	train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
	test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
	dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
else:
	train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
	test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
	dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'valid.txt')

all_times = np.concatenate([train_times, dev_times, test_times])

num_e, num_r = get_total_number('./data/{}'.format(args.dataset), 'stat.txt')
num_times = int(max(all_times) / args.time_stamp) + 1
print('num_times', num_times)

train_samples = []
for tim in train_times:
	print(str(tim)+'\t'+str(max(train_times)))
	data = get_data_with_t(train_data, tim)
	train_samples.append(len(data))

model = link_prediction(num_e, args.hidden_dim, num_r, num_times, use_cuda)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

if args.entity == 'object':
	mkdirs('./results/bestmodel/{}'.format(args.dataset))
	model_state_file = './results/bestmodel/{}/model_state.pth'.format(args.dataset)
if args.entity == 'subject':
	mkdirs('./results/bestmodel/{}_sub'.format(args.dataset))
	model_state_file = './results/bestmodel/{}_sub/model_state.pth'.format(args.dataset)

forward_time = []
backward_time = []

best_mrr = 0
best_hits1 = 0
best_hits3 = 0
best_hits10 = 0

batch_size = args.batch_size
mkdirs('./scalar/{}/'.format(args.dataset))
writer = SummaryWriter(log_dir='scalar/{}/'.format(args.dataset))

print('start train')
n = 0
with open(output_file, "w") as file:
	file.write("My arguments: {}\n".format(args))
	file.write('num_times ' + str(num_times)+"\n")
	for i in range(args.n_epochs):
		time_start_epoch = time.time()
		epoch_num=i
		train_loss = 0
		sample_read = 0
		sample_end = 0
		all_tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * 2*num_r, num_e))
		for train_samples_tim in range(len(train_samples)):
			model.train()
			sample_end += train_samples[train_samples_tim]

			if train_samples_tim > 0:
				train_tim = train_times[train_samples_tim-1]
				if args.entity == 'object':
					tim_tail_seq = sp.load_npz('./data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[train_samples_tim - 1]))
				if args.entity == 'subject':
					tim_tail_seq = sp.load_npz('./data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[train_samples_tim - 1]))
				all_tail_seq = all_tail_seq + tim_tail_seq

			train_sample_data = train_data[sample_read:sample_end, :]

			n_batch = (train_sample_data.shape[0] + batch_size - 1) // batch_size
			for idx in range(n_batch):
				batch_start = idx * batch_size
				batch_end = min(train_sample_data.shape[0], (idx + 1) * batch_size)
				train_batch_data_ = train_sample_data[batch_start: batch_end]
				train_batch_data_ = torch.from_numpy(train_batch_data_).long().cuda()
				train_batch_data_inverse =train_batch_data_[:, [2, 1, 0, 3]]
				train_batch_data_inverse[:, 1] = train_batch_data_inverse[:, 1] + num_r # create inverse relation
				train_batch_data = torch.cat([train_batch_data_, train_batch_data_inverse])	

				train_batch_data = train_batch_data.cpu().numpy()
				
				if args.path:
					all_rel_path = sp.load_npz(
						'{}/history/rel_path_{}.npz'.format(args.dataset,  train_times[train_samples_tim]))
					rel_path_idx = train_batch_data[:,0] * num_e + train_batch_data[:,2]
					rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
					one_hot_rel_path = rel_path.masked_fill(rel_path != 0, 1)

					one_hot_rel_path = one_hot_rel_path.to(device)
				else:
					one_hot_rel_path = None
				if args.entity == 'object':
					labels = torch.LongTensor(train_batch_data[:, 2])
					seq_idx = train_batch_data[:, 0] * 2 * num_r + train_batch_data[:, 1]
				if args.entity == 'subject':
					labels = torch.LongTensor(train_batch_data[:, 0])
					seq_idx = train_batch_data[:, 2] * 2 * num_r + train_batch_data[:, 1]

				tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
				one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

				if use_cuda:
					labels, one_hot_tail_seq = labels.to(device), one_hot_tail_seq.to(device)

				t0 = time.time()
				score = model(train_batch_data, one_hot_tail_seq,one_hot_rel_path, train_times[train_samples_tim], entity=args.entity)
				if args.withRegul:
					loss = F.nll_loss(score, labels) + model.regularization_loss(reg_param=0.01)
				else:
					loss = F.nll_loss(score, labels)
				if args.path:
					loss = loss + model.match_loss + model.path_loss
				train_loss += loss.item()

				t1 = time.time()
				loss.backward()
				optimizer.step()
				t2 = time.time()

				writer.add_scalar('{}_loss'.format(args.dataset), loss.item(), n)
				n += 1

				forward_time.append(t1 - t0)
				backward_time.append(t2 - t1)
				optimizer.zero_grad()

			sample_read += train_samples[train_samples_tim]

		if i>=args.valid_epoch:
			mrr, hits1, hits3, hits10 = 0,0,0,0

			dev_sample_read = 0
			dev_sample_end = 0
			if args.entity == 'object':
				tim_tail_seq = sp.load_npz('./data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[-1]))
			if args.entity == 'subject':
				tim_tail_seq = sp.load_npz('./data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[-1]))
			all_tail_seq = all_tail_seq + tim_tail_seq

			n_batch = (dev_data.shape[0] + batch_size - 1) // batch_size

			for idx in range(n_batch):
				batch_start = idx * batch_size
				batch_end = min(dev_data.shape[0], (idx + 1) * batch_size)
				dev_batch_data = dev_data[batch_start: batch_end]
				dev_batch_data = torch.from_numpy(dev_batch_data).long().cuda()
				dev_batch_data_inverse =dev_batch_data[:, [2, 1, 0, 3]]
				dev_batch_data_inverse[:, 1] = dev_batch_data_inverse[:, 1] + num_r # create inverse relation
				dev_batch_data = torch.cat([dev_batch_data, dev_batch_data_inverse])	

				dev_batch_data = dev_batch_data.cpu().numpy()
				if args.path:
					all_rel_path = sp.load_npz(
						'{}/history/rel_path_{}.npz'.format(args.dataset,  train_times[-1]+1))
					rel_path_idx = dev_batch_data[:,0] * num_e + dev_batch_data[:,2]
					rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
					one_hot_rel_path = rel_path.masked_fill(rel_path != 0, 1)

					one_hot_rel_path = one_hot_rel_path.to(device)
				else:
					one_hot_rel_path = None
				if args.entity == 'object':
					dev_label = torch.LongTensor(dev_batch_data[:, 2])
					seq_idx = dev_batch_data[:, 0] * 2 * num_r + dev_batch_data[:, 1]
				if args.entity == 'subject':
					dev_label = torch.LongTensor(dev_batch_data[:, 0])
					seq_idx = dev_batch_data[:, 2] * 2 *  num_r + dev_batch_data[:, 1]

				tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
				one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)

				if use_cuda:
					dev_label, one_hot_tail_seq = dev_label.to(device), one_hot_tail_seq.to(device)
				dev_score = model.forward_eval(dev_batch_data, one_hot_tail_seq, one_hot_rel_path, train_times[-1]+1, entity=args.entity)
				
				if args.raw:
					tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(dev_score, dev_label, hits=[1, 3, 10])
				else:
					tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_filtered_mrr_timeaware(num_e, 
													dev_score, 
													torch.LongTensor(train_data),
														torch.LongTensor(dev_data),
													torch.LongTensor(dev_batch_data),
													entity=args.entity,
													hits=[1, 3, 10])

				mrr += tim_mrr * len(dev_batch_data)
				hits1 += tim_hits1 * len(dev_batch_data)
				hits3 += tim_hits3 * len(dev_batch_data)
				hits10 += tim_hits10 * len(dev_batch_data)

			mrr = mrr / (2*dev_data.shape[0])
			hits1 = hits1 /(2* dev_data.shape[0])
			hits3 = hits3 / (2*dev_data.shape[0])
			hits10 = hits10 / (2*dev_data.shape[0])
			print("MRR : {:.6f}".format(mrr))
			print("Hits @ 1: {:.6f}".format(hits1))
			print("Hits @ 3: {:.6f}".format(hits3))
			print("Hits @ 10: {:.6f}".format(hits10))
			file.write("MRR : {:.6f}\n".format(mrr))
			file.write("Hits @ 1: {:.6f}\n".format(hits1))
			file.write("Hits @ 3: {:.6f}\n".format(hits3))
			file.write("Hits @ 10: {:.6f}\n".format(hits10))

			

			if mrr > best_mrr:
				best_mrr = mrr
				torch.save({'state_dict': model.state_dict(), 'epoch': i+1},
						model_state_file)
				count = 0
			else:
				count += 1
			
			if hits1 > best_hits1:
				best_hits1 = hits1
			if hits3 > best_hits3:
				best_hits3 = hits3
			if hits10 > best_hits10:
				best_hits10 = hits10
			
			print("Seen Object: Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
				format(i+1, train_loss, best_mrr, best_hits1, best_hits3, best_hits10, forward_time[-1], backward_time[-1]))
			
			file.write("Seen Object: Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f} | Forward {:.4f}s | Backward {:.4f}s\n".
				format(i+1, train_loss, best_mrr, best_hits1, best_hits3, best_hits10, forward_time[-1], backward_time[-1]))
			
			all_tail_seq_obj = sp.csr_matrix(([], ([], [])), shape=(num_e * 2 * num_r, num_e))
			# all_tail_seq_sub = sp.csr_matrix(([], ([], [])), shape=(num_e * 2 * num_r, num_e))
			for i in range(len(train_times)):
				tim_tail_seq_obj = sp.load_npz(
					'./data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[i]))
				# tim_tail_seq_sub = sp.load_npz(
				# 	'./data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[i]))
				all_tail_seq_obj = all_tail_seq_obj + tim_tail_seq_obj

			obj_test_mrr, obj_test_hits1, obj_test_hits3, obj_test_hits10 = 0, 0, 0, 0
			n_batch = (test_data.shape[0] + batch_size - 1) // batch_size

			for idx in range(n_batch):
				batch_start = idx * batch_size
				batch_end = min(test_data.shape[0], (idx + 1) * batch_size)
				test_batch_data = test_data[batch_start: batch_end]
				test_batch_data = torch.from_numpy(test_batch_data).long().cuda()
				test_batch_data_inverse =test_batch_data[:, [2, 1, 0, 3]]
				test_batch_data_inverse[:, 1] = test_batch_data_inverse[:, 1] + num_r # create inverse relation
				test_batch_data = torch.cat([test_batch_data, test_batch_data_inverse])	
				test_batch_data = test_batch_data.cpu().numpy()
				if args.path:
					all_rel_path = sp.load_npz(
						'{}/history/rel_path_{}.npz'.format(args.dataset,  train_times[-1]+1))
					rel_path_idx = test_batch_data[:,0] * num_e + test_batch_data[:,2]
					rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
					one_hot_rel_path = rel_path.masked_fill(rel_path != 0, 1)

					one_hot_rel_path = one_hot_rel_path.to(device)
				else:
					one_hot_rel_path = None

				# test_label = torch.LongTensor(test_batch_data[:, 0])
				test_label = torch.LongTensor(test_batch_data[:, 2])
				# seq_idx = test_batch_data[:, 2] * 2* num_r + test_batch_data[:, 1]
				seq_idx = test_batch_data[:, 0] * 2* num_r + test_batch_data[:, 1]

				tail_seq = torch.Tensor(all_tail_seq_obj[seq_idx].todense())
				one_hot_tail_seq_obj = tail_seq.masked_fill(tail_seq != 0, 1)


				if use_cuda:
					test_label, one_hot_tail_seq_obj = test_label.to(device), one_hot_tail_seq_obj.to(device)
				test_score = model.forward_eval(test_batch_data, one_hot_tail_seq_obj, one_hot_rel_path,  train_times[-1]+1, entity='object')

				if args.raw:
					tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(test_score, test_label, hits=[1, 3, 10])
				elif args.time_aware_filter:
					tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_filtered_test_mrr_timeaware(num_e, test_score,
																						torch.LongTensor(
																						train_data),
																					torch.LongTensor(
																						dev_data),
																					torch.LongTensor(
																						test_data),
																					torch.LongTensor(
																						test_batch_data),                                                                          
																					entity='object',
																					hits=[1, 3, 10])     
				else:
					tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_filtered_test_mrr(num_e, test_score,
																					torch.LongTensor(
																						train_data),
																					torch.LongTensor(
																						dev_data),
																					torch.LongTensor(
																						test_data),
																					torch.LongTensor(
																						test_batch_data),
																					entity='object',
																					hits=[1, 3, 10])

				obj_test_mrr += tim_mrr * len(test_batch_data)
				obj_test_hits1 += tim_hits1 * len(test_batch_data)
				obj_test_hits3 += tim_hits3 * len(test_batch_data)
				obj_test_hits10 += tim_hits10 * len(test_batch_data)

			obj_test_mrr = obj_test_mrr / (2*test_data.shape[0])
			obj_test_hits1 = obj_test_hits1 / (2*test_data.shape[0])
			obj_test_hits3 = obj_test_hits3 / (2*test_data.shape[0])
			obj_test_hits10 = obj_test_hits10 / (2*test_data.shape[0])

			print("unseen object-- Epoch {:04d} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
				format(epoch_num+1, obj_test_mrr, obj_test_hits1, obj_test_hits3, obj_test_hits10))

			file.write("unseen object-- Epoch {:04d} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}\n".
				format(epoch_num+1, obj_test_mrr, obj_test_hits1, obj_test_hits3, obj_test_hits10))
			if count == args.counts:
				break
			
	writer.close()
	print("training done")
	print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
	print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))