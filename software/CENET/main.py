import argparse
import datetime
import os
import pickle
import time

import numpy as np
import torch
import scipy.sparse as sp

import utils
import test
import valid
from cenet_model import CENET
import copy


# GPU_ID = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
# use_cuda = torch.cuda.is_available()
# if use_cuda:
#     print('***************GPU_ID***************: ', GPU_ID)
# else:
#     raise NotImplementedError
#
#
# seed = 999
# np.random.seed(seed)
# torch.manual_seed(seed)


def main_portal(args):
    settings = {}

    num_nodes, num_rels, num_t = utils.get_total_number('./data/' + args.dataset, 'stat.txt')
    train_data, train_times = utils.load_quadruples('./data/' + args.dataset, 'train.txt')
    max_train_t = train_times[-1]
    try:
        dev_data, _ = utils.load_quadruples('./data/' + args.dataset, 'valid.txt')
    except:
        print(args.dataset, 'does not have valid set.')
    test_data, _ = utils.load_quadruples('./data/' + args.dataset, 'test.txt')
    try:
        total_data, _ = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'valid.txt', 'test.txt')
    except:
        total_data, _ = utils.load_quadruples('./data/' + args.dataset, 'train.txt', 'test.txt')
        print(args.dataset, 'does not have valid set.')

    train_sub = '/train_history_sub.txt'
    train_ob = '/train_history_ob.txt'
    train_s_label_f = '/train_s_label.txt'
    train_o_label_f = '/train_o_label.txt'
    train_s_frequency_f = '/train_s_frequency.txt'
    train_o_frequency_f = '/train_o_frequency.txt'

    dev_sub = '/dev_history_sub.txt'
    dev_ob = '/dev_history_ob.txt'
    dev_s_label_f = '/dev_s_label.txt'
    dev_o_label_f = '/dev_o_label.txt'
    dev_s_frequency_f = '/dev_s_frequency.txt'
    dev_o_frequency_f = '/dev_o_frequency.txt'

    test_sub = '/test_history_sub.txt'
    test_ob = '/test_history_ob.txt'
    test_s_label_f = '/test_s_label.txt'
    test_o_label_f = '/test_o_label.txt'
    test_s_frequency_f = '/test_s_frequency.txt'
    test_o_frequency_f = '/test_o_frequency.txt'

    with open('./data/' + args.dataset + train_sub, 'rb') as f:
        s_history_data = pickle.load(f)
    with open('./data/' + args.dataset + train_ob, 'rb') as f:
        o_history_data = pickle.load(f)
    with open('./data/' + args.dataset + train_s_label_f, 'rb') as f:
        train_s_label = pickle.load(f)
    with open('./data/' + args.dataset + train_o_label_f, 'rb') as f:
        train_o_label = pickle.load(f)
    with open('./data/' + args.dataset + train_s_frequency_f, 'rb') as f:
        if args.dataset == 'GDELT':
            train_s_frequency = torch.load(f).toarray()
        else:
            train_s_frequency = pickle.load(f).toarray()
    with open('./data/' + args.dataset + train_o_frequency_f, 'rb') as f:
        if args.dataset == 'GDELT':
            train_o_frequency = torch.load(f).toarray()
        else:
            train_o_frequency = pickle.load(f).toarray()

    s_history = s_history_data[0]
    s_history_t = s_history_data[1]
    o_history = o_history_data[0]
    o_history_t = o_history_data[1]

    with open('data/' + args.dataset + test_sub, 'rb') as f:
        s_history_test_data = pickle.load(f)
    with open('data/' + args.dataset + test_ob, 'rb') as f:
        o_history_test_data = pickle.load(f)
    with open('./data/' + args.dataset + test_s_label_f, 'rb') as f:
        test_s_label = pickle.load(f)
    with open('./data/' + args.dataset + test_o_label_f, 'rb') as f:
        test_o_label = pickle.load(f)
    with open('./data/' + args.dataset + test_s_frequency_f, 'rb') as f:
        test_s_frequency = pickle.load(f).toarray()
    with open('./data/' + args.dataset + test_o_frequency_f, 'rb') as f:
        test_o_frequency = pickle.load(f).toarray()

    s_history_test = s_history_test_data[0]
    s_history_test_t = s_history_test_data[1]
    o_history_test = o_history_test_data[0]
    o_history_test_t = o_history_test_data[1]

    with open('data/' + args.dataset + dev_sub, 'rb') as f:
        s_history_dev_data = pickle.load(f)
    with open('data/' + args.dataset + dev_ob, 'rb') as f:
        o_history_dev_data = pickle.load(f)
    with open('./data/' + args.dataset + dev_s_label_f, 'rb') as f:
        dev_s_label = pickle.load(f)
    with open('./data/' + args.dataset + dev_o_label_f, 'rb') as f:
        dev_o_label = pickle.load(f)
    with open('./data/' + args.dataset + dev_s_frequency_f, 'rb') as f:
        dev_s_frequency = pickle.load(f).toarray()
    with open('./data/' + args.dataset + dev_o_frequency_f, 'rb') as f:
        dev_o_frequency = pickle.load(f).toarray()

    s_history_dev = s_history_dev_data[0]
    s_history_dev_t = s_history_dev_data[1]
    o_history_dev = o_history_dev_data[0]
    o_history_dev_t = o_history_dev_data[1]

    if not args.only_eva and not args.only_oracle:
        model = CENET(num_nodes, num_rels, num_t, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if use_cuda:
            model = model.cuda()
        now = datetime.datetime.now()
        dt_string = args.description + now.strftime("%d-%m-%Y,%H-%M-%S") + \
                    args.dataset + '-EPOCH' + str(args.max_epochs)
        main_dirName = os.path.join(args.save_dir, dt_string)
        if not os.path.exists(main_dirName):
            os.makedirs(main_dirName)

        model_path = os.path.join(main_dirName, 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        settings['main_dirName'] = main_dirName
        file_training = open(os.path.join(main_dirName, "training_record.txt"), "w")
        file_training.write("Training Configuration: \n")
        file_validation = open(os.path.join(main_dirName, "validation_record.txt"), "w")
        file_validation.write("Configuration: \n")
        for key in settings:
            file_training.write(key + ': ' + str(settings[key]) + '\n')
            file_validation.write(key + ': ' + str(settings[key]) + '\n')
        for arg in vars(args):
            file_training.write(arg + ': ' + str(getattr(args, arg)) + '\n')
            file_validation.write(arg + ': ' + str(getattr(args, arg)) + '\n')
        print("Start training...")
        file_training.write("Training Start \n")
        file_training.write("===============================\n")

        epoch = 0
        best_oracle_all_mrr_lk = 0
        while epoch < args.max_epochs:
            model.train()
            epoch += 1
            print('$Start Epoch: ', epoch)
            loss_epoch = 0
            time_begin = time.time()
            _batch = 0

            for batch_data in utils.make_batch(train_data,
                                               s_history,
                                               o_history,
                                               train_s_label,
                                               train_o_label,
                                               train_s_frequency,
                                               train_o_frequency,
                                               args.batch_size):
                q_for_path = copy.deepcopy(batch_data[0][:, -1])
                batch_data[0] = torch.from_numpy(batch_data[0])
                batch_data[3] = torch.from_numpy(batch_data[3]).float()
                batch_data[4] = torch.from_numpy(batch_data[4]).float()
                batch_data[5] = torch.from_numpy(batch_data[5]).float()
                batch_data[6] = torch.from_numpy(batch_data[6]).float()

                if args.path:
                    if args.dataset.lower() == 'icews21':
                        one_hot_rel_path_list = []
                        cur_ts_sorted, ts_idx = torch.sort(batch_data[0][:, 3])
                        ts_all, example_group = torch.unique_consecutive(cur_ts_sorted, return_counts=True)
                        if len(ts_all) == 1:
                            cur_file = (ts_all.item() // 6)
                            all_rel_path = sp.load_npz(
                                'data/{}/history/rel_path_all_{}.npz'.format(args.dataset, cur_file + 1))
                            rel_path_idx = (ts_all.item() - (ts_all.item() // 6) * 6) * batch_data[0][:,
                                                                                        0] * num_nodes + batch_data[0][
                                                                                                         :, 2]
                            rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
                            one_hot_rel_path = rel_path.masked_fill(rel_path != 0, 1)
                        else:
                            sub, obj = batch_data[0][:, 0], batch_data[0][:, 2]
                            sub_sorted, obj_sorted = sub[ts_idx], obj[ts_idx]
                            k = 0
                            for i, t in enumerate(ts_all):
                                cur_file = (t.item() // 6)
                                all_rel_path = sp.load_npz(
                                    'data/{}/history/rel_path_all_{}.npz'.format(args.dataset, cur_file + 1))
                                rel_path_idx = (t.item() - (t.item() // 6) * 6) * sub_sorted[k:(
                                            k + example_group[i])] * num_nodes + obj_sorted[k:(k + example_group[i])]
                                rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
                                one_hot_rel_path_list.append(rel_path.masked_fill(rel_path != 0, 1))
                                k += example_group[i]
                            one_hot_rel_path_ = torch.cat(one_hot_rel_path_list, dim=0)
                            one_hot_rel_path = torch.zeros_like(one_hot_rel_path_)
                            one_hot_rel_path[ts_idx] = one_hot_rel_path_

                    else:
                        all_rel_path = sp.load_npz(
                            'data/{}/history/rel_path_all.npz'.format(args.dataset))
                        rel_path_idx = batch_data[0][:, 3] * batch_data[0][:, 0] * num_nodes + batch_data[0][:, 2]
                        rel_path = torch.Tensor(all_rel_path[rel_path_idx].todense())
                        one_hot_rel_path = rel_path.masked_fill(rel_path != 0, 1)
                else:
                    one_hot_rel_path = None

                if use_cuda:
                    batch_data[0] = batch_data[0].cuda()
                    batch_data[3] = batch_data[3].cuda()
                    batch_data[4] = batch_data[4].cuda()
                    batch_data[5] = batch_data[5].cuda()
                    batch_data[6] = batch_data[6].cuda()
                    if args.path:
                        one_hot_rel_path = one_hot_rel_path.cuda()

                batch_loss = model(batch_data, 'Training', rel_path_vocabulary=one_hot_rel_path)
                if batch_loss is not None:
                    error = batch_loss
                else:
                    continue
                error.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += error.item()

                print("# CENET batch: " + str(_batch) + ' finished. Used time: ' +
                      str(time.time() - time_begin) + ', Loss: ' + str(error.item()))
                file_training.write(
                    "epoch: " + str(epoch) + "batch: " + str(_batch) + ' finished. Used time: '
                    + str(time.time() - time_begin) + ', Loss: ' + str(error.item()) + '\n')
                _batch += 1

            epoch_time = time.time()
            print("Done\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
                  format(epoch, loss_epoch / _batch, epoch_time - time_begin))
            file_training.write("******\nEpoch {:04d} | Loss {:.4f}| time {:.4f}".
                                format(epoch, loss_epoch / _batch, epoch_time - time_begin) + '\n')

            if epoch % args.valid_epochs == 0 and args.dataset != 'ICEWS14T':
                s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3 = valid.execute_valid(args, total_data,
                                                                                                     model, dev_data,
                                                                                                     s_history_dev,
                                                                                                     o_history_dev,
                                                                                                     dev_s_label,
                                                                                                     dev_o_label,
                                                                                                     dev_s_frequency,
                                                                                                     dev_o_frequency,
                                                                                                     num_nodes,
                                                                                                     max_train_t)
                file_validation.write("\nNo Oracle: \n")
                raw_all_mrr_lk = utils.write2file(s_ranks2, o_ranks2, all_ranks2, file_validation)
                file_validation.write("\nGT Oracle: \n")
                oracle_all_mrr_lk = utils.write2file(s_ranks3, o_ranks3, all_ranks3, file_validation)
                file_validation.write("\n-------------------------------------\n")
                if oracle_all_mrr_lk > best_oracle_all_mrr_lk:
                    best_oracle_all_mrr_lk = oracle_all_mrr_lk
                    torch.save(model, model_path + '/' + args.dataset + '_best.pth')

        s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3 = valid.execute_valid(args, total_data,
                                                                                             model, dev_data,
                                                                                             s_history_dev,
                                                                                             o_history_dev,
                                                                                             dev_s_label,
                                                                                             dev_o_label,
                                                                                             dev_s_frequency,
                                                                                             dev_o_frequency,
                                                                                             num_nodes,
                                                                                             max_train_t)
        file_validation.write("No Oracle: \n")
        raw_all_mrr_lk = utils.write2file(s_ranks2, o_ranks2, all_ranks2, file_validation)
        file_validation.write("\nGT Oracle: \n")
        oracle_all_mrr_lk = utils.write2file(s_ranks3, o_ranks3, all_ranks3, file_validation)
        if oracle_all_mrr_lk > best_oracle_all_mrr_lk:
            best_oracle_all_mrr_lk = oracle_all_mrr_lk
            torch.save(model, model_path + '/' + args.dataset + '_best.pth')

        print("Training done")
        file_training.write("Training done")
        file_training.close()
        file_validation.close()

    if args.only_oracle or args.only_eva:
        dt_string = args.model_dir
        main_dirName = os.path.join(args.save_dir, dt_string)
        model_path = os.path.join(main_dirName, 'models')
        settings['main_dirName'] = main_dirName

    oracle_epoch = 0
    model = torch.load(model_path + '/' + args.dataset + '_best.pth')
    optimizer_oracle = torch.optim.Adam(model.parameters(), lr=args.oracle_lr, weight_decay=args.weight_decay)
    # optimizer_oracle = torch.optim.SGD(model.parameters(), lr=args.oracle_lr, momentum=0.9, weight_decay=args.weight_decay)
    model.freeze_parameter()  # freeze parameter except Oracle
    file_oracle = open(os.path.join(main_dirName, "training_oracle_record.txt"), "w")

    while oracle_epoch < args.oracle_epochs:
        oracle_epoch += 1
        total_oracle_loss = 0

        for batch_data in utils.make_batch(train_data,
                                           s_history,
                                           o_history,
                                           train_s_label,
                                           train_o_label,
                                           train_s_frequency,
                                           train_o_frequency,
                                           args.batch_size):
            batch_data[0] = torch.from_numpy(batch_data[0])
            batch_data[3] = torch.from_numpy(batch_data[3]).float()
            batch_data[4] = torch.from_numpy(batch_data[4]).float()
            batch_data[5] = torch.from_numpy(batch_data[5]).float()
            batch_data[6] = torch.from_numpy(batch_data[6]).float()
            if use_cuda:
                batch_data[0] = batch_data[0].cuda()
                batch_data[3] = batch_data[3].cuda()
                batch_data[4] = batch_data[4].cuda()
                batch_data[5] = batch_data[5].cuda()
                batch_data[6] = batch_data[6].cuda()
            oracle_loss = model(batch_data, 'Oracle')
            if oracle_loss is not None:
                error = oracle_loss
            else:
                continue
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            optimizer_oracle.step()
            optimizer_oracle.zero_grad()
            total_oracle_loss += error.item()
            # print('Oracle batch loss:', error.item())
            file_oracle.write('Oracle batch loss:' + str(error.item()) + '\n')
        print('oracle_epoch:', oracle_epoch, ' Oracle loss: ', total_oracle_loss)
        file_oracle.write('oracle_epoch:' + str(oracle_epoch) +
                          ' Oracle loss: ' + str(total_oracle_loss) + '\n')
    torch.save(model, model_path + '/' + args.dataset + '_best.pth')
    file_oracle.close()

    # Evaluation
    if args.only_eva:
        dt_string = args.model_dir
        main_dirName = os.path.join(args.save_dir, dt_string)
        model_path = os.path.join(main_dirName, 'models')
        settings['main_dirName'] = main_dirName
    if args.filtering:
        if args.only_eva:
            file_test_path = os.path.join(main_dirName, "test_record_filtering_eva.txt")
            # file_test_dev_path =
        else:
            file_test_path = os.path.join(main_dirName, "test_record_filtering.txt")
            # file_test_dev_path =
    else:
        if args.only_eva:
            file_test_path = os.path.join(main_dirName, "test_record_raw_eva.txt")
        else:
            file_test_path = os.path.join(main_dirName, "test_record_raw.txt")

    file_test = open(file_test_path, "w")
    # file_test_dev = open(file_test_dev_path)
    print("Testing starts")
    file_test.write("Testing starts: \n")
    model = torch.load(model_path + '/' + args.dataset + '_best.pth')
    model.eval()
    model.args = args

    s_ranks1, o_ranks1, all_ranks1, s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3 \
        = test.execute_test(args, total_data, model, test_data,
                            s_history_test, o_history_test,
                            test_s_label, test_o_label,
                            test_s_frequency, test_o_frequency, num_nodes, max_train_t)

    # evaluation for link prediction
    file_test.write("Oracle: \n")
    utils.write2file(s_ranks1, o_ranks1, all_ranks1, file_test)

    file_test.write("\n\nNo Oracle: \n")
    utils.write2file(s_ranks2, o_ranks2, all_ranks2, file_test)
    file_test.write("\nGT Oracle: \n")
    utils.write2file(s_ranks3, o_ranks3, all_ranks3, file_test)

    s_ranks1, o_ranks1, all_ranks1, s_ranks2, o_ranks2, all_ranks2, s_ranks3, o_ranks3, all_ranks3 \
        = test.execute_test(args, total_data, model, dev_data,
                            s_history_dev, o_history_dev,
                            dev_s_label, dev_o_label,
                            dev_s_frequency, dev_o_frequency, num_nodes, max_train_t)

    # evaluation for link prediction
    file_test.write("=====VALIDATION===== \n")
    file_test.write("Oracle: \n")
    utils.write2file(s_ranks1, o_ranks1, all_ranks1, file_test)

    file_test.write("\n\nNo Oracle: \n")
    utils.write2file(s_ranks2, o_ranks2, all_ranks2, file_test)
    file_test.write("\nGT Oracle: \n")
    utils.write2file(s_ranks3, o_ranks3, all_ranks3, file_test)

    file_test.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CENET')
    parser.add_argument("--description", type=str, default='your_description_for_folder_name', help="description")
    parser.add_argument("-d", "--dataset", type=str, default='YAGO', help="dataset")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=30, help="maximum epochs")
    parser.add_argument("--oracle-epochs", type=int, default=20, help="maximum oracle epochs")
    parser.add_argument("--valid-epochs", type=int, default=5, help="validation epochs")
    parser.add_argument("--alpha", type=float, default=0.1, help="alpha for nceloss")
    parser.add_argument("--lambdax", type=float, default=10, help="lambda")

    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--oracle_lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--oracle_mode", type=str, default='soft', help="soft and hard mode for Oracle")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--grad-norm", type=float, default=1.0, help="norm to clip gradient to")
    parser.add_argument("--filtering", type=utils.str2bool, default=True)

    parser.add_argument("--only_oracle", type=utils.str2bool, default=False, help="whether only to train oracle")
    parser.add_argument("--only_eva", type=utils.str2bool, default=False, help="whether only evaluation on test set")
    parser.add_argument("--model_dir", type=str, default="", help="model directory")
    parser.add_argument("--save_dir", type=str, default="SAVE", help="save directory")
    parser.add_argument("--eva_dir", type=str, default="SAVE", help="saved dir of the testing model")
    parser.add_argument("--gpu", type=str, default="3", help="which gpu")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    # new settings
    parser.add_argument("--LLM", action='store_true', default=False,
                        help="whether to use LLM encoded embedding")
    parser.add_argument("--pure_LLM", action='store_true', default=False,
                        help="whether to only use LLM encoded embedding for relations")
    parser.add_argument("--concept", action='store_true', default=False,
                        help="whether to use concept")
    parser.add_argument("--path", action='store_true', default=False,
                        help="whether to use relation path")
    parser.add_argument("--path_method", type=str, default='tucker',
                        help="which method used in relation path")
    parser.add_argument("--LLM_path", action='store_true', default=False,
                        help="whether to only use LLM encoded embedding for relation path")
    parser.add_argument('--gamma_init', type=float, default=0.01, help='initialiation of gamma value')
    args_main = parser.parse_args()
    print(args_main)
    if not os.path.exists(args_main.save_dir):
        os.makedirs(args_main.save_dir)

    GPU_ID = args_main.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('***************GPU_ID***************: ', GPU_ID)
    else:
        raise NotImplementedError

    # seed = 999
    np.random.seed(args_main.seed)
    torch.manual_seed(args_main.seed)
    main_portal(args_main)
