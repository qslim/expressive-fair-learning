import yaml
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
from torch.multiprocessing import Process, set_start_method, Queue
sys.path.append('..')
from uniconv import UniConvWrapper
from data.Preprocessing import load_data
import scipy as sp
from utils import seed_everything, init_params, accuracy, fair_metric, orthogonal_projection


def main_worker(seed, result_queue, config, E, U, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train):
    seed_everything(seed)

    net = UniConvWrapper(nfeat=x.size(1),
                         config=config).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_acc = 0.0
    best_epoch = -1
    best_test = 0.0
    best_dp_test = 1e5
    best_eo_test = 1e5
    for epoch in range(config['epoch_fit'] + config['epoch_debias']):
        net.train()
        optimizer.zero_grad()

        output, output_sens = net(E, U, x)

        if epoch >= config['epoch_fit']:
            output = orthogonal_projection(output, output_sens, config)

        loss_sens = F.binary_cross_entropy_with_logits(output_sens[idx_sens_train],
                                                  sens[idx_sens_train].unsqueeze(1).float())
        loss_cls = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        loss = loss_cls + loss_sens
        acc_train = accuracy(output[idx_train], labels[idx_train]) * 100.0
        loss.backward()
        optimizer.step()

        net.eval()
        output, output_sens = net(E, U, x)

        acc_val = accuracy(output[idx_val], labels[idx_val]) * 100.0
        acc_test = accuracy(output[idx_test], labels[idx_test]) * 100.0
        parity_test, equality_test = fair_metric(output, idx_test, labels, sens)
        parity_test, equality_test = parity_test * 100.0, equality_test * 100.0

        if (config['epoch_debias'] == 0 or epoch > config['epoch_fit'] + config['patience']) and acc_val > best_acc:
            best_acc = acc_val.item()
            best_epoch = epoch
            best_test = acc_test.item()
            best_dp_test = parity_test
            best_eo_test = equality_test

            print("Epoch {}:".format(epoch),
                  "[Acc tr: {:.4f}".format(acc_train.item()),
                  "va: {:.4f}".format(acc_val.item()),
                  "te: {:.4f}]".format(acc_test.item()),
                  "[DP: {:.4f}".format(parity_test),
                  "EO: {:.4f}]".format(equality_test),
                  "{}/{:.4f}".format(best_epoch, best_test))

    # Put the results in the queue
    result_queue.put((best_test, best_dp_test, best_eo_test, best_epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--dataset', default='german')
    args = parser.parse_args()

    config = yaml.load(open('./config.yaml'), Loader=yaml.SafeLoader)[args.dataset]
    config['seeds'] = args.seeds
    config['dataset'] = args.dataset

    adj, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_data(path_root='./',
                                                                                   dataset=config['dataset'],
                                                                                   label_number=config['label_number'],
                                                                                   sens_number=config['sens_number'])
    assert (torch.equal(torch.abs(labels[idx_train] - 0.5) * 2.0, torch.ones_like(labels[idx_train])))
    assert (torch.equal(torch.abs(labels[idx_val] - 0.5) * 2.0, torch.ones_like(labels[idx_val])))
    assert (torch.equal(torch.abs(labels[idx_test] - 0.5) * 2.0, torch.ones_like(labels[idx_test])))
    assert (torch.equal(torch.abs(sens - 0.5) * 2.0, torch.ones_like(sens)))

    eigsh_which = 'LM'
    dataset_path = './pt/{}_{}{}.pt'.format(config['dataset'], eigsh_which, config['eigk'])
    if os.path.exists(dataset_path):
        e, u = torch.load(dataset_path)
    else:
        e, u = [], []
        deg = np.array(adj.sum(axis=0)).flatten()
        for eps in [-0.5]:
            print("Start building e, u with {}...".format(eps), end='')
            # build graph matrix
            D_ = sp.sparse.diags(deg ** eps)
            A_ = D_.dot(adj.dot(D_))
            # eigendecomposition
            _e, _u = sp.sparse.linalg.eigsh(A_, which=eigsh_which, k=config['eigk'], tol=1e-5)
            e.append(torch.FloatTensor(_e))
            u.append(torch.FloatTensor(_u))
            print("Done.")
        e, u = torch.cat(e, dim=0).cuda(), torch.cat(u, dim=1).cuda()
        torch.save([e, u], dataset_path)

    # Set the start method to 'spawn' for Windows compatibility
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    acc_test, dp_test, eo_test = [], [], []
    # Create a queue for collecting results
    result_queue = Queue()
    processes = []
    for seed in config['seeds']:
        p = Process(target=main_worker, args=(seed, result_queue, config, e, u, x, labels, idx_train, idx_val, idx_test, sens, idx_sens_train))
        # Start the process
        p.start()
        # Append the process to the list
        processes.append(p)

    for p in processes:
        p.join()
    # Collect results from the queue
    while not result_queue.empty():
        (_acc_test, _dp_test, _eo_test, _best_epoch) = result_queue.get()
        acc_test.append(_acc_test)
        dp_test.append(_dp_test)
        eo_test.append(_eo_test)

        print("Test results:",
              "Acc: {:.4f}".format(_acc_test),
              "DP: {:.4f}".format(_dp_test),
              "EO: {:.4f}".format(_eo_test),
              "Epoch: {}".format(_best_epoch))

    acc_test = np.array(acc_test, dtype=float)
    dp_test = np.array(dp_test, dtype=float)
    eo_test = np.array(eo_test, dtype=float)

    ACC = "{:.2f} $\pm$ {:.2f}".format(np.mean(acc_test), np.std(acc_test))
    DP = "{:.2f} $\pm$ {:.2f}".format(np.mean(dp_test), np.std(dp_test))
    EO = "{:.2f} $\pm$ {:.2f}".format(np.mean(eo_test), np.std(eo_test))
    print("Mean over {} run:".format(len(config['seeds'])),
          "Acc: " + ACC,
          "DP: " + DP,
          "EO: " + EO)

if __name__ == '__main__':
    main()
