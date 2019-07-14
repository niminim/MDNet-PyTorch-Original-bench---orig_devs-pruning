import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np

import torch

sys.path.insert(0,'.')
from data_prov import RegionDataset
from modules.model import MDNet, set_optimizer, BCELoss, Precision

# python pretrain/train_mdnet.py -d otb100

# cd Desktop/NN/MDNet-PyTorch-Original-bench\ -\ orig_devs-pruning
# python pretrain/train_mdnet.py


def train_mdnet(opts):

    # Init dataset
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp) # keys - folder names, values - ['images'] - list of images, and ['gt'] -  list of gt. For each folder
        
    K = len(data)
    dataset = [None] * K
    print('Create Dataset')
    
    #print('data: ' , data.values())
    for k, seq in enumerate(data.values()):
        # seq['images'] - datasets/VOT/vot2016/bag/00000001.jpg,...... (in VOT)
        # seq['images'] - datasets/OTB/otb100/Basketball/img/0001.jpg,...... (in OTB)
        # seq['gt'] - a list of all gt of current sequence
    
        dataset[k] = RegionDataset(seq['images'], seq['gt'], opts) # returns regions   
        
    print('')

    # Init model
    print('Create model')
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
        
    print('Set Learnable parameters: ') 
    model.set_learnable_params(opts['ft_layers']) # all layers
    #print(model.get_learnable_params().keys())
    #print('')

    # Init criterion and optimizer
    print('Init loss criterions')
    criterion_scores = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])

    tic_all = time.time()
    loss_all = np.zeros((opts['n_cycles'],K)) # loss of all sequences of each cycle
    mean_loss_all = np.zeros((opts['n_cycles'],1))

    print('Main training loop')
    # Main trainig loop
    for i in range(opts['n_cycles']): # go over all epochs
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            #print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        #print('Start training')
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K) # a list of all branches (numbers)
        losses = np.zeros((len(k_list),1))
        #print('branches: ', k)
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            #print('pos_regions, neg_regions')

            pos_regions, neg_regions = dataset[k].next()
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda() # shape: [32, 3, 107, 107]
                neg_regions = neg_regions.cuda() # shape: [96, 3, 107, 107]
                                        
            #print('Get pos & neg score from the model')
            pos_score = model(pos_regions, k) # shape: [32,2]
            neg_score = model(neg_regions, k) # shape: [96,2]
            
            #print('')
            
            #print('Loss criterions')
            criterion_scores = BCELoss()

            #print('Calculating loss:')
            loss = criterion_scores(pos_score, neg_score)
            losses[k] = loss.data[0] 
            loss_all[i,k] = losses[k]

            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            #print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
            #        .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc))
            
            
        mean_losses = losses.mean()
        mean_loss_all[i] = loss_all[i,:].mean()

        toc_all = time.time() - tic_all
        print('Mean Precision: {:.3f}'.format(prec.mean()))
        print('Mean Loss: {:.2f}'.format(mean_losses))
        print('Total Time [min]: {:.2f}'.format(toc_all/60))     
        print('Save model to {:s}'.format(opts['model_path']))
        
        if opts['use_gpu']:
            model = model.cuda()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()
        print('')
        if opts['use_gpu']:
            model = model.cuda()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()
            
    import matplotlib.pyplot as plt
    plt.plot(mean_loss_all)
    plt.ylabel('accuracy loss')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='otb100', help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    opts = yaml.safe_load(open('pretrain/options_{}.yaml'.format(args.dataset), 'r'))
    train_mdnet(opts)
