import os
import json
import numpy as np

import os

from defs import *

cwd = os.getcwd()
in_tracking_folder = (cwd[-8:] == 'tracking')


def prin_gen_config(args):
    if args.seq != '':
        # generate config from a sequence name

        # seq_home = '/data1/tracking/princeton/validation'
        # seq_home = '../dataset/OTB'
        seq_home = args.seq_home
        save_home = '../result_fig'
        result_home = '../result'

        seq_name = args.seq
        
        if (benchmark_dataset[0:3] == 'OTB'):
            # img_dir = os.path.join(seq_home, seq_name, 'rgb')
            img_dir = os.path.join(seq_home, seq_name, 'img')
            # img_dir = seq_home + '/' + seq_name + '/rgb'
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')
            # gt_path = seq_home + '/' + seq_name + '/groundtruth_rect.txt'
        elif (benchmark_dataset[0:3] == 'VOT'):
            # img_dir = os.path.join(seq_home, seq_name, 'rgb')
            img_dir = os.path.join(seq_home, seq_name)
            # img_dir = seq_home + '/' + seq_name + '/rgb'
            gt_path = os.path.join(seq_home, seq_name, 'groundtruth.txt')
            # gt_path = seq_home + '/' + seq_name + '/groundtruth_rect.txt'

        # print('loading images from: ', img_dir)

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list if os.path.splitext(x)[1] == '.jpg'] # look for jpg (in vot there are also tag files)
        
        try:
            gt = np.loadtxt(gt_path, delimiter=',')
        except:
            gt = np.loadtxt(gt_path, delimiter='	')

        #gt = np.loadtxt(gt_path, delimiter=',')
        if gt.shape[1] == 5:
            gt = gt[:, :-1]
        init_bbox = gt[0]
        
        if gt.shape[1] == 8: # vot (code from prepro_vot)
            x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
            y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
            gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
            
            
        savefig_dir = os.path.join(save_home, seq_name)
        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # result_path = os.path.join(result_dir, 'result.json')
        result_path = result_dir

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path


def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name

        if in_tracking_folder:
            seq_home = '../datasets/OTB'
            result_home = '../results'
        else:
            seq_home = 'datasets/OTB'
            result_home = 'results'

        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]
        
        with open(gt_path) as f:
            gt = np.loadtxt((x.replace('\t',',') for x in f), delimiter=',')
        init_bbox = gt[0]

        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        savefig_dir = os.path.join(result_dir, 'figs')
        result_path = os.path.join(result_dir, 'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path
