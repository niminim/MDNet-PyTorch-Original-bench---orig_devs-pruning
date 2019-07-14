import scipy
import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config import gen_config
from gen_config import prin_gen_config
from gen_config import in_tracking_folder

import statistics
import itertools
from cycler import cycler

from tqdm import tqdm


#######################################################
from defs import *

if in_tracking_folder:
    opts = yaml.safe_load(open('options.yaml','r'))
    print(' in tracking_folder')
else:
    opts = yaml.safe_load(open('tracking/options.yaml', 'r'))
    opts['model_path'] = opts['model_path'][3:]
    print('not in tracking folder')

models_paths[1] = opts['model_path'] # define model path

#if torch.cuda.is_available():
#    torch.cuda.empty_cache()
#    training_device = torch.device('cuda:0')
#    total_mem = torch.cuda.get_device_properties(training_device).total_memory
#    if total_mem > 2500000000:  # 2.5 GB
#        opts['use_gpu'] = True
#    else:
#        opts['use_gpu'] = False
#else:
#    opts['use_gpu'] = False

fewer_images = not opts['use_gpu']
#fewer_images = False

#######################################################


def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats



def train(model, loss_criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4', loss_index=1):
    
    # train the network (in_layer = first layer to train)
    # k = 0 in online learning
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        #print('hard negative mining')
        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer) ### shape: [256, 3], chop the iou. iou_prev is dummy
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        ### get the scores and iou. the higher the scores the higher the performance in pos/neg task
        pos_score = model(batch_pos_feats, k=0, in_layer=in_layer)
        neg_score = model(batch_neg_feats, k=0, in_layer=in_layer)
        
        criterion = BCELoss()
        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False,
              loss_index=1, model_path=opts['model_path'], seq_name=None):

#def run_mdnet(k, img_list, init_bbox, gt=None, savefig_dir='', display=False,
#              loss_index=1, model_path=opts['model_path'], seq_name=None):
    

    ############################
    if fewer_images:
        num_images = min(sequence_len_limit, len(img_list))
    else:
        num_images = len(img_list)
    ############################

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox
    
    # Init iou and pred_iou
    iou_list = np.zeros((len(img_list),1)) # shape: [113.1) ### list of ious
    iou_list[0] = 1.0                      ### in first frame gt=result_bb (by definition)
    

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1
        
    # Init model
    # model = MDNet(model_path=opts['model_path'],use_gpu=opts['use_gpu'])
    model = MDNet(model_path=model_path, use_gpu=opts['use_gpu'])
    if opts['use_gpu']:
        model = model.cuda()
        
    print('Init criterion and optimizer')
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])


    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    print('Draw pos/neg samples')
    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    print('Extract pos/neg features')
    # Extract pos/neg features
    if fewer_images:  # shorter run in general, less accurate
        pos_feats = forward_samples(model, image, pos_examples)  
        neg_feats = forward_samples(model, image, neg_examples)
    else:
        pos_feats = forward_samples(model, image, pos_examples) 
        neg_feats = forward_samples(model, image, neg_examples)
        
    print('Initial training')
    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'],loss_index=loss_index)  ### iou_pred_list
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    print('Train bbox regressor')
    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples) # calc features ### shape: [927, 4608] ###
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    print('Init pos/neg features for update')
    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
            #################
            num_gts = np.minimum(gt.shape[0], num_images)
            # print('num_gts.shape: ', num_gts.shape)
            gt_centers = gt[:num_gts, :2] + gt[:num_gts, 2:] / 2
            result_centers = np.zeros_like(gt[:num_gts, :2])
            result_centers[0] = gt_centers[0]
            result_ious = np.zeros(num_gts, dtype='float64')
            result_ious[0] = 1.
            #################

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    print('Main Loop')
    # Main loop
    spf_total = 0  # I don't want to take into account initialization
    for i in tqdm(range(1, num_images)):
    # for i in range(1, len(img_list)):
        #print('Frame: ', i)
        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        #print('Estimate target bbox (in run_mdnet)')
        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])
            
        #print('Bbox regression (in run_mdnet)')
        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox
        iou_list[i] = overlap_ratio(gt[i],result_bb[i])

        ###########################################
        # identify tracking failure and abort when in VOT mode
        
        IoU = overlap_ratio(result_bb[i], gt[i])[0]
        if (IoU == 0) and init_after_loss:
            print('    * lost track in frame %d since init*' % (i))
            result_distances = scipy.spatial.distance.cdist(result_centers[:i], gt_centers[:i], metric='euclidean').diagonal()
            num_images_tracked = i - 1  # we don't count frame 0 and current frame (lost track)

            im.set_data(image)
            if gt is not None:
                if i < gt.shape[0]:
                    gt_rect.set_xy(gt[i, :2])
                    gt_rect.set_width(gt[i, 2])
                    gt_rect.set_height(gt[i, 3])
                else:
                    gt_rect.set_xy(np.array([np.nan, np.nan]))
                    gt_rect.set_width(np.nan)
                    gt_rect.set_height(np.nan)

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            plt.pause(.01)
            plt.draw()
            
            print('Finished identify tracking failure and abort when in VOT mode')
            return result[:i], result_bb[:i], num_images_tracked, spf_total, result_distances, result_ious[:i], True
        ########################################


        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf
        #print('Time: ', spf)


        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

                #################
                result_ious[i] = overlap_ratio(result_bb[i], gt[i])[0]
                result_centers[i] = result_bb[i, :2] + result_bb[i, 2:] / 2
                #################

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)

        ####################################
        if detailed_printing:
            if gt is None:
                print("      Frame %d/%d, Score %.3f, Time %.3f" % \
                      (i, num_images-1, target_score, spf))
            else:
                if i<gt.shape[0]:
                    print("      Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                        (i, num_images-1, overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
                else:
                    print("      Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                        (i, num_images-1, overlap_ratio(np.array([np.nan,np.nan,np.nan,np.nan]), result_bb[i])[0], target_score, spf))
        ####################################

        # if gt is None:
        #     print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
        #         .format(i, len(img_list), target_score, spf))
        # else:
        #     overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
        #     print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
        #         .format(i, len(img_list), overlap[i], target_score, spf))

    ########################
    plt.close()
    result_distances = scipy.spatial.distance.cdist(result_centers, gt_centers, metric='euclidean').diagonal()
    num_images_tracked = num_images - 1  # I don't want to count initialization frame (i.e. frame 0)
    print('    main loop finished, %d frames' % (num_images))
    
    print('mean IoU: ', iou_list.mean())
    print('Finished run_mdnet()')
         
    return result, result_bb, num_images_tracked, spf_total, result_distances, result_ious, False
    ########################


    # if gt is not None:
    #     print('meanIOU: {:.3f}'.format(overlap.mean()))
    # fps = len(img_list) / spf_total
    # return result, result_bb, fps, iou_list, iou_pred_list ### add iou_list and pred_list


if __name__ == "__main__":

    #np.random.seed(123)
    #torch.manual_seed(456)
    #torch.cuda.manual_seed(789)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='seq_list', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true', default=False)
    parser.add_argument('-d', '--display', action='store_true', default=False)  # display tracking

    parser.add_argument('-sh', '--seq_home', default=seq_home, help='input seq_home')
    # parser.add_argument('-tr', '--perform_tracking', default='True', help='(boolean) perform tracking')
    parser.add_argument('-at', '--attributes', default='selection', help='attributes separated by -')
    parser.add_argument('-i', '--init_after_loss', action='store_true')
    parser.add_argument('-l', '--lmt_seq', action='store_true')
    parser.add_argument('-ln', '--seq_len_lmt', default='0', help='sequence length limit')

    parser.add_argument('-p', '--plt_bnch', action='store_true')  # plot benchmark graphs from saved results
    parser.add_argument('-tr', '--perform_tracking', action='store_true')  # plot benchmark graphs from saved results
    parser.add_argument('-save', '--save_model', action='store_true', default=False, help='save the model after training' )  # save the model after training

    args = parser.parse_args()
    
    print('Benchmark dataset: ', benchmark_dataset[0:3]) 
    print('Model path: ', opts['model_path'])
    print('perform tracking: ', args.perform_tracking)
    print('args.lmt_seq: ', args.lmt_seq) 
    print('fewer_images: ', fewer_images) 

    if args.lmt_seq:
        fewer_images = True
        args.seq_len_lmt = int(float(args.seq_len_lmt))
        if args.seq_len_lmt > 0:
            sequence_len_limit = args.seq_len_lmt
        # else, sequence_len_limit stays with default value
        print('sequence_len_limit: ', sequence_len_limit) 
        
    init_after_loss = args.init_after_loss
    
    #print('OTB_attributes_dict: ', OTB_attributes_dict) #print  (a dictionary of each attribute and its videos)

    print('args.attributes: ', args.attributes) #print
    if args.attributes == 'selection':
        select_attributes_strings = OTB_select_attributes_strings
        print('select_attributes_strings1: ', select_attributes_strings) #print
    elif args.attributes == 'all':
        select_attributes_strings = list(OTB_attributes_dict.keys())
        print('select_attributes_strings2: ', select_attributes_strings)#print
    else:
        select_attributes_strings = args.attributes.split('-')
        select_attributes_strings = list(set(select_attributes_strings).intersection(set(OTB_attributes_dict.keys())))  # clean typos
        select_attributes_strings.sort()
        print('select_attributes_strings3: ', select_attributes_strings)#print

    print('attributes selected: ', select_attributes_strings)
    

    if (benchmark_dataset[0:3] == 'OTB'):
        
        sequence_wish_list = []
        for att in select_attributes_strings:
            sequence_wish_list.extend(OTB_attributes_dict[att])
        sequence_wish_list = list(set(sequence_wish_list))
        sequence_wish_list.sort()
        my_sequence_list = list(set(my_sequence_list).intersection(set(sequence_wish_list)))
        my_sequence_list.sort()
        print('my_sequence_list: ', my_sequence_list) #print
        print('my_sequence_list_len: ', len(my_sequence_list)) #print

        print('args.seq: ', args.seq) 
        if args.seq == 'seq_list':
            sequence_list = my_sequence_list
        elif args.seq == 'all':
           sequence_list = next(os.walk(seq_home))[1]
        else:
            sequence_list = [args.seq]  # e.g. -s DragonBaby
        print('')
        print('sequence_list ', sequence_list) 
            
    elif (benchmark_dataset[0:3] == 'VOT'):
        
        print('args.seq: ', args.seq) 
        if args.seq == 'seq_list':
            sequence_list = my_sequence_list
        print('sequence_list ', sequence_list) 
        print('')
        
    
    # perform_tracking = (args.perform_tracking == 'True')
    perform_tracking = args.perform_tracking
    print('perform_tracking: ', perform_tracking) #print
    print('')

    # ------
    # tracking + online training + save results
    if perform_tracking:
        print('Inside perform tracking')
        # for loss_index in loss_indices_for_tracking:  # we comapare several loss functions
        tracking_started = time.time()
        # model_index - iterate over different weights learnt
        # loss_index - iterate over different loss functions for online training
        # sequnce - iterate over different sequences
           
        for model_index, loss_index, sequence in itertools.product(models_indices_for_tracking, loss_indices_for_tracking, sequence_list):
            # ------
            # img_list - list of (relative path) file names of the jpg images
            #   example: '../dataset/OTB/DragonBaby/img/img####.jpg'
            # gt - a (2-dim, N x 4) list of 4 coordinates of ground truth BB for each image
            # init_bbox - this is gt[0]

            # Generate sequence config
            # img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
            
            # Generate sequence of princeton dataset config
            args.seq = sequence
            img_list, init_bbox, gt, savefig_dir, display, result_path = prin_gen_config(args)
            # ------

            tracking_start = time.time()
            print('')
            print('tracking: model ' + models_strings[model_index] + ', loss: ' + losses_strings[loss_index] + ', sequence: ' + sequence)
        
            # each run is random, so we need to average before comparing
            # each iteration starts from the finish of the offline training
            # there is no dependency between iterations
            for avg_iter in np.arange(0, avg_iters_per_sequence):
                
                print('Iteration %d / %d started' % (avg_iter+1, avg_iters_per_sequence))
                iteration_start = time.time()

                if init_after_loss:  # loss means loss of tracking
                    init_frame_index = 0
                    while init_frame_index < len(img_list) - 1:  # we want at least one frame for tracking after init
                        result, result_bb, num_images_tracked, spf_total, result_distances, result_ious, lost_track = run_mdnet(img_list[init_frame_index:], gt[init_frame_index], gt=gt[init_frame_index:], savefig_dir=savefig_dir, display=display, loss_index=loss_index, model_path=models_paths[model_index], seq_name=sequence)
                        if init_frame_index == 0:
                            result_ious_tot = result_ious
                            num_images_tracked_tot = num_images_tracked
                            spf_total_tot = spf_total

                            # init_frame_index does not include init frame nor frame where tracking was lost
                            if lost_track:
                                lost_track_tot = 1
                                init_frame_index = num_images_tracked + 1 + 5
                            else:
                                lost_track_tot = 0
                                init_frame_index = len(img_list)
                        else:
                            result_ious_tot = np.concatenate((result_ious_tot, result_ious))
                            num_images_tracked_tot += num_images_tracked
                            spf_total_tot += spf_total

                            if lost_track:
                                lost_track_tot += 1
                                init_frame_index += num_images_tracked + 1 + 5
                            else:
                                init_frame_index = len(img_list)
                    accuracy = np.mean(result_ious_tot)
                    fps = num_images_tracked_tot / spf_total_tot
                else:
                    lost_track_tot = 0
                    result, result_bb, num_images_tracked, spf_total, result_distances, result_ious, lost_track = run_mdnet(
                        img_list, gt[0], gt=gt,
                        savefig_dir=savefig_dir, display=display, loss_index=loss_index,
                        model_path=models_paths[model_index], seq_name=sequence)
                    accuracy = np.mean(result_ious)
                    fps = num_images_tracked / spf_total

                # compute step of running average of results over current sequence
                if not init_after_loss:
                    if avg_iter == 0:
                        result_distances_avg = result_distances
                        result_ious_avg = result_ious
                    else:
                        result_distances_avg = (result_distances_avg*avg_iter + result_distances) / (avg_iter+1)
                        result_ious_avg = (result_ious_avg * avg_iter + result_ious) / (avg_iter + 1)
                # else:
                if avg_iter == 0:
                    failures_per_seq_avg = lost_track_tot
                    accuracy_avg = accuracy
                else:
                    failures_per_seq_avg = (failures_per_seq_avg * avg_iter + lost_track_tot) / (avg_iter + 1)
                    accuracy_avg = (accuracy_avg * avg_iter + accuracy) / (avg_iter + 1)

                iteration_time = time.time() - iteration_start
                
                    
                print('End of iter')
                print('avg_iter: ', avg_iter) 
                print('avg_iters_per_sequence: ', avg_iters_per_sequence)
                print('Iteration time elapsed: %.3f' % (iteration_time))
                print('')
                
                
            print('End of sequence')
            
            # Save result
            res = {}
            res['type'] = 'rect'
            res['fps'] = fps
            if not init_after_loss:
                res['res'] = result_bb.round().tolist()  # what to save when we average ????
                res['ious'] = result_ious_avg.tolist()
                res['distances'] = result_distances_avg.tolist()
            # else:
            res['fails_per_seq'] = failures_per_seq_avg
            res['accuracy'] = accuracy_avg
            result_fullpath = os.path.join(result_path, 'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_init-' + str(init_after_loss) + '.json')
            json.dump(res, open(result_fullpath, 'w'), indent=2)

            tracking_time = time.time() - tracking_start
            print('tracking: model ' + models_strings[model_index] + ' loss ' + losses_strings[loss_index] + ' sequence ' + sequence + ' - elapsed %.3f' % (tracking_time))
            
         

        tracking_time = time.time() - tracking_started
        print('finished %d losses x %d models x %d sequences - elapsed %d' % (len(loss_indices_for_tracking), len(models_indices_for_tracking), len(sequence_list), tracking_time))
        print('Result IoUs: ' , result_ious.mean())




    # ------
    display_benchmark_results = args.plt_bnch
    if display_benchmark_results:

        for model_index, loss_index in itertools.product(models_indices_for_tracking, loss_indices_for_tracking):

            if show_average_over_sequences:
                if display_VOT_benchmark:
                    avg_accuracy = []
                    avg_fails = []
                if display_OTB_benchmark:
                    avg_success_rate = np.zeros((len(sequence_list),np.arange(0, 1.01, step=0.01).size))
                    avg_precision = np.zeros((len(sequence_list),np.arange(0, 50.5, step=0.5).size))

            for seq_iter, sequence in enumerate(sequence_list):

                args.seq = sequence
                img_list, init_bbox, gt, savefig_dir, display, result_path = prin_gen_config(args)

                if display_OTB_benchmark:

                    result_fullpath = os.path.join(result_path,'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_init-False' + '.json')
                    with open(result_fullpath, "r") as read_file:
                        res = json.load(read_file)

                    result_distances = np.asarray(res['distances'])
                    result_ious = np.asarray(res['ious'])

                    overlap_threshold = np.arange(0, 1.01, step=0.01)  # X axis
                    success_rate = np.zeros(overlap_threshold.size)
                    for i in range(overlap_threshold.shape[0]):
                        success_rate[i] = np.sum(result_ious > overlap_threshold[i]) / result_ious.shape[0]
                    # AUC = accuracy = sum(success_rate)

                    location_error_threshold = np.arange(0, 50.5, step=0.5)  # X axis
                    precision = np.zeros(location_error_threshold.size)
                    for i in range(location_error_threshold.shape[0]):
                        precision[i] = np.sum(result_distances < location_error_threshold[i]) / result_distances.shape[0]

                    if show_average_over_sequences:
                        avg_success_rate[seq_iter,:] = success_rate
                        avg_precision[seq_iter,:] = precision
                        if sequence == sequence_list[-1]:
                            avg_success_rate = avg_success_rate.mean(axis=0)
                            avg_precision = avg_precision.mean(axis=0)
                            plt.figure(11)
                            plt.plot(avg_success_rate,label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.ylabel('success rate')
                            plt.xlabel('overlap threshold')
                            plt.legend()

                            plt.figure(12)
                            plt.plot(avg_precision,label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.ylabel('precision')
                            plt.xlabel('location error threshold')
                            plt.legend()
                    if show_per_sequence:
                        plt.figure(2)
                        # plt.plot(result_distances, label=losses_strings[loss_index])
                        plt.plot(result_distances, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('distances')
                        plt.xlabel('image number')
                        plt.legend()

                        plt.figure(3)
                        plt.plot(result_ious, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('ious')
                        plt.xlabel('image number')
                        plt.legend()

                        plt.figure(4)
                        plt.plot(success_rate, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('success rate')
                        plt.xlabel('overlap threshold')
                        plt.legend()

                        plt.figure(5)
                        plt.plot(precision, label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('precision')
                        plt.xlabel('location error threshold')
                        plt.legend()

                if display_VOT_benchmark:

                    result_fullpath = os.path.join(result_path,'result_model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_init-' + str(init_after_loss) + '.json')
                    with open(result_fullpath, "r") as read_file:
                        res = json.load(read_file)

                    if show_average_over_sequences:
                        avg_accuracy.append(res['accuracy'])
                        avg_fails.append(res['fails_per_seq'])
                        if sequence == sequence_list[-1]:
                            avg_accuracy = statistics.mean(avg_accuracy)
                            avg_fails = statistics.mean(avg_fails)
                            plt.figure(6)
                            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) *
                                                       cycler('marker', ["o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "X", "D", "d"])))
                            plt.plot([avg_fails], [avg_accuracy], label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index])
                            plt.ylabel('accuracy')
                            plt.xlabel('failures per sequence')
                            plt.legend()
                    if show_per_sequence:
                        plt.figure(13)
                        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'c', 'k']) *
                                               cycler('marker',["o", "v", "^", "<", ">", "8", "s", "p", "P", "*", "h", "H", "X", "D", "d"])))
                        plt.plot([res['fails_per_seq']], [res['accuracy']], label='model-' + models_strings[model_index] + '_loss-' + losses_strings[loss_index] + '_sequence-' + sequence)
                        plt.ylabel('accuracy')
                        plt.xlabel('failures per sequence')
                        plt.legend()


        plt.show()



    # ---------------

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--seq', default='', help='input seq')
    # parser.add_argument('-j', '--json', default='', help='input json')
    # parser.add_argument('-f', '--savefig', action='store_true')
    # parser.add_argument('-d', '--display', action='store_true')


    # args.seq = 'DragonBaby'

    # assert(args.seq != '' or args.json != '')

    # #np.random.seed(0)
    # #torch.manual_seed(0)
    #
    # # Generate sequence config
    # img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)
    #
    # # Run tracker
    # result, result_bb, fps, iou_list, iou_pred_list = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=True) ### add iou_list
    #
    # # Save result
    # res = {}
    # res['res'] = result_bb.round().tolist()
    # res['type'] = 'rect'
    # res['fps'] = fps
    # json.dump(res, open(result_path, 'w'), indent=2)
