import os
import numpy as np
import pickle
from collections import OrderedDict

seq_home = 'datasets/OTB'
seqlist_path = 'datasets/list/otb100.txt'
output_path = 'pretrain/data/otb100.pkl'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines() # otb100/Basketball,....
    
print('seq_list: ')
print(seq_list)
print('')

# python pretrain/prepro_otb100.py

# Construct db
data = OrderedDict()
for i, seq in enumerate(seq_list): # run over all sequences
    print('i: ', i)
    print('len(seq): ', len(seq))
    #print('seq.shape: ', seq.shape)
    print('seq: ', seq)

    #for p in os.listdir(os.path.join(seq_home, seq)): # list of file in datases/VOT/vot2016/bag,....
    #    print('p: ' , p) # files in directory
    #    print('split: ', os.path.splitext(p)[1])
    
    img_list = sorted([p for p in os.listdir(os.path.join(seq_home, seq, 'img')) if os.path.splitext(p)[1] == '.jpg']) # check if file format is .jpg and put in img_list    
    #img_list - a list of image file name in currect sequence, like '0001.jpg',....
    try:
        gt = np.loadtxt(os.path.join(seq_home, seq, 'groundtruth_rect.txt'), delimiter=',')
    except:
        gt = np.loadtxt(os.path.join(seq_home, seq, 'groundtruth_rect.txt'), delimiter='	')

       
    print('len(img_list): ', len(img_list))
    print('len(gt): ', len(gt))

    assert len(img_list) == len(gt), "Lengths do not match!!"

    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    img_list = [os.path.join(seq_home, seq,'img', img) for img in img_list]
    data[seq] = {'images': img_list, 'gt': gt}

# Save db
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)
with open(output_path, 'wb') as fp:
    pickle.dump(data, fp)

