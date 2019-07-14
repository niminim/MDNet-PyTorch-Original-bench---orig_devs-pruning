# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torch
import copy

# seq_home = '../dataset/'
usr_home = os.path.expanduser('~')
#if OS == 'Windows':
    # usr_home = 'C:/Users/smush/'
#    seq_home = os.path.join(usr_home, 'downloads/')
#elif OS == 'Linux':
#    # usr_home = '~/'
#    #seq_home = os.path.join(usr_home, 'MDNet-data/')
#    seq_home = os.path.join(usr_home, '/Downloads/datasets/')
#else:
#    sys.exit("aa! errors!")
seq_home = os.path.join(usr_home, '/Downloads/datasets/')

seq_home = 'datasets'

# benchmark_dataset = 'VOT/vot2016'
benchmark_dataset = 'OTB'
seq_home = os.path.join(seq_home, benchmark_dataset)

seq2 = 'home/nimrod/Downloads/datasets'
seq2 = os.path.join(seq2, benchmark_dataset)

db = 'DragonBaby'
seq2 = os.path.join(seq2, db)


a = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
a2 = torch.tensor([[20,30,40], [50,60,70], [80,90,100]])
b = torch.tensor([[10,11,12], [13,14,15]])

print('a:')
print(a)
print('a.shape: ', a.shape)
print('b:')
print(b)
print('b.shape: ', b.shape)

c = torch.cat((a,b))
print('c:')
print(c)
print('c.shape: ', c.shape)

c2 = torch.cat((a,a2),1)
print('c2:')
print(c2)
print('c2.shape: ', c2.shape)


w = c2[:,:] # copy the reference
w2 = copy.copy(c2) # copy the reference
w3 = copy.deepcopy(c2) # copy the value
print('w (using :,: -')
print(w)
print('w2 (using copy.copy:')
print(w2)
print('w3 (using copy.deepcopy):')
print(w3)

print('')
print('Now change c2:')
c2[1,1] = 15
print('c2:')
print(c2)
print('w')
print(w)
print('w (using :,: -')
print(w)
print('w2 (using copy.copy):')
print(w2)
print('w3 (using copy.deepcopy):')
print(w3)