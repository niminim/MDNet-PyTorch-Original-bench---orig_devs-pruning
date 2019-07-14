import os
import scipy.io
import numpy as np
from collections import OrderedDict
from modules.utils import overlap_ratio  ###


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer(model, lr_base, lr_mult, train_all=False, momentum=0.9, w_decay=0.0005):
    if train_all:
        params = model.get_all_params()
    else:
        params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1, use_gpu=True):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(256 * 3 * 3, 512),
                                        nn.ReLU(inplace=True))),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])
    
        #print('The Net: ')
        #for k in self.layers.state_dict(): print("Module Layer", k)

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in self.branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_{:d}'.format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params
    
    def get_all_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        # forward model from in_layer to out_layer
        run = False
        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                x = module(x)
                if name == 'conv3':
                    x = x.view(x.size(0), -1)
                if name == out_layer:
                    return x
                

        x = self.branches[k](x)
        if out_layer=='fc6':
            return x
        elif out_layer=='fc6_softmax':
            return F.softmax(x, dim=1)

    def load_model(self, model_path):
        #states = torch.load(model_path)
        #shared_layers = states['shared_layers']
        #print('D: ')
        #print(shared_layers.keys())
        #self.layers.load_state_dict(shared_layers)

        model_weights = torch.load(model_path) ###
        d = model_weights['shared_layers'] ###
        #print('model weights ')
        #for k in model_weights: print(k)
        #print('D: ')
        #print(d.keys())
        
        
        d['conv3.0.weight'] = torch.randn((256, 256,3,3)) * 0.01 ###
        d['conv3.0.bias'] = torch.zeros(256) ###                
        d['fc4.0.weight'] = torch.randn((512, 256*3*3)) * 0.01 ###
        d['fc4.0.bias'] = torch.zeros(512) ###
        d['fc5.1.weight'] = torch.randn((512, 512)) * 0.01 ###
        d['fc5.1.bias'] = torch.zeros(512) ###
        self.layers.load_state_dict(d) ###

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


### calc loss between target bb and reult bb   ###     
class IouLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(IouLoss, self).__init__()
        self.use_gpu = use_gpu
        
    def forward(self, result_bb, gt):   
        iou_loss = (1.0 - overlap_ratio(result_bb, gt)[0]) * 10
        iou_loss_tensor = torch.tensor(iou_loss).type(torch.FloatTensor)
        if self.use_gpu:
            iou_loss_tensor = iou_loss_tensor.cuda()
        #print('iou_loss: ', iou_loss_tensor)
        return iou_loss_tensor
    
  
# calc loss between target bb and reult bb   using IoU and localization of centers  ###     
class IouLocalizationLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(IouLocalizationLoss, self).__init__()
        
    def forward(self, result_bb, gt):
        # calc iou loss
        iou = overlap_ratio(result_bb, gt)[0]
        iou_loss = 1.0 - iou
        
        # calc distance between centers
        cx_pred = result_bb[0] + result_bb[2]/2
        cy_pred = result_bb[1] + result_bb[3]/2
        cx_gt = gt[0] + gt[2]/2
        cy_gt = gt[1] + gt[3]/2
        distance = math.sqrt( ((cx_pred - cx_gt)**2) + ((cy_pred - cy_gt)**2) ) # distance between (x1_c,y1_c) and (x2_c,y2_c)
        
        iou_localization_loss = iou_loss*10.0 + distance/2.0 
                
        iou_localization_loss_tensor = torch.tensor(iou_localization_loss).type(torch.FloatTensor)

        #if opts['use_gpu']:
        iou_localization_loss_tensor = iou_localization_loss_tensor.cuda()
        #print('iou_localization_loss: ', iou_localization_loss_tensor)
        return iou_localization_loss_tensor
    
    
####
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)  # alone row [#batch_size, 1]

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()