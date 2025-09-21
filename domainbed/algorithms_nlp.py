# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

# import copy
# import numpy as np
# from collections import OrderedDict
# try:
#     from backpack import backpack, extend
#     from backpack.extensions import BatchGrad
# except:
#     backpack = None

#from domainbed import networks
from transformers import AutoModelForSequenceClassification
# from domainbed.lib.misc import (
#     random_pairs_of_minibatches, split_meta_train_test, ParamDict,
#     MovingAverage, ErmPlusPlusMovingAvg, l2_between_dicts, proj, Nonparametric,
#             LARS,  SupConLossLambda
#     )


ALGORITHMS = [
    'ERM',
    'ERMPlusPlus',
    # Added ---
    'StudentSingle',
    'StudentSingleHard',
    # ----
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, featurizer = None):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if featurizer:
            self.featurizer = featurizer
            try:
                n_outputs = featurizer.n_outputs
            except:
                print('No n_output attribute for featurizer, set to 2048 by default')
                n_outputs = 2048
        else:
            self.featurizer = networks.Featurizer(input_shape, self.hparams)
            if 'd_s' in hparams and hparams['d_s']:
                self.featurizer = networks.FeaturizerJLT(self.featurizer,self.hparams)
            n_outputs = self.featurizer.n_outputs

        self.classifier = networks.Classifier(
                n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        #self.network = network
        
        if 'SGD' in hparams and hparams['SGD']:
            if "momentum" not in self.hparams:
                self.hparams["momentum"] = 0.0
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.hparams["lr"],
                momentum=self.hparams["momentum"],
                weight_decay=self.hparams['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
            )

    def update(self, batch, unlabeled=None):
        self.optimizer.zero_grad()

        #all_x = torch.cat([x for x, y in minibatches])
        #all_y = torch.cat([y for x, y in minibatches])

        loss = F.cross_entropy(self.predict(all_x), all_y)

        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x, get_pool=False):
        if get_pool:
            pool = self.featurizer(x)
            y = self.classifier(pool)
            return pool,y
        else:
            return self.network(x)
    
    def freeze(self):
        self.featurizer.eval()
        for param in self.featurizer.parameters():
            param.requires_grad = False
    
def padding(l):
    l2 = [torch.Tensor(i) for i in l]
    return pad_sequence(l2,batch_first=True,padding_value=0)

class Student(ERM):
    def __init__(self, input_shape, num_classes, num_domains, all_data, test_env, hparams, featurizer = None, domain_len = None):
        super().__init__(input_shape, num_classes, num_domains, hparams, featurizer=featurizer)
        if 'freeze' in hparams and hparams['freeze']:
            self.freeze()
        if domain_len:
            if len(domain_len) != (num_domains+1):
                raise ValueError("domain number doesn't match the length of domain_len")
            domain_len = torch.Tensor(domain_len)
            domain_len /= torch.sum(domain_len)
            domain_len = torch.log(domain_len)
        else:
            domain_len = torch.zeros(num_domains+1)
        self.T = self.hparams['T'] if 'T' in self.hparams else 1
        self.dc_temperature = self.hparams['dc_temperature'] if 'dc_temperature' in self.hparams else 3
        self.tc_temperature = self.hparams['tc_temperature'] if 'tc_temperature' in self.hparams else 3
        self.loss_temperature = self.hparams['loss_temperature'] if 'loss_temperature' in self.hparams else 1
        #self.all_data = all_data
        self.target_domain = test_env
        teachers = padding([[data['tc'] for data in all_data[j]] for j in range(len(all_data))])
        domain_classifier = padding([[data['dc'] for data in all_data[j]] for j in range(len(all_data))])
        self.register_buffer('teachers',teachers,persistent=False)
        self.register_buffer('domain_classifier',domain_classifier,persistent=False)
        self.register_buffer('domain_len',domain_len,persistent=False)

    def update(self, minibatches=None, unlabeled=None):
        return
        
    
    def calculate_kd_loss(self, all_x, all_i, reduction='mean'):
        return 
    
    def teachers_outputs(self,all_i):
        return
    
    def drop_column(self, tensor, col_index):
        # Ensure col_index is valid
        if col_index < 0 or col_index >= tensor.size(1):
            raise IndexError("Column index out of range.")
        
        # Slice the tensor to exclude the specified column
        if col_index == 0:
            # Drop the first column
            result = tensor[:, 1:]
        elif col_index == tensor.size(1) - 1:
            # Drop the last column
            result = tensor[:, :-1]
        else:
            # Drop an arbitrary column
            left = tensor[:, :col_index]
            right = tensor[:, col_index+1:]
            result = torch.cat((left, right), dim=1)
        
        return result
    
    def loss_function(self, student_logits, soft_targets, reduction = 'mean'):
        cross_entropy = self.hparams['cross_entropy'] if 'cross_entropy' in self.hparams else 0
        if cross_entropy:
            return F.cross_entropy(student_logits, soft_targets, reduction=reduction)
        #Soften the student logits by applying softmax first and log() second
        
        soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = soft_targets * (soft_targets.log() - soft_prob)
        class_dim = student_logits.dim() - 1
        C = student_logits.size(class_dim)
        soft_targets_loss = soft_targets_loss.sum(dim=class_dim) / C *(self.T**2)
        if reduction != 'none':
            soft_targets_loss = torch.sum(soft_targets_loss)
        if reduction == 'mean':
            soft_targets_loss = soft_targets_loss / soft_prob.size()[0]
        return soft_targets_loss

class StudentSingle(Student):
    def __init__(self, input_shape, num_classes, num_domains, all_data, test_env, hparams, featurizer = None):
        ERM.__init__(self, input_shape, num_classes, num_domains, hparams, featurizer=featurizer)
        if 'freeze' in hparams and hparams['freeze']:
            self.freeze()
        self.tc_temperature = self.hparams['tc_temperature'] if 'tc_temperature' in self.hparams else 3
        self.loss_temperature = self.hparams['loss_temperature'] if 'loss_temperature' in self.hparams else 1
        self.T = self.hparams['T'] if 'T' in self.hparams else 1
        #self.all_data = all_data
        self.target_domain = test_env
        teacher = torch.Tensor([data for data in all_data[1]])
        self.register_buffer('teacher',teacher,persistent=False)
    
    def update(self, minibatches=None, unlabeled=None):
        total_loss = 0
        result = dict()
        #rkd = 'rkd' in self.hparams and self.hparams['rkd']

        if unlabeled:
            all_x = torch.cat([x for x,i in unlabeled])
            all_i = torch.cat([i for x,i in unlabeled]).int()
            loss2 = self.calculate_kd_loss(all_x,all_i,reduction='mean')
            total_loss += loss2
            result['loss'] = loss2.item()
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        #result['total_loss']=total_loss.item()
        return result
    
    def calculate_kd_loss(self, all_x, all_i, reduction='mean'):
        student_logits = self.predict(all_x)
        teacher_prob = self.teachers_outputs(all_i)
        return self.loss_function(student_logits,teacher_prob,reduction=reduction)
    
    def teachers_outputs(self,all_i):
        outputs = self.teacher[all_i]/self.tc_temperature
        outputs = F.softmax(outputs,dim=-1)
        return outputs
    
class StudentSingleHard(StudentSingle):
    def __init__(self, input_shape, num_classes, num_domains, all_data, test_env, hparams, featurizer=None):
        super().__init__(input_shape, num_classes, num_domains, all_data, test_env, hparams, featurizer)

    def calculate_kd_loss(self, all_x, all_i, reduction='mean'):
        student_logits = self.predict(all_x)
        teacher_prob = self.teachers_outputs(all_i)
        weak_labels = torch.argmax(teacher_prob,dim=1)
        return F.cross_entropy(student_logits, weak_labels, reduction=reduction)
    