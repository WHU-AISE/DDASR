import json
import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable

def read_corpus(file_path):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        data.append(sent)
    return data


def load_dict(file_path):
    return json.load(open(file_path))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0., 0.5 * (1. + math.cos(math.pi * float(num_cycles) * 2. * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_PL_dist(scores, other_scores_tail):
    above = torch.sum(scores)
    below = flip(scores)
    below = torch.exp(below)
    below = torch.cumsum(below, 0)
    below = torch.log(below + other_scores_tail)
    below_sum = torch.sum(below)

    log_pl = above - below_sum
    return log_pl

def flip(scores):
    scores_ncol = scores.size()[0]

    idx = [i for i in range(scores_ncol - 1, -1, -1)]
    idx = Variable(torch.LongTensor(idx))
    if scores.is_cuda: idx = idx.cuda()

    scores = scores.index_select(0, idx)
    return scores

def index2str(list, vocab):
    list = [i for i in list if i>0]
    result_str = ''
    for item in list:
        result_str += vocab[item]+' '
    return result_str

def index2strlist(list, vocab):
    sentences = []
    for item in list:
        item = [i for i in item if i>0]
        result = [vocab[i] for i in item if i > 0]
        sentences.append(' '.join(result))
    return sentences

def index2apirefstr(list, longtail, vocab):
    apilist = []
    for i in range(len(list)):
        for item in longtail[i]:
            if item > 0:
                apilist.append(item)
        apilist.append(list[i])

    apilist = [vocab[i] for i in apilist]
    return ' '.join(apilist)

def getWeight():
    weight_list = []
    weight_idf = json.load(open('./data/api_id_idf.json'))
    for keys in weight_idf.keys():
        weight_list.append([weight_idf[keys]])
    return weight_list

def getWeight1():
    weight_list = []
    weight_idf = json.load(open('./data/api_id_idf.json'))
    for keys in weight_idf.keys():
        weight_list.append([1])
    return weight_list
if __name__ == '__main__':
    index2str([1,3,2,8,0,0,0,0],'ww')
