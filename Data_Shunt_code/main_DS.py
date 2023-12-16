import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import DS
import random
from time import time
from torch import nn
import clip
import torch.nn.functional as F
from loss import SCELoss
def random_seed_setup(seed:int=None):
    torch.backends.cudnn.enabled = True
    if seed:
        print('Set random seed as',seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.backends.cudnn.benchmark = True

def main(config):
    random_seed_setup(1)
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader',module_data)
    test_data_loader, valid_data_loader, classes_word, pre_test_dataset, pre_data_loader   = data_loader.split_validation()

    print(len(data_loader),len(test_data_loader),len(valid_data_loader))
    model_samll = torch.load("./save/Small_model_.pth").cuda()
    model_large, preprocess = clip.load("ViT-B/32",jit=False) #loading the CLIP model based on ViT
    model_large = model_large.cuda()
    # criterion = config.init_obj('loss',module_loss, cls_num_list=data_loader.cls_num_list)
    # criterion = nn.KLDivLoss(reduction="batchmean")
    criterion = F.cross_entropy
    # criterion = SCELoss()
    optimizer = torch.optim.Adam(model_samll.parameters(), lr=1e-4)
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # optimizer = config.init_obj('optimizer',torch.optim,model_samll.parameters())



    trainer = DS(
        preprocess,
        model_samll                                  ,
        model_large                                  ,
        after_train_loader = data_loader             , 
        data_loader         = valid_data_loader      ,
        test_data_loader  = test_data_loader    ,
        classes_word = classes_word             ,
        pre_data_loader =  pre_data_loader      ,
        pre_test_dataset = pre_test_dataset     ,
        criterion = criterion                   ,
        opt = optimizer                         ,
        config = config                         ,
        # lr_scheduler = lr_scheduler
    )

    trainer.inference()



if __name__=='__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c','--config',default=None,type=str,help='config file path (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs','flags type target')
    options = [
        CustomArgs(['--name'],type=str,target='name'),
        CustomArgs(['--save_period'],type=int,target='trainer;save_period'),
        CustomArgs(['--distribution_aware_diversity_factor'],type=float,target='loss;args;additional_diversity_factor'),
        CustomArgs(['--pos_weight'],type=float,target='arch;args;pos_weight'),
        CustomArgs(['--collaborative_loss'],type=int,target='loss;args;collaborative_loss'),
    ]
    config = ConfigParser.from_args(args,options)


    main(config)
 
