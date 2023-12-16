import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
from .imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import clip
_, preprocess = clip.load("RN50",jit=False)
class ImbalanceCIFAR100DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,training=True,retain_epoch_size=True):
        normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32,padding=4) ,
            transforms.RandomHorizontalFlip()   ,
            transforms.RandomRotation(15)       ,
            transforms.ToTensor()               ,
            normalize
        ])
        test_trsfm = transforms.Compose([transforms.ToTensor(),normalize])

        if training:
            self.dataset = IMBALANCECIFAR100(data_dir,train=True,download=True,transform=preprocess)
            self.val_dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)#preprocess
            # self.preprocess_dataset = IMBALANCECIFAR100(data_dir,train=True,download=True,transform=train_trsfm)
            self.preprocess_val_dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=preprocess)#preprocess

        else:
            self.dataset = datasets.CIFAR100(data_dir,train=False,download=True,transform=test_trsfm)
            self.val_dataset = None

        num_classes = max(self.dataset.targets)+1
        assert num_classes == 100

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size    ,
            'shuffle'       : True          ,
            'num_workers'   : num_workers   ,
            'drop_last'     : False
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs,sampler=None)

    def split_validation(self,type='test'):

        # index_val = np.random.choice(np.arange(len(self.val_dataset)), 2000, replace=False).astype(int)
        # index_test = np.array([i for i in np.arange(len(self.val_dataset)) if i not in index_val]).astype(int)
        # print(index_test)
        # test_dataset = self.val_dataset[index_test]
        # val_dataset = self.val_dataset[index_val]
        test_dataset, val_dataset = torch.utils.data.random_split(self.val_dataset, [8000,2000],generator=torch.Generator().manual_seed(0))
        pre_test_dataset, pre_val_loader   = torch.utils.data.random_split(self.preprocess_val_dataset, [8000,2000],generator=torch.Generator().manual_seed(0))
        batch_size_  = 1
        test_dataset = DataLoader(
            dataset     = test_dataset ,
            batch_size  = batch_size_                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        val_dataset =  DataLoader(
            dataset     = val_dataset ,
            batch_size  = batch_size_                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        pre_test_dataset =  DataLoader(
            dataset     = pre_test_dataset                                      ,
            batch_size  = batch_size_                                                     ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        pre_val_loader  =  DataLoader(
            dataset     = pre_val_loader                                        ,
            batch_size  = batch_size_                                                     ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        classes_word = self.val_dataset.classes
        return test_dataset, val_dataset, classes_word, pre_test_dataset, pre_val_loader 

    # def split_validation(self,type='test'):

    #     test_dataset =  DataLoader(
    #         dataset     = self.val_dataset                                      ,
    #         batch_size  = 128                                                   ,
    #         shuffle     = False                                                 ,
    #         num_workers = 2                                                     ,
    #         drop_last   = False
    #     )
    #     classes_word = self.val_dataset.classes
    #     return test_dataset, 1, classes_word

class ImbalanceCIFAR10DataLoader(DataLoader):
    def __init__(self,data_dir,batch_size,num_workers,training=True,retain_epoch_size=True):
        normalize = transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])
        train_trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()   ,
            transforms.RandomRotation(15)       ,
            transforms.ToTensor()               ,
            normalize
        ])
        test_trsfm = transforms.Compose([transforms.ToTensor(),normalize])

        if training:
            self.dataset = IMBALANCECIFAR10(data_dir,train=True,download=True,transform=train_trsfm)
            self.val_dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=test_trsfm)
            self.preprocess_val_dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=preprocess)#preprocess
    
        else:
            self.dataset = datasets.CIFAR10(data_dir,train=False,download=True,transform=test_trsfm)
            self.val_dataset = None

        # Uncomment to use OOD datasets
        self.OOD_dataset = None
        # self.OOD_dataset = datasets.SVHN(data_dir,split="test",download=True,transform=test_trsfm)
        # self.OOD_dataset = LT_Dataset('../ImageNet_LT/ImageNet_LT_open','../ImageNet_LT/ImageNet_LT_open.txt',train_trsfm)
        # self.OOD_dataset = LT_Dataset('../Places_LT/Places_LT_open','../Places_LT/Places_LT_open.txt',train_trsfm)

        num_classes = max(self.dataset.targets)+1
        assert num_classes == 10

        self.cls_num_list = np.histogram(self.dataset.targets,bins=num_classes)[0].tolist()

        self.init_kwargs = {
            'batch_size'    : batch_size    ,
            'shuffle'       : True          ,
            'num_workers'   : num_workers   ,
            'drop_last'     : False
        }
        super().__init__(dataset=self.dataset,**self.init_kwargs,sampler=None)



    def split_validation(self,type='test'):

        # index_val = np.random.choice(np.arange(len(self.val_dataset)), 2000, replace=False).astype(int)
        # index_test = np.array([i for i in np.arange(len(self.val_dataset)) if i not in index_val]).astype(int)
        # print(index_test)
        # test_dataset = self.val_dataset[index_test]
        # val_dataset = self.val_dataset[index_val]
        test_dataset, val_dataset = torch.utils.data.random_split(self.val_dataset, [int(0.8 * len(self.val_dataset)),int(0.2* len(self.val_dataset))] ,generator=torch.Generator().manual_seed(0))
        pre_test_dataset, pre_val_loader   = torch.utils.data.random_split(self.preprocess_val_dataset, [int(0.8 * len(self.val_dataset)),int(0.2* len(self.val_dataset))],generator=torch.Generator().manual_seed(0))
        batch_size_  = 1
        test_dataset = DataLoader(
            dataset     = test_dataset ,
            batch_size  = batch_size_                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        val_dataset =  DataLoader(
            dataset     = val_dataset ,
            batch_size  = batch_size_                                                  ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        pre_test_dataset =  DataLoader(
            dataset     = pre_test_dataset                                      ,
            batch_size  = batch_size_                                                     ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        pre_val_loader  =  DataLoader(
            dataset     = pre_val_loader                                        ,
            batch_size  = batch_size_                                                     ,
            shuffle     = False                                                 ,
            num_workers = 2                                                     ,
            drop_last   = False
        )
        classes_word = self.val_dataset.classes
        return test_dataset, val_dataset, classes_word, pre_test_dataset, pre_val_loader 
