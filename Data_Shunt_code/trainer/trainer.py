import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker,load_state_dict,rename_parallel_state_dict,autocast
import model.model as module_arch
from model.metric import *
from tqdm import tqdm
from torch import nn
import clip
from PIL import Image
from torchvision import datasets, transforms
torch.multiprocessing.set_sharing_strategy('file_system')



class DS:
    def __init__(self,preprocess,model_samll,model_large,after_train_loader,data_loader,test_data_loader,classes_word,pre_data_loader, pre_test_dataset,criterion,opt,config,lr_scheduler=None):
        self.model_samll = model_samll
        self.model_large = model_large
        self.device = torch.device('cuda:0')
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.pre_data_loader = pre_data_loader
        self.pre_test_dataset = pre_test_dataset
        # self.val_targets = torch.tensor(test_data_loader.dataset.targets,device=self.device).long()
        self.num_class = 100#self.val_targets.max().item()+1
        self.test_trsfm = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])])
        self.classes_word = classes_word
        self.preprocess = preprocess
        self.criterion = criterion
        self.opt = opt
        self.lr_scheduler = lr_scheduler
        self.after_train_loader = after_train_loader
        self.model_samll_o = torch.load("./save/Small_model.pth").cuda()
  


    def inference(self):
        self.model_samll.eval(), self.model_large.eval(), self.model_samll_o.eval()
        output = torch.empty(0,self.num_class,dtype=torch.float32,device=self.device)
        # uncertainty = torch.empty(0,dtype=torch.float32,device=self.device) # 
        val_targets = []

        # for k,item in enumerate(self.classes_word):
        #     if k < 3:
        #         self.classes_word[k] = "None"


        # text_descriptions = [f"This is a photo of a {label}" for label in self.classes_word] #self.test_data_loader.dataset.classes
        # text_tokens_ensembled = clip.tokenize(text_descriptions).cuda()
        # with torch.no_grad():
        #     text_features_ensembled = self.model_large.encode_text(text_tokens_ensembled).float()
        #     text_features_ensembled /= text_features_ensembled.norm(dim=-1, keepdim=True)
        large_count = 0
        for data_1,data_2 in zip(self.test_data_loader,self.pre_test_dataset):
            data,target = data_1
            pre_data,_ = data_2
    
            data = data.to(self.device)
            # image_samll = self.test_trsfm(data)
            val_targets.extend(target)
            with torch.no_grad():
                o = self.model_samll_o(data).softmax(dim=-1)
                # pred = torch.argmax(o,dim=1)
                # o_r = o
                

                if torch.max(o) < 0.99999: 
                    # o_r[0][:10] = 0
                    # o = self.model_samll(data).softmax(dim=-1)
                    temp = self.model_samll(data)
                    temp[0][:10] = 0
                    o = temp.softmax(dim=-1)
                    if torch.max(o) < 0.99999:
                        large_count += 1
                        text_descriptions = [f"This is a photo of a {label} with probability {p}" for label,p in zip(self.classes_word,o[0])] #self.test_data_loader.dataset.classes
                        text_tokens_ensembled = clip.tokenize(text_descriptions).cuda()
                        with torch.no_grad():
                            text_features_ensembled = self.model_large.encode_text(text_tokens_ensembled).float()
                            text_features_ensembled /= text_features_ensembled.norm(dim=-1, keepdim=True)

                        # data = Image.fromarray(np.uint8(data[0].cpu()).transpose(1,2,0))
                        # image = self.preprocess(data)
                        # pre_data = pre_data.unsqueeze(0)
                        
                        image_features = self.model_large.encode_image(pre_data.cuda()).float()
                        o = ( 100.0 * image_features @ text_features_ensembled.T).softmax(dim=-1)
                    
                # else:
                #     o = self.model_samll(data).softmax(dim=-1)
            
            output = torch.cat([output,o.detach()],dim=0)
            # uncertainty = torch.cat([uncertainty,u.detach()],dim=0)
        print(len(val_targets),large_count)
        val_targets = torch.tensor(val_targets,device=self.device).long()
        ACC(output,val_targets,0,region_len=self.num_class/3)



class Small_model_inference:
    def __init__(self,model,data_loader,test_data_loader):
        self.model = model
        self.device = torch.device('cuda:0')
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        # self.val_targets = torch.tensor(test_data_loader.dataset.targets,device=self.device).long()
        self.num_class = 100#self.val_targets.max().item()+1




    def inference(self):
        self.model.eval()
        output = torch.empty(0,self.num_class,dtype=torch.float32,device=self.device)
        # uncertainty = torch.empty(0,dtype=torch.float32,device=self.device) # 
        val_targets = []
        for _,(data,target) in enumerate(self.test_data_loader):
            data = data.to(self.device)
            val_targets.extend(target)
            with torch.no_grad():
                o = self.model(data)
                # u = self.model.backbone.w[-1]
            output = torch.cat([output,o.detach()],dim=0)
            # uncertainty = torch.cat([uncertainty,u.detach()],dim=0)
        print(len(val_targets))
        val_targets = torch.tensor(val_targets,device=self.device).long()
        ACC(output,val_targets,0,region_len=self.num_class/3)
  






class CLIP_inference(BaseTrainer):
    def __init__(self,preprocess,model,data_loader,test_data_loader, classes_word):
        self.model = model
        self.device = torch.device('cuda:0')
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.test_data_loader = test_data_loader
        # self.val_targets = torch.tensor(test_data_loader.dataset.targets,device=self.device).long()
        self.num_class = 100#self.val_targets.max().item()+1
        self.preprocess = preprocess
        self.classes_word = classes_word




    def inference(self):
        self.model.eval()
        output = torch.empty(0,self.num_class,dtype=torch.float32,device=self.device)


        # text_descriptions = [f"This is a photo of a {label}" for label in self.classes_word] #self.test_data_loader.dataset.classes
        # text_tokens_ensembled = clip.tokenize(text_descriptions).cuda()
        # with torch.no_grad():
        #     text_features_ensembled = self.model.encode_text(text_tokens_ensembled).float()
        #     text_features_ensembled /= text_features_ensembled.norm(dim=-1, keepdim=True)
        val_targets = []
        for _,(data,target) in enumerate(self.test_data_loader):
            image = data.to(self.device)
            val_targets.extend(target)
            # data = Image.fromarray(np.uint8(data.transpose(1,2,0) ))
            # data = np.uint8(data.transpose(1,2,0))
            # image = self.preprocess(image).unsqueeze(0)
            with torch.no_grad():
                image_features = self.model.encode_image(image).float()
     

            text_descriptions = [f"This is a photo of a {label}" for label in self.classes_word] #self.test_data_loader.dataset.classes
            text_tokens_ensembled = clip.tokenize(text_descriptions).cuda()
            with torch.no_grad():
                text_features_ensembled = self.model.encode_text(text_tokens_ensembled).float()
                text_features_ensembled /= text_features_ensembled.norm(dim=-1, keepdim=True)

            #the 100.0 works as temperature parameter, raising the softmax confidence 
            text_probs = ( 100.0 * image_features @ text_features_ensembled.T).softmax(dim=-1)

            # top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)



            output = torch.cat([output,text_probs.detach()],dim=0)
 
        val_targets = torch.tensor(val_targets,device=self.device).long()

        ACC(output,val_targets,0,region_len=self.num_class/3)





