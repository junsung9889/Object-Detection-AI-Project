import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
import numpy as np
from utils_2016312160.Model import resnet50
from utils_2016312160.Loss import lossFunction
from utils_2016312160.Dataset import Dataset

use_gpu = torch.cuda.is_available()
learning_rate = 0.001
num_epochs = 40
batch_size = 2
net = resnet50()
resnet = models.resnet50(pretrained=True)
resnet_dic = resnet.state_dict()
dic = net.state_dict()
for k in resnet_dic.keys():
    if k in dic.keys() and not k.startswith('fc'):
        dic[k] = resnet_dic[k]
net.load_state_dict(dic)

if use_gpu:
     net.cuda()
net.train()

params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
        
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
criterion = lossFunction(7,2,5,0.5)
train_dataset = Dataset(root='./VOC2012/TrainImages/',data_file='./voc2012_train.txt',train=True,transform = [transforms.ToTensor()] )
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
valid_dataset = Dataset(root='./VOC2012/ValidImages/',data_file='./voc2012_valid.txt',train=False,transform = [transforms.ToTensor()] )
valid_loader = DataLoader(valid_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

num_iter = 0
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate=0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.0
    
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 1000 == 0:
            print ('Iter: [%d/%d] Loss: %.4f, Average: %.4f' 
            %(i+1, len(train_loader), loss.item(), total_loss / (i+1)))
            num_iter += 1

    validation_loss = 0.0
    net.eval()
    for i,(images,target) in enumerate(valid_loader):
        with torch.no_grad():
            images = Variable(images)
            target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        validation_loss += loss.item()
    validation_loss /= len(valid_loader)

    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        print('Current best test loss %.4f' % best_test_loss)
        torch.save(net.state_dict(),'model_2016312160.pth')