import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class lossFunction(nn.Module):
    def __init__(self,S,B,l_coord,l_noobj):
        super(lossFunction,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        N = box1.size(0)
        M = box2.size(0)
        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),
            box2[:,:2].unsqueeze(0).expand(N,M,2),
        )
        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),
            box2[:,2:].unsqueeze(0).expand(N,M,2),
        )
        wh = rb - lt 
        wh[wh<0] = 0 
        inter = wh[:,:,0] * wh[:,:,1]
        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])
        area1 = area1.unsqueeze(1).expand_as(inter)
        area2 = area2.unsqueeze(0).expand_as(inter)
        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self,pred_tensor,target_tensor):
        N = pred_tensor.size()[0]
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)
        coo_pred = pred_tensor[coo_mask].view(-1,30)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5)
        class_pred = coo_pred[:,10:]
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)
        noo_pred_mask = torch.cuda.BoolTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
        noo_pred_c = noo_pred[noo_pred_mask]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,reduction='sum')

        #compute contain obj loss
        coo_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.cuda.BoolTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size()).cuda()
        for i in range(0,box_target.size()[0],2):
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2]/14. -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2]/14. +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2]/14. -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2]/14. +0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4])
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda()
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1
            box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        box_target_iou = Variable(box_target_iou).cuda()

        # response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],reduction='sum')
        loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],reduction='sum') + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),reduction='sum')
        
        # not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:,4]= 0
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],reduction='sum')
        class_loss = F.mse_loss(class_pred,class_target,reduction='sum')
        
        return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N