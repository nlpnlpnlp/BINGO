import torch
import torch.nn.functional as F
from metric import get_sparsity_loss, get_continuity_loss
import numpy as np
import math
from tqdm import tqdm
import sys

def train_sp_norm(model, optimizer, dataset, device, args,writer_epoch,grad,grad_loss):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    train_sp = []
    batch_len=len(dataset)
    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()

        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)

        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(
            rationales[:, :, 1], masks, args.sparsity_percentage)

        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
        train_sp.append(
            (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item())

        continuity_loss = args.continuity_lambda * get_continuity_loss(
            rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient


        if args.dis_lr==1:
            if sparsity==0:
                lr_lambda=1
            else:
                lr_lambda=sparsity
            if lr_lambda<0.05:
                lr_lambda=0.05
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda
        elif args.dis_lr == 0:
            pass
        else:
            optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] / args.dis_lr

        loss.backward()





        optimizer.step()


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

def get_grad(model,dataloader,p,use_rat,device):
    data=0
    # device=model.device()
    model.train()
    grad=[]
    for batch,d in enumerate(dataloader):
        data=d
        inputs, masks, labels = data
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        rationale,logit,embedding2,cls_embed=model.grad(inputs, masks)
        loss=torch.mean(torch.softmax(logit,dim=-1)[:,1])
        cls_embed.retain_grad()
        loss.backward()
        if use_rat==0:
            k_mask=masks
        elif use_rat==1:
            k_mask=rationale[:,:,1]
        masked_grad=cls_embed.grad*k_mask.unsqueeze(-1)
        gradtemp=torch.sum(abs(masked_grad),dim=1)       
        gradtemp=gradtemp/torch.sum(k_mask,dim=-1).unsqueeze(-1)      
        gradtempmask = gradtemp
        norm_grad=torch.linalg.norm(gradtempmask, ord=p, dim=1)           
        grad.append(norm_grad.clone().detach().cpu())
    grad=torch.cat(grad,dim=0)
    tem=[]
    for g in grad:
        if math.isnan(g.item()):
            continue
        else:
            tem.append(g)

    tem=torch.tensor(tem)
    maxg=torch.max(tem)*1000
    meang=torch.mean(tem)*1000
    return maxg,meang

def train_rnp(model, optimizer, dataset, device, args,epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    sparsity_item = 0
    total_loss = 0


    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()

        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])
        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient
        sparsity_item += sparsity

        if(args.optimizer=="Adam"):
            loss.backward()
            optimizer.step()
        elif(args.optimizer=="AdamW"):
            loss.backward()
            optimizer.step()
        elif(args.optimizer=="MTAdam"  or args.optimizer=="MTAdamW"):
            ranks = [1]*3
            optimizer.step([cls_loss,sparsity_loss,continuity_loss], ranks)
        elif(args.optimizer=="SophiaG"):
            loss.backward()
            optimizer.update_hessian()  
            optimizer.step()             
        elif(args.optimizer=="AdamWSN" or args.optimizer=="AdamWSNG"):
            loss.backward()
            optimizer.step()
        elif(args.optimizer=="MyAdam"):
            loss.backward()
            optimizer.step()

        elif(args.optimizer=="CAGrad"):
            loss1 = cls_loss
            loss2 = sparsity_loss
            loss3 = continuity_loss

            optimizer.backward([loss1, loss2, loss3], retain_graph=True)
            optimizer.step()
        elif(args.optimizer=="PCGrad"):

            loss1 = cls_loss
            loss2 = sparsity_loss
            loss3 = continuity_loss

            optimizer.pc_backward([loss1, loss2, loss3])
            optimizer.step()

        elif(args.optimizer=="NashAdam"):
            loss1 = cls_loss
            loss2 = sparsity_loss
            loss3 = continuity_loss
            losses = [loss1, loss2, loss3]

            optimizer.step_moo(losses)

        else:
            print("Training Exception One")
            sys.exit(1)

        
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        
        FP += ((pred == 1) & (labels == 0)).cpu().sum()



        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        total_loss += loss.item()

    print("cls_l:{} spar_l:{} cont_l:{},sparsity_item:{}".format(cls_l,spar_l,cont_l,sparsity_item))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    avg_loss = total_loss / len(dataset)

    train_loss = [total_loss,cls_l,spar_l,cont_l,sparsity_item,avg_loss]



    return train_loss, precision, recall, f1_score, accuracy

def train_rnp_bingo(model, optimizer, dataset, device, args,epoch,para):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    sparsity_item = 0
    total_loss = 0


    for (batch, (inputs, masks, labels)) in enumerate(dataset):

        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()

        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient

        sparsity_item += sparsity

        if(args.optimizer=="Adam"):
            loss.backward()
            optimizer.step()
        elif(args.optimizer=="BINGO"):
            loss1 = cls_loss
            loss2 = sparsity_loss
            loss3 = continuity_loss
            optimizer.pc_backward([loss1, loss2, loss3])
            optimizer.step()
        else:
            print("Training Exception One")
            sys.exit(1)



        
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        
        FP += ((pred == 1) & (labels == 0)).cpu().sum()



        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        total_loss += loss.item()

    print("cls_l:{} spar_l:{} cont_l:{},sparsity_item:{}".format(cls_l,spar_l,cont_l,sparsity_item))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    avg_loss = total_loss / len(dataset)

    train_loss = [total_loss,cls_l,spar_l,cont_l,sparsity_item,avg_loss]



    return train_loss, precision, recall, f1_score, accuracy

def train_rnp_adam(model, optimizer, dataset, device, args,epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    sparsity_item = 0
    total_loss = 0

    for (batch, (inputs, masks, labels)) in enumerate(tqdm(dataset)):

        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()

        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient

        sparsity_item += sparsity


        lr_lambda=1.0
        optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda

        # loss.backward()
        # optimizer.step()

        optimizer.step([cls_loss,sparsity_loss,continuity_loss])


        

        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        total_loss += loss.item()

    print("cls_l:{} spar_l:{} cont_l:{},sparsity_item:{}".format(cls_l,spar_l,cont_l,sparsity_item))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    avg_loss = total_loss / len(dataset)

    train_loss = [total_loss,cls_l,spar_l,cont_l,sparsity_item,avg_loss]

    





    return train_loss, precision, recall, f1_score, accuracy

def train_rnp_adam3(model, optimizer, dataset, device, args,epoch):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    sparsity_item = 0
    total_loss = 0

    for (batch, (inputs, masks, labels)) in enumerate(tqdm(dataset)):

        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        # rationales, cls_logits
        rationales, logits = model(inputs, masks)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
        sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()

        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])

        loss = cls_loss + sparsity_loss + continuity_loss
        # update gradient

        sparsity_item += sparsity


        lr_lambda=1.0
        optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr'] * lr_lambda

        # loss.backward()
        # optimizer.step()

        # optimizer.step([cls_loss,sparsity_loss,continuity_loss])


        ranks = [1]*3
        optimizer.step([cls_loss,sparsity_loss,continuity_loss], ranks, None)


        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        
        FP += ((pred == 1) & (labels == 0)).cpu().sum()

        cls_l += cls_loss.cpu().item()
        spar_l += sparsity_loss.cpu().item()
        cont_l += continuity_loss.cpu().item()
        total_loss += loss.item()

    print("cls_l:{} spar_l:{} cont_l:{},sparsity_item:{}".format(cls_l,spar_l,cont_l,sparsity_item))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    avg_loss = total_loss / len(dataset)

    train_loss = [total_loss,cls_l,spar_l,cont_l,sparsity_item,avg_loss]

    





    return train_loss, precision, recall, f1_score, accuracy

def train_g_skew(model, optimizer, dataset, device, args):
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for (batch, (inputs, masks, labels)) in enumerate(dataset):
        optimizer.zero_grad()
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        logits=model.g_skew(inputs,masks)[:,0,:]  # (batch_size, seq_length, 1)
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        cls_loss.backward()
        optimizer.step()
        cls_soft_logits = torch.softmax(logits, dim=-1)
        _, pred = torch.max(cls_soft_logits, dim=-1)

        
        TP += ((pred == 1) & (labels == 1)).cpu().sum()
        
        TN += ((pred == 0) & (labels == 0)).cpu().sum()
        
        FN += ((pred == 0) & (labels == 1)).cpu().sum()
        
        FP += ((pred == 1) & (labels == 0)).cpu().sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * recall * precision / (recall + precision)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return precision, recall, f1_score, accuracy

