import argparse
import os
import time

import torch

import matplotlib.pyplot as plt
from tqdm import tqdm

from beer import BeerData, BeerAnnotation
from hotel import HotelData,HotelAnnotation
from embedding import get_glove_embedding
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import Rnp_model
from optimizers.optimizer_mtadam import MTAdam
from optimizers.optimizer_mtadamw import MTAdamW
from optimizers.optimizer_sophia import SophiaG
from optimizers.optimizer_adamw_sn import AdamWSN 
from optimizers.optimizer_adamw_sng import AdamWSN as AdamWSNG
from optimizers.optimizer_pcgrad import PCGrad

from train_util import train_rnp
from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from metric import get_sparsity_loss, get_continuity_loss
import sys

import pandas as pd

from tensorboardX import SummaryWriter



def parse():
    #default： nonorm, data=beer, save=0
    parser = argparse.ArgumentParser(
        description="RNP")
    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
                        help='Path to the annotation')
    
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')


    # model parameters
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    
    parser.add_argument('--cls_lambda',
                        type=float,
                        default=0.9,
                        help='lambda for classification loss')

    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=1.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=1.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument('--sparsity_percentage',
                        type=float,
                        default=1.,
                        help='Regularizer to control highlight percentage [default: .2]')

    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')

    # Logging
    parser.add_argument('--writer',
                        type=str,
                        default='./noname',
                        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
                        '--results_dir',
                        type=str,
                        default='./noname',
                        help='Logging results')

    # Optimizer
    parser.add_argument('--optimizer',
                        type=str,
                        default="Adam",
                        help='Adam MTAdam MTAdamW DAdam')
    parser.add_argument('--model_name',
                        type=str,
                        default="RNP",
                        help='Adam MTAdam MTAdamW DAdam')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    

    args = parser.parse_args()
    return args




def grad_to_vec(grad,params):
    vec = []
    for p, gi in zip(params, grad):
        if gi is None:
            vec.append(torch.zeros(p.numel(), device=p.device))
        else:
            vec.append(gi.reshape(-1))
    # return torch.cat([g.reshape(-1) for g in grad])
    return torch.cat(vec)


#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)

######################
# load embedding
######################
pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
if args.data_type=='beer':       #beer
    train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)
    dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)

elif args.data_type == 'hotel':       #hotel
    args.data_dir='./data/hotel'
    args.annotation_path='./data/hotel/annotations'
    train_data = HotelData(args.data_dir, args.aspect, 'train', word2idx, balance=True)
    dev_data = HotelData(args.data_dir, args.aspect, 'dev', word2idx)
    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_data, batch_size=args.batch_size)
annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
writer=SummaryWriter(args.writer)

######################
# Training
######################
if(args.model_name=="RNP"):
    model=Rnp_model(args)
    model.to(device)

    g_para=list(map(id, model.generator.parameters()))
    p_para=filter(lambda p: id(p) not in g_para, model.parameters())
    lr2=args.lr
    lr1=args.lr
    print('lr1={},lr2={}'.format(lr1,lr2))


    if (args.optimizer=="AdamWSN"):
        para=[
            {'params': model.generator.parameters(), "sn": True},
            {'params':p_para, "sn": True}
        ]
    else:
        para=[
        {'params': model.generator.parameters(), 'lr':lr1},
        {'params':p_para,'lr':lr2}
    ]


else:
    print("Model_name: Exception One")
    sys.exit(1)

if(args.optimizer=="Adam"):
    optimizer = torch.optim.Adam(para)  
elif(args.optimizer=="AdamW"):
    optimizer = torch.optim.AdamW(para)
elif(args.optimizer=="MTAdam"):
    optimizer = MTAdam(para, lr=args.lr)
elif(args.optimizer=="SophiaG"):
    optimizer = SophiaG(para, lr=args.lr)
elif(args.optimizer=="AdamWSN"):
    optimizer = AdamWSN(para,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
    )
elif(args.optimizer=="AdamWSNG"):
    optimizer = AdamWSNG(
        para,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        subset_size=-2 
    )
elif(args.optimizer=="MTAdamW"):
    optimizer = MTAdamW(para, lr=args.lr)
elif(args.optimizer=="PCGrad"):
    optimizer = torch.optim.Adam(para, lr=args.lr)
    optimizer = PCGrad(optimizer)
else:
    print("Exception One")
    sys.exit(1)


######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]

results_logging = []


sum_train_cost_time = 0.0

for epoch in range(args.epochs):
    start = time.time()
    model.train()
    train_loss, precision, recall, f1_score, train_accuracy = train_rnp(model, optimizer, train_loader, device, args,epoch,cagrad=False)
    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    print("traning epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} train_accuracy:{:.4f}".format(epoch, recall,precision, f1_score,train_accuracy))

    sum_train_cost_time = sum_train_cost_time + (end-start)

    for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
        rationales, logits = model(inputs, masks)
        # pdb.set_trace()
        logits = torch.softmax(logits, dim=-1)

        # computer loss
        cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
        sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
        continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])

        params = [p for group in para for p in group["params"] if p.requires_grad]

        g1_cls = torch.autograd.grad(cls_loss, params, retain_graph=True, allow_unused=True)
        g2_spar = torch.autograd.grad(sparsity_loss, params, retain_graph=True, allow_unused=True)
        g3_cont = torch.autograd.grad(continuity_loss, params,retain_graph=True, allow_unused=True)

        del cls_loss, sparsity_loss, continuity_loss, g1_cls, g2_spar, g3_cont
        import gc
        gc.collect()
        torch.cuda.empty_cache()


    TP = 0
    TN = 0
    FN = 0
    FP = 0
    cls_l = 0
    spar_l = 0
    cont_l = 0
    sparsity_item = 0
    total_loss = 0
    model.eval()
    print("Validate")
    with torch.no_grad():

        for (batch, (inputs, masks, labels)) in enumerate(dev_loader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            rationales, logits = model(inputs, masks)
            # pdb.set_trace()
            logits = torch.softmax(logits, dim=-1)

            # computer loss
            cls_loss = args.cls_lambda * F.cross_entropy(logits, labels)
            sparsity_loss = args.sparsity_lambda * get_sparsity_loss(rationales[:, :, 1], masks, args.sparsity_percentage)
            sparsity = (torch.sum(rationales[:, :, 1]) / torch.sum(masks)).cpu().item()
            continuity_loss = args.continuity_lambda * get_continuity_loss(rationales[:, :, 1])
            sparsity_item += sparsity
            loss = cls_loss + sparsity_loss + continuity_loss


            _, pred = torch.max(logits, axis=-1)
            TP += ((pred == 1) & (labels == 1)).cpu().sum()
            TN += ((pred == 0) & (labels == 0)).cpu().sum()
            FN += ((pred == 0) & (labels == 1)).cpu().sum()
            FP += ((pred == 1) & (labels == 0)).cpu().sum()

            cls_l += cls_loss.cpu().item()
            spar_l += sparsity_loss.cpu().item()
            cont_l += continuity_loss.cpu().item()
            total_loss += loss.item()

        print("cls_l:{} spar_l:{} cont_l:{},sparsity_item:{}".format(cls_l,spar_l,cont_l,sparsity_item))
        avg_loss = total_loss / len(dev_loader)



        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (recall + precision)
        dev_accuracy = (TP + TN) / (TP + TN + FP + FN)

        dev_loss = [total_loss,cls_l,spar_l,cont_l,sparsity_item,avg_loss]

        print("dev epoch:{} recall:{:.4f} precision:{:.4f} f1-score:{:.4f} dev_accuracy:{:.4f}".format(epoch, recall,precision,f1_score, dev_accuracy))


        print("Validate Sentence")
        validate_dev_sentence(model, dev_loader, device,(writer,epoch))
        
        print("Annotation")
        annotation_results = validate_share(model, annotation_loader, device)
        print("The annotation performance: sparsity: %.4f, accuracy:%.4f,  precision: %.4f, recall: %.4f, f1: %.4f"
            % (100 * annotation_results[0], 100 * annotation_results[1],100 * annotation_results[2], 100 * annotation_results[3],100 * annotation_results[4]))


        train_total_loss,train_cls_l,train_spar_l,train_cont_l,train_sparsity_item,avg_train_loss = train_loss

        val_total_loss,val_cls_l,val_spar_l,val_cont_l,val_sparsity_item,avg_val_loss = dev_loss

        results_logging.append(
        [train_total_loss,train_cls_l,train_spar_l,train_cont_l,train_sparsity_item,avg_train_loss,
         val_total_loss,val_cls_l,val_spar_l,val_cont_l,val_sparsity_item,avg_val_loss,
         train_accuracy.item(),dev_accuracy.item(),annotation_results[1].item(),annotation_results[0].item(),annotation_results[2].item(),annotation_results[3].item(),
         annotation_results[4].item()]
        )

        print("Annotation Sentence")
        validate_annotation_sentence(model, annotation_loader, device)
        print("Rationale")
        validate_rationales(model, annotation_loader, device,(writer,epoch))
        if dev_accuracy>acc_best_dev[-1]:
            acc_best_dev.append(dev_accuracy)
            best_dev_epoch.append(epoch)
            f1_best_dev.append(annotation_results[4])
        if best_all<annotation_results[4]:
            best_all=annotation_results[4]


train_cost_time_epoch_per = sum_train_cost_time / int(args.epochs)
print("Cost Time per epoch: %f second"%train_cost_time_epoch_per)

print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)
if args.save==1:
    if args.data_type=='beer':
        torch.save(model.state_dict(),'./trained_model/beer/aspect{}_dis{}.pkl'.format(args.aspect))
        print('save the model')
    elif args.data_type=='hotel':
        torch.save(model.state_dict(), './trained_model/hotel/aspect{}_dis{}.pkl'.format(args.aspect))
        print('save the model')
else:
    print('not save')
