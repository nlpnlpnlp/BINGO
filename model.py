import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
Tensor = torch.Tensor

class Rnn(nn.Module):

    def __init__(self, cell_type, embedding_dim, hidden_dim, num_layers):
        super(Rnn, self).__init__()
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_dim // 2,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_dim // 2,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        else:
            raise NotImplementedError('cell_type {} is not implemented'.format(cell_type))

    def forward(self, x):
        """
        Inputs:
        x - - (batch_size, seq_length, input_dim)
        Outputs:
        h - - bidirectional(batch_size, seq_length, hidden_dim)
        """
        h = self.rnn(x)
        return h

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = True

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)


class Rnp_model(nn.Module): 
    def __init__(self, args):
        super(Rnp_model, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z 

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)   
        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  
        
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  
        
        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. - masks_) * (-1e6)
        
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding

    def g_skew(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_output, _ = self.gen(embedding)  
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

    def train_skew(self,inputs,masks,labels):
        masks_ = masks.unsqueeze(-1)

        labels_=labels.detach().unsqueeze(-1)       #batch*1
        pos=torch.ones_like(inputs)[:,:10]*labels_
        neg=-pos+1
        skew_pad=torch.cat((pos,neg),dim=1)
        latter=torch.zeros_like(inputs)[:,20:]

        masks_=torch.cat((skew_pad,latter),dim=1).unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.gen(embedding)  
        outputs = self.layernorm1(outputs)
        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def train_skew_new(self, inputs, skew_masks):
        """
        Pretrain the predictor using only the first sentence.

        inputs:      (batch_size, seq_length)
        skew_masks:  (batch_size, seq_length), 1 for first sentence, 0 otherwise
        labels:      (batch_size,)
        """
        skew_masks_ = skew_masks.unsqueeze(-1)
        embedding = self.embedding_layer(inputs)
        embedding = embedding * skew_masks_

        cls_outputs, _ = self.cls(embedding)  
        cls_outputs = cls_outputs * skew_masks_ + (1. - skew_masks_) * (-1e6)
        
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def get_cls_param(self):
        gen_param_ids = set(map(id, self.generator.parameters()))
        cls_params = filter(lambda p: id(p) not in gen_param_ids, self.parameters())
        return list(cls_params)


    
    def get_gen_param(self):
        layers = [self.gen_fc]
        layers = [self.embedding_layer, self.gen, self.layernorm1, self.gen_fc]
        params = []
        for layer in layers:
            params.extend([param for param in layer.parameters() if param.requires_grad])
        return params


class Fr_model(nn.Module):
    def __init__(self, args):
        super(Fr_model, self).__init__()
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.enc = Rnn(args.cell_type,
                       args.embedding_dim,
                       args.hidden_dim,
                       args.num_layers)
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm = nn.LayerNorm(args.hidden_dim)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits,  tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_output, _ = self.enc(embedding)  
        gen_output = self.layernorm(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  
        z = self.independent_straight_through_sampling(gen_logits)  
        
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.enc(cls_embedding)  
        cls_outputs = self.layernorm(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        
        cls_logits = self.cls_fc(self.dropout(cls_outputs))

        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.enc(embedding)  
        outputs = self.layernorm(outputs)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def g_skew(self,inputs, masks):
        #  masks_ (batch_size, seq_length, 1)
        masks_ = masks.unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_output, _ = self.enc(embedding)  
        gen_output = self.layernorm(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log



class Perturb_model(nn.Module):
    def __init__(self, args):
        super(Perturb_model, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)

        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        if args.fr==1:
            self.cls=self.gen
        else:
            self.cls = nn.GRU(input_size=args.embedding_dim,
                              hidden_size=args.hidden_dim // 2,
                              num_layers=args.num_layers,
                              batch_first=True,
                              bidirectional=True)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)


    def perturb_gumbel(self,mask, logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, threshold_perturb: float =0.1) -> Tensor:

        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)


        if hard:
            index = y_soft.max(dim, keepdim=True)[1]

            perturbation_sample = torch.FloatTensor(index.size()).to(index.device)
            perturbation_sample = torch.rand(index.size(),out=perturbation_sample)      


            perturbation_threshold=torch.ones_like(perturbation_sample)*threshold_perturb     
            perturbation_p=torch.cat((perturbation_threshold,perturbation_sample),dim=-1)
            perturbation_or_not=perturbation_p.min(dim,keepdim=True)[1].to(index.device)      

            perturb_selected=perturbation_or_not*index*mask         
            num_of_perturb=int(torch.sum(perturb_selected)/perturb_selected.size()[0])    

            index_unselected=torch.ones_like(index)-index           
            index_unselected=index_unselected*mask                  
            index_unselected=index_unselected.squeeze(-1).float()          
            num_unselected=(torch.sum(index_unselected,dim=1))
            if min(num_unselected)==0:
                perturb_all =perturb_selected
            else:
                try:
                    index_perturb_unselected=torch.LongTensor([index_unselected.shape[0],max(num_of_perturb,1)]).to(index.device)
                    torch.multinomial(index_unselected,max(num_of_perturb,1),out=index_perturb_unselected)

                    perturb_unselected = torch.zeros_like(index_unselected).scatter_(1, index_perturb_unselected,
                                                                                     1).unsqueeze(
                        -1).long()  

                    perturb_all = torch.max(perturb_unselected, perturb_selected)  
                except:
                    print('fail to sample unselected parts')
                    print(torch.prod(torch.sum(index_unselected,dim=1)))
                    print(torch.sum(index_unselected,dim=1))





            index2=abs(index-perturb_all) 

            y_hard_perturb = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index2, 1.0)
            ret_perturb= y_hard_perturb - y_soft.detach() + y_soft      




            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft         
        else:
            ret = y_soft
        return ret,ret_perturb

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        # z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)    

        embedding = masks_ * self.embedding_layer(inputs)  

        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  
        
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        
        logits = self.cls_fc(self.dropout(outputs))
        return logits


    def g_skew(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)
        
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_output, _ = self.gen(embedding)  
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log

    def perturb_forward(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)

        
        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits = self.generator(embedding)

        if perturb_rate==0:
            z = self.independent_straight_through_sampling(gen_logits)  
            z_perturb=z
        else:
            z,z_perturb=self.perturb_gumbel(masks_,gen_logits,tau=1, hard=True,threshold_perturb=perturb_rate)



        
        cls_embedding = embedding * (z_perturb[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def get_perturb_rationale(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)

        
        embedding = masks_ * self.embedding_layer(inputs)  

        gen_emb, _ = self.gen(embedding)
        gen_emb = self.layernorm1(gen_emb)

        gen_logits = self.gen_fc(self.dropout(gen_emb))

        if perturb_rate == 0:
            z = self.independent_straight_through_sampling(gen_logits)  
            z_perturb = z
        else:
            z, z_perturb = self.perturb_gumbel(masks_, gen_logits, tau=1, hard=True, threshold_perturb=perturb_rate)

        return z, z_perturb

    def get_rationale(self, inputs, masks, perturb_rate):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)  


        gen_emb, _ = self.gen(embedding)
        gen_emb = self.layernorm1(gen_emb)

        gen_logits = self.gen_fc(self.dropout(gen_emb))
        z = self.independent_straight_through_sampling(gen_logits)  



        return z

    def pred_with_rationale(self, inputs, masks, z):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -masks_) * (-1e6)

        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)

        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits) 
        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)

        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding










