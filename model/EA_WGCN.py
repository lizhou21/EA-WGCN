import torch
import torch.nn as nn
from utils import constant
from utils import torch_utils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from model.LAM import Tree,head_to_tree,tree_to_adj

class GCNClassifier(nn.Module):
    def __init__(self,opt,emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt,emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_features=in_dim,out_features=opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()


    def forward(self, inputs):
        outputs,pooling_output = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits,pooling_output



class GCNRelationModel(nn.Module):#产生adj作为GCN的输入之一
    def __init__(self,opt,emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.emb = nn.Embedding(num_embeddings=opt['vocab_size'],embedding_dim=opt['emb_dim'],padding_idx=constant.PAD_ID)#全用0填补,保证第一行为0
        self.pos_emb = nn.Embedding(num_embeddings=len(constant.POS_TO_ID),embedding_dim=opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(num_embeddings=len(constant.NER_TO_ID),embedding_dim=opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb,self.pos_emb,self.ner_emb)
        self.init_embeddings()
        self.gcn = GCN(opt,embeddings,opt['hidden_dim'],opt['num_layers'])
        self.sub_att = Entity_Attention(opt,opt['hidden_dim'])
        self.obj_att = Entity_Attention(opt,opt['hidden_dim'])
        # self.self_att = Self_Attention(opt,opt['hidden_dim'],opt['att_dropout'])
        in_dim = opt['hidden_dim']*3
        layers = [nn.Linear(in_dim,opt['hidden_dim']),nn.ReLU()]
        for _ in range(opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'],opt['hidden_dim']),nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0,1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)

        #'topn':Only finetune top N word embeddings.
        if self.opt['topn'] <= 0:
            print('Do not finetune word embedding layer.')
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print('Finetune all embeddings.')

    def forward(self,inputs):
        #GCN中需用到adj，构造adj
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head,words,l,prune,subj_pos,obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(),subj_pos.cpu().numpy(),obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))] #为每个数据构造1个tree
            adj = [tree_to_adj(maxlen,tree,directed=False,self_loop=True).reshape(1,maxlen,maxlen) for tree in trees]
            adj = np.concatenate(adj,axis=0) #三维
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data) #变量张量，可计算梯度
        h, pool_mask = self.gcn(adj, inputs) #h:(batch,seq_len,m) 完成了GCN


        #pool  #此刻进行pool
        # subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2),obj_pos.eq(0).eq(0).unsqueeze(2)# mask没有的为1
        # pool_type = self.opt['pooling']
        # h_out = pool(h, pool_mask, pool_type)
        # subj_out = pool(h, subj_mask, pool_type)#batch*m
        # obj_out = pool(h, obj_mask, pool_type)
        # outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        # outputs = self.out_mlp(outputs)

        #进行attention_pool
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # mask没有的为1
        pool_type = self.opt['pooling']

        subj_out = pool(h, subj_mask, pool_type)  # batch*m
        subj_out = self.sub_att(h, subj_out, subj_mask, pool_type)

        obj_out = pool(h, obj_mask, pool_type)
        obj_out = self.obj_att(h, obj_out, obj_mask, pool_type)

        # h_att = self.self_att(h)
        h_out = pool(h, pool_mask, pool_type)

        outputs = torch.cat([h_out, subj_out, obj_out], dim=1)
        outputs = self.out_mlp(outputs)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               






        return outputs,h_out



class GCN(nn.Module):
    def __init__(self,opt,embeddings,mem_dim,num_layers):
        super(GCN,self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim

        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.emb, self.pos_emb, self.ner_emb = embeddings
        # if opt['isNER']:
        #     self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        #     self.emb, self.pos_emb, self.ner_emb = embeddings
        # else:
        #     self.in_dim = opt['emb_dim'] + opt['pos_dim']
        #     self.emb, self.pos_emb = embeddings

        #rnn layers
        if self.opt.get('rnn',False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size,opt['rnn_hidden'],opt['rnn_layers'],batch_first=True,dropout=opt['rnn_dropout'],bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2 #用于GCN层的输入，双向RNN，输出维度200*2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout'])  # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])#embedding层后
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        #gcn layers
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer==0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self,adj,inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs  # unpack,all tensor
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        embs = torch.cat(embs,dim=2)
        embs = self.in_drop(embs)

        #rnn layers
        if self.opt.get('rnn', False):
            # gcn_input = self.rnn(embs) 不能这么单纯的直接传入RNN layers
            gcn_input = self.gcn_drop(self.encode_with_rnn(embs,masks,words.size()[0])) #words.size()[0]不用opt['batch_size']。采用实际的batch大小
        else:
            gcn_input = embs

        #gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2) #adj的mask。节点是否在图中
        if self.opt.get('no_adj', False):#没有adj,取0
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_input) #adj（b*n*n）,gcn_input:(b*n*m),Ax:(b*b*m)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_input) #自环
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gcn_output = self.gcn_drop(gAxW) if l<self.layers-1 else gAxW
            gcn_input = gcn_output + gcn_input

        return gcn_output, mask



class GRU_gcn(nn.Module):
    def __init__(self,opt,mem_dim):
        super(GRU_gcn,self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.W_at = nn.Linear(self.mem_dim,self.mem_dim)
        self.U_at_1 = nn.Linear(self.mem_dim,self.mem_dim)

    def forward(self, adj, inputs):
        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)  # adj的mask。节点是否在图中
        att = adj.bmm(inputs)
        att = att / denom
        forg = torch.sigmoid(self.W_at(att) + self.U_at_1(inputs)) #遗忘门
        rem = 1 - forg







class Entity_Attention(nn.Module):
    def __init__(self,opt,mem_dim):
        super(Entity_Attention,self).__init__()
        self.opt = opt
        self.mem_dim = mem_dim
        self.W = nn.Linear(self.mem_dim,self.mem_dim)
        
    def forward(self,inputs,entity,entity_mask,pool_type): #inputs 表示GCN后的句子表示，形状b*n*m，entity：GCN后经过pool后得到的entity表示 b*m,要得到结果b*n*1  或者b*n
        entity = torch.unsqueeze(entity,2)  #(b*m*1)
        if self.opt['score_type'] == 'HE':   # HE  or  HWE
            score = inputs.bmm(entity)
        if self.opt['score_type'] == 'HWE':
            HW = self.W(inputs)
            score = HW.bmm(entity) #b*n*1
        att = torch.softmax(score,dim=1)
        entity_out = att.mul(inputs) #(b*n*m)
        entity_pool = pool(entity_out, entity_mask, pool_type)
        return entity_pool



class Self_Attention(nn.Module):
    def __init__(self, opt, hidden_dim, att_dropout=0.1):
        super(Self_Attention, self).__init__()
        self.opt = opt
        self.hidden_dim = hidden_dim
        # self.att_drop = nn.Dropout(att_dropout)
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, inputs):
        inputs_T = inputs.transpose(1,2)
        QW = self.w(inputs)
        score = torch.bmm(QW,inputs_T)
        attn = torch.softmax(score,dim=2) #(b*n*n)
        # attn = attn / self.hidden_dim #scale
        # attn = self.att_drop(attn)
        output = torch.bmm(attn, inputs) #(b*n*m)
        return output



def rnn_zero_state(batch_size,hidden_dim,num_layers,bidirectional=True, use_cuda=True):
    total_layes = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layes, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


def pool(h,mask,type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)  # mask中为1的元素所在的索引，在h中相同的的索引处替换为value

        # 返回的是每个（60，200）中的（1，200）
        return torch.max(h, 1)[0]  # torch.max(h, 1)，在h的维度中的第1维里选取最大值，返回list(最大值+对应的索引)
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)