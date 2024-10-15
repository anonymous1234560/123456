import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import dgl
#from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,1, bias=False),
        )


    def forward(self, z):
        w2 = self.project(z)
        w1 = self.project(z).mean(0)  
        beta1 = torch.softmax(w1, dim=0)  
        beta = beta1.expand((z.shape[0],) + beta1.shape) 
        return (beta * z).sum(1)
    
class HGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(HGCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, incidence_matrix):
        deg = incidence_matrix.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  
        
        norm_incidence = deg_inv_sqrt.view(-1, 1) * incidence_matrix * deg_inv_sqrt.view(-1, 1)

        x = torch.matmul(norm_incidence.t(), x)
        
        x = self.linear(x)
        
        return x

class HGCN(nn.Module):
    def __init__(self, num_features, hidden_dim, out_featuer):
        super(HGCN, self).__init__()
        self.layer1 = HGCNLayer(num_features, hidden_dim)
        self.layer2 = HGCNLayer(hidden_dim, out_featuer)

    def forward(self, x, incidence_matrix):
        x = self.layer1(x, incidence_matrix)
        print(x, x.shape)
        return x




class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x, adj):
        
        adj = adj + torch.eye(adj.size(0)) 
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        x = self.linear(x)
        x = torch.matmul(norm_adj, x)
        return x

class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, out_feature):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(num_features, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, out_feature)

    def forward(self, x, adj):
        x = F.relu(self.layer1(x, adj))
        x = self.layer2(x, adj)
        return x
class DimensionalityReduction(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimensionalityReduction, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")

from info_nce import InfoNCE

class HypergraphRec(nn.Module):
    def __init__(self, n_u, n_i, d, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device,features,sentence_embeddings,features_item):
        super(HypergraphRec,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        print("self.E_u_0:",self.E_u_0,self.E_u_0.shape)
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))
        self.feat_user = features
        self.feat_item = features_item
        self.encoder1 = GCN(self.feat.shape[1], 64,d)
        self.encoder2 = GCN(self.feat_item.shape[1], 32,d)
        self.model_hgcn1 = HGCN(self.E_u_0.shape[1],d,d)
        self.model_hgcn2 = HGCN(self.E_i_0.shape[1],d,d)

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None
        self.device = device
        self.emd = sentence_embeddings
        self.reducer = DimensionalityReduction(self.emd.shape[1],32)
        self.attention =  SemanticAttention(32)
        self.loss = InfoNCE()
        
        self.semantic_attention = SemanticAttention(
            in_size=32 * 2
        )
    def forward(self, uids, iids, pos, neg,adj_T,adj_ur,adj_pr,incidence_matrix_user, incidence_matrix_item,test=False):
        semantic_embeddings = []
        semantic_embeddings1 = []
        if test==True: 
            print(" self.E_u", self.E_u.shape)
            print(" self.E_i.T", self.E_i.T.shape)
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).to(device)
            print("mask",mask.shape)
            print("preds",preds.shape)
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = self.model_hgcn1(self.E_i_0 ,incidence_matrix_user )
                self.Z_i_list[layer] = self.model_hgcn2(self.E_u_0 ,incidence_matrix_item ) )



                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]
            self.G_u1 = self.encoder1(self.feat_user,adj_T)
            semantic_embeddings1.append(self.G_u1)
            self.G_u2 = self.encoder1(self.feat_user,adj_ur)
            semantic_embeddings1.append(self.G_u2)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)
            self.G_i1 = self.encoder2(self.feat_item,adj_pr)
            semantic_embeddings.append(self.G_i1)
            
            # bert

            self.G_i2 = self.reducer( self.emd)
            semantic_embeddings.append(self.G_i2)
            semantic_embeddings1 = torch.stack(
            semantic_embeddings1, dim=1 )
            self.G_u = self.attention(semantic_embeddings1)

            semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1 )
            self.G_i = self.attention(semantic_embeddings)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm =self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean()+ (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            
            loss_s = (-pos_score + neg_score)
            print("loss_s:",loss_s)
            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()
            print('loss_r:',loss_r)

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2
            print('loss_reg:',loss_reg)

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            return loss, loss_r, self.lambda_1 * loss_s
