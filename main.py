import numpy as np
import torch
import pickle
from model import HypergraphRec
from utils import metrics, scipy_sparse_mat_to_torch_sparse_tensor
import pandas as pd
from parser import args
from tqdm import tqdm
import time
import torch.utils.data as data
from utils import TrnData
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyperparameters
d = args.d
l = args.gnn_layer
temp = args.temp
batch_user = args.batch
epoch_no = args.epoch
lambda_1 = args.lambda1
lambda_2 = args.lambda2
dropout = args.dropout
lr = args.lr
decay = args.decay

# load data
path = 'data/' + args.data + '/'
f = open(path+'trn_matrix.pkl','rb')
train = pickle.load(f)
train_csr = (train!=0).astype(np.float32)
f = open(path+'tst_matrix.pkl','rb')
test = pickle.load(f)

# Items'Review embedding
sentence_embeddings = torch.load(path + 'review_embedding.pt')

#Incidence Matrix

df = pd.read_csv(path + 'UI_rating.csv')

incidence_matrix_user = pd.pivot_table(df, index='user', columns='product', aggfunc=len, fill_value=0)
incidence_matrix_user = incidence_matrix_user.float()

incidence_matrix_item = pd.pivot_table(df, index='product', columns='user', aggfunc=len, fill_value=0)
incidence_matrix_item = incidence_matrix_item.float()

# Graphs
graph_tust,_ = dgl.load_graphs(path +'Trust_graph.bin')
graph_tust = graph_tust[0]
graph_tust = dgl.add_self_loop(graph_tust)
adjacency_matrix_T = graph_tust.adjacency_matrix(scipy_fmt='coo')  
adj_T = adjacency_matrix_T.toarray()
adj_T = torch.tensor(adj_T)
adj_T = adj_T.float()
graph_user, _ = dgl.load_graphs(path + 'graph_review_user.bin')
graph_user = graph_user[0]
graph_user = dgl.add_self_loop(graph_user)
djacency_matrix_ur = graph_user.adjacency_matrix(scipy_fmt='coo')  
adj_ur = adjacency_matrix_ur.toarray()
adj_ur = torch.tensor(adj_ur)
adj_ur = adj_ur.float()
graph_product,_ = dgl.load_graphs(path + 'graph_metapath_item.bin')
graph_product = graph_product[0]
graph_product = dgl.add_self_loop(graph_product)
djacency_matrix_pr = graph_user.adjacency_matrix(scipy_fmt='coo')  
adj_pr = adjacency_matrix_pr.toarray()
adj_pr = torch.tensor(adj_pr)
adj_pr = adj_ur.float()

class Embedding(nn.Module):
    def __init__(self,num_nodes,hidden_dim):
        super(Embedding, self).__init__()
        self.user_embedding = nn.Embedding(num_nodes+1, hidden_dim)

    def forward(self,nodes):
        user_embeds = self.user_embedding(users)
        return user_embeds
nodes = list(nu for nu in range(graph_product.num_nodes()))
num_nodes = len(nodes)
Embeddings = Embedding(num_nodes,80)
features_user = Embeddings(nodes)
nodes = list(nu for nu in range(graph_user.num_nodes()))
num_nodes = len(nodes)
Embeddings = Embedding(num_nodes,80)
features_item = Embeddings(nodes)

# normalizing the adj matrix
epoch_user = min(train.shape[0], 30000)
rowD = np.array(train.sum(1)).squeeze()
colD = np.array(train.sum(0)).squeeze()
for i in range(len(train.data)):
    train.data[i] = train.data[i] / pow(rowD[train.row[i]]*colD[train.col[i]], 0.5)

# construct data loader
train = train.tocoo()
train_data = TrnData(train)
train_loader = data.DataLoader(train_data, batch_size=args.inter_batch, shuffle=True, num_workers=0)

adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(train)
adj_norm = adj_norm.coalesce().cuda(torch.device(device))
print('Adj matrix normalized.')

# process test set
test_labels = [[] for i in range(test.shape[0])]
for i in range(len(test.data)):
    row = test.row[i]
    col = test.col[i]
    test_labels[row].append(col)
print('Test data processed.')

loss_list = []
loss_r_list = []
loss_s_list = []
HR_10_x = []
HR_10_y = []
ndcg_10_y = []
HR_20_y = []
ndcg_20_y = []

model = HypergraphRec(adj_norm.shape[0], adj_norm.shape[1], 32 , train_csr, adj_norm, 2, 0.2, 0.2, 1e-7, 0.0, 256, device, features_user,sentence_embeddings,features_item)

model.to(device)
batch_user=256
current_lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(),weight_decay= decay,lr= lr)

batch_user = 256

for epoch in range(100):

    epoch_loss = 0
    epoch_loss_r = 0
    epoch_loss_s = 0
    train_loader.dataset.neg_sampling()
    for i, batch in enumerate(tqdm(train_loader)):
        uids, pos, neg = batch
        print("uids",len(uids),uids)
        print("pos",len(pos),pos)
        print("neg",len(neg),neg)
        uids = uids.long().to(device)
        pos = pos.long().to(device)
        neg = neg.long().to(device)
        iids = torch.concat([pos, neg], dim=0)
        print("iids:", len(iids))

        # feed
        optimizer.zero_grad()
        loss, loss_r, loss_s= model(self, uids, iids, pos, neg,adj_T,adj_ur,adj_pr,incidence_matrix_user, incidence_matrix_item)
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.cpu().item()
        epoch_loss_r += loss_r.cpu().item()
        epoch_loss_s += loss_s.cpu().item()

    batch_n = len(train_loader)
    epoch_loss = epoch_loss/batch_n
    epoch_loss_r = epoch_loss_r/batch_n
    epoch_loss_s = epoch_loss_s/(batch_n)
    loss_list.append(epoch_loss)
    loss_r_list.append(epoch_loss_r)
    loss_s_list.append(epoch_loss_s)
    #print('Epoch:',epoch,'Loss:',epoch_loss,'Loss_r:',epoch_loss_r,'Loss_s:',epoch_loss_s)

    if epoch % 3 == 0:
        test_uids = np.array([i for i in range(adj_norm.shape[0])])
        batch_n = int(np.ceil(len(test_uids)/batch_user))

        all_HR_10 = 0
        all_ndcg_10 = 0
        all_HR_20 = 0
        all_ndcg_20 = 0
        for batch in tqdm(range(batch_n)):
            start = batch*batch_user
            end = min((batch+1)*batch_user,len(test_uids))

            test_uids_input = torch.LongTensor(test_uids[start:end]).to(device)
            predictions = model(test_uids_input,None,None,None,None,None,None,None,None,test=True)
            predictions = np.array(predictions.cpu())

            #top@10
            HR_10,ndcg_10 = metrics(test_uids[start:end],predictions,10,test_labels)
            #top@20
            HR_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)

            all_HR_10+=HR_10
            all_ndcg_10+=ndcg_10
            all_HR_20+=HR_20
            all_ndcg_20+=ndcg_20
        print('-------------------------------------------')
        #print('Final test:','HR@10:',all_HR_10/batch_n,'Ndcg@10:',all_ndcg_20/batch_n,'Ndcg@10:',all_ndcg_10/batch_n)
        HR_10_x.append(epoch)
        HR_10_y.append(all_HR_10/batch_n)
        ndcg_10_y.append(all_ndcg_10/batch_n)
        HR_20_y.append(all_HR_20/batch_n)
        ndcg_20_y.append(all_ndcg_20/batch_n)

# final test
test_uids = np.array([i for i in range(adj_norm.shape[0])])
batch_n = int(np.ceil(len(test_uids)/batch_user))

all_HR_10 = 0
all_ndcg_10 = 0
all_HR_20 = 0
all_ndcg_20 = 0
for batch in range(batch_n):
    start = batch*batch_user
    end = min((batch+1)*batch_user,len(test_uids))

    test_uids_input = torch.LongTensor(test_uids[start:end]).to(device)
    predictions = model(test_uids_input,None,None,None,None,None,None,None,None,test=True)
    predictions = np.array(predictions.cpu())

    #top@10
    HR_10,ndcg_10 = metrics(test_uids[start:end],predictions,10,test_labels)
    #top@20
    HR_20, ndcg_20 = metrics(test_uids[start:end],predictions,20,test_labels)
    all_HR_10+=HR_10
    all_ndcg_10+=ndcg_10
    all_HR_20+=HR_20
    all_ndcg_20+=ndcg_20
print('-------------------------------------------')
print('Final test:','HR@10:',all_HR_10/batch_n,'Ndcg@10:',all_ndcg_20/batch_n,'Ndcg@10:',all_ndcg_10/batch_n)
