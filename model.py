import numpy as np
import torch.nn as nn
import torch
import pickle

import torch.nn.functional
import param
import losses

class ParameterPackage:
    def __init__(self):
        self.user_size = -1
        self.item_size = -1
        self.embedding_dimensionality = 32 # d
        self.hyperedge_count = 128 # H
    def update(self):
        self.user_size = param.args.user
        self.item_size = param.args.item

class HCCFModel(nn.Module):
    def __init__(self, params:ParameterPackage):
        super(HCCFModel, self).__init__()
        self.user_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(params.user_size, params.embedding_dimensionality)))
        self.item_embedding = nn.Parameter(nn.init.xavier_uniform_(torch.empty(params.item_size, params.embedding_dimensionality)))
        self.GCN = GCN()
        self.HGNN = HGNN()
        self.user_hyper_matrix = nn.Parameter(nn.init.xavier_uniform_(torch.empty(params.embedding_dimensionality, params.hyperedge_count)))
        self.item_hyper_matrix = nn.Parameter(nn.init.xavier_uniform_(torch.empty(params.embedding_dimensionality, params.hyperedge_count)))
        self.edgedrop = SpAdjDropEdge()
        
    def forward(self, adjacent, keep_rate):
        embedddings = torch.concat([self.user_embedding, self.item_embedding])
        user_hyper_product = self.user_embedding @ self.user_hyper_matrix
        item_hyper_product = self.item_embedding @ self.item_hyper_matrix
        
        sequence = [embedddings]
        gnn_sequence = []
        hyper_sequence = []
        
        
        for i in range(param.args.gnn_layers):
            temp_embedding = self.GCN(self.edgedrop(adjacent, keep_rate), sequence[-1])
            hyper_user_sequence = self.HGNN(torch.nn.functional.dropout(user_hyper_product, 1-keep_rate), sequence[-1][:param.args.user])
            hyper_item_sequence = self.HGNN(torch.nn.functional.dropout(item_hyper_product, 1-keep_rate), sequence[-1][param.args.user:])
            gnn_sequence.append(temp_embedding)
            hyper_sequence.append(torch.concat([hyper_user_sequence, hyper_item_sequence]))
            sequence.append(temp_embedding + hyper_sequence[-1])
        
        result_embedding = sum(sequence)
        return result_embedding, gnn_sequence, hyper_sequence
    
    def loss(self, rows, cols, negs, adjacent, keepRate):
        embedding, gnn_sequence, hyper_sequence = self.forward(adjacent, keepRate)
        user_embedding, item_embedding = embedding[:param.args.user], embedding[param.args.user:]

        # row -> user, col -> item
        rows_embedding = user_embedding[rows]
        cols_embedding = item_embedding[cols]
        negs_embedding = item_embedding[negs]
        
        Pr = -torch.sum(rows_embedding * negs_embedding, dim=-1) + torch.sum(rows_embedding * cols_embedding, dim=-1)
        pw_marginal_loss = -(Pr).sigmoid().log().mean()
        #pw_marginal_loss = torch.maximum(torch.zeros_like(Pr), 1 - Pr).mean()
        
        contrast_loss = 0
        for i in range(param.args.gnn_layers):
            gnn_embedding = gnn_sequence[i]
            hyper_embedding = hyper_sequence[i]
            contrast_loss += (losses.contrast_loss(gnn_embedding[:param.args.user], hyper_embedding[:param.args.user], torch.unique(rows), param.args.temperture) +
                losses.contrast_loss(gnn_embedding[param.args.user:],   hyper_embedding[param.args.user:], torch.unique(cols), param.args.temperture))
        return pw_marginal_loss, contrast_loss
    
    def predict(self, adj):
        embeddings, gnn_sequence, hyper_sequence = self.forward(adj, 1.0)
        return embeddings[:param.args.user], embeddings[param.args.user:]
        
# Local Collaborative Relation Encoding Part
class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # The paper described the slope
        self.activation = nn.LeakyReLU(0.5)
        
    def forward(self, adjacent, embedding):
        mul = adjacent @ embedding
        #mul = self.activation(mul)
        return mul
    
# Hypergraph Global Dependency Learning Part
class HGNN(nn.Module):
    def __init__(self):
        super(HGNN, self).__init__()
        self.activation = nn.LeakyReLU(0.5)
    
    def forward(self, hyperedges, embedding):
        lamb = hyperedges.T @ embedding
        product = hyperedges @ lamb
        return self.activation(product)
    
class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((torch.rand(edgeNum) + keepRate).floor()).type(torch.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
