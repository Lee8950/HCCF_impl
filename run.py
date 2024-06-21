import torch.utils.data.dataloader
import model
import param
import pickle
import torch
import torch.utils.data
import numpy as np
import scipy.sparse
import torch.sparse
import json
import time
import datetime

def normalizeAdj(mat):
	degree = np.array(mat.sum(axis=-1))
	dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
	dInvSqrt[np.isinf(dInvSqrt)] = 0.0
	dInvSqrtMat = scipy.sparse.diags(dInvSqrt)
	return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

class TrainData(torch.utils.data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(param.args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TestData(torch.utils.data.Dataset):
	def __init__(self, coomat, trainMat):
		self.csrmat = (trainMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])

def calcRes(topLocs, tstLocs, batIds):
	assert topLocs.shape[0] == len(batIds)
	allRecall = allNdcg = 0
	for i in range(len(batIds)):
		temTopLocs = list(topLocs[i])
		temTstLocs = tstLocs[batIds[i]]
		tstNum = len(temTstLocs)
		maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, param.args.topk))])
		recall = dcg = 0
		for val in temTstLocs:
			if val in temTopLocs:
				recall += 1
				dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
		recall = recall / tstNum
		ndcg = dcg / maxDcg
		allRecall += recall
		allNdcg += ndcg
	return allRecall, allNdcg

if __name__ == '__main__':
    params = model.ParameterPackage()
    params.hyperedge_count = param.args.hyperedges

    if param.args.dataset == 'yelp':
        train_file_loc = 'data/yelp/trnMat.pkl'
        test_file_loc = 'data/yelp/tstMat.pkl'
    elif param.args.dataset == 'amazon':
        train_file_loc = 'data/amazon/trnMat.pkl'
        test_file_loc = 'data/amazon/tstMat.pkl'
    elif param.args.dataset == 'ml10m':
        train_file_loc = 'data/ml10m/trnMat.pkl'
        test_file_loc = 'data/ml10m/tstMat.pkl'
    else:
        raise RuntimeError("Dataset not implemented")
    
    with open(train_file_loc, 'rb') as fp:
        train_Mat = (pickle.load(fp) != 0).astype(np.float32)
    if type(train_Mat) is not scipy.sparse.coo_matrix:
        train_Mat = scipy.sparse.coo_matrix(train_Mat)
    
    param.args.user, param.args.item = train_Mat.shape
        
    with open(test_file_loc, 'rb') as fp:
        test_Mat = (pickle.load(fp) != 0).astype(np.float32)
    if type(test_Mat) is not scipy.sparse.coo_matrix:
        test_Mat = scipy.sparse.coo_matrix(test_Mat)
        
    train_data_loader = torch.utils.data.dataloader.DataLoader(TrainData(train_Mat), batch_size=param.args.batch_size, shuffle=True, num_workers=0)
    test_data_loader = torch.utils.data.dataloader.DataLoader(TestData(test_Mat, train_Mat), batch_size=param.args.test_batch, shuffle=False, num_workers=0)
    
    # making adjacent matrix, describing relationship between user/user item/item
    raw_and_not_hyper_edges = torch.sparse.FloatTensor()
    inter_user = scipy.sparse.csr_matrix((param.args.user, param.args.user))
    inter_item = scipy.sparse.csr_matrix((param.args.item, param.args.item))
    adjacent = train_Mat.copy()
    adjacent = scipy.sparse.vstack([scipy.sparse.hstack([inter_user, adjacent]), scipy.sparse.hstack([adjacent.transpose(), inter_item])])
    
    adjacent = (adjacent != 0) * 1.0
    adjacent = normalizeAdj(adjacent)
    
    adjacent = torch.sparse.FloatTensor(torch.from_numpy(np.vstack([adjacent.row, adjacent.col]).astype(np.int64)),
                                        torch.from_numpy(adjacent.data.astype(np.float32)),
                                        torch.Size(adjacent.shape)).cuda()
    
    params.update()
    
    HCCF = model.HCCFModel(params).cuda()
    optimizer = torch.optim.Adam(HCCF.parameters(), param.args.learning_rate, weight_decay=0)
    
    train_data_loader.dataset.negSampling()
    
    epochLoss, epochRecall, epochNdcg = [], [], []
    
    print()
    for _ in range(param.args.epoch):
        epoch_msg = f"Epoch {_+1}/{param.args.epoch}"
        steps = len(train_data_loader.dataset) // param.args.batch_size
        #Training
        epLoss = 0.0
        for idx, dat in enumerate(train_data_loader):
            rows, cols, negs = dat
            rows = rows.long().cuda()
            cols = cols.long().cuda()
            negs = negs.long().cuda()
        
            pw_marginal_loss, contrast_loss = HCCF.loss(rows, cols, negs, adjacent, param.args.keep_rate)
            contrast_loss = contrast_loss * param.args.ssl
        
            weight_decay = 0
            for w in HCCF.parameters():
                weight_decay += w.norm(2).square()
            weight_decay = weight_decay * param.args.reg

            loss = pw_marginal_loss + contrast_loss + weight_decay
            loss_val = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epLoss += loss_val
            print(f"Training:{epoch_msg}:Step:{idx}/{steps} loss:{loss_val}                            ",end='\r')
        print(f"Training:{epoch_msg}:loss:{epLoss/steps}                                       ")
        
        epochLoss.append({_+1:epLoss/steps})
        
        epRecall = 0
        epNdcg = 0
        
        if (_+1) % param.args.training_per_eval != 0:
            continue
        
        test_steps = len(test_data_loader.dataset) // param.args.test_batch
        for idx, (usr, train_mask) in enumerate(test_data_loader):
            usr = usr.long().cuda()
            train_mask = train_mask.cuda()
            
            user_embedding, item_embedding = HCCF.predict(adjacent)
            
            predictions = torch.mm(user_embedding[usr], torch.transpose(item_embedding, 1, 0)) * (1 - train_mask) - train_mask * 1e8
            
            __, toplocs = torch.topk(predictions, param.args.topk)
            recall, ndcg = calcRes(toplocs.cpu().numpy(), test_data_loader.dataset.tstLocs, usr)
            
            epRecall += recall
            epNdcg += ndcg
            
            print(f"Testing :{epoch_msg}:Step:{idx}/{test_steps} recall:{recall} ndcg:{ndcg}       ",end='\r')
        print(f"Testing :{epoch_msg}:recall:{epRecall/len(test_data_loader.dataset)} ndcg:{epNdcg/len(test_data_loader.dataset)}                                            ")
        epochRecall.append({_+1:epRecall/len(test_data_loader.dataset)})
        epochNdcg.append({_+1:epNdcg/len(test_data_loader.dataset)})
    
    epRecall = 0
    epNdcg = 0
        
    test_steps = len(test_data_loader.dataset) // param.args.test_batch
    for idx, (usr, train_mask) in enumerate(test_data_loader):
        usr = usr.long().cuda()
        train_mask = train_mask.cuda()
        
        user_embedding, item_embedding = HCCF.predict(adjacent)
       
        predictions = torch.mm(user_embedding[usr], torch.transpose(item_embedding, 1, 0)) * (1 - train_mask) - train_mask * 1e8
        
        __, toplocs = torch.topk(predictions, param.args.topk)
        recall, ndcg = calcRes(toplocs.cpu().numpy(), test_data_loader.dataset.tstLocs, usr)
        
        epRecall += recall
        epNdcg += ndcg
        
        print(f"Testing :{epoch_msg}:Step:{idx}/{test_steps} recall:{recall} ndcg:{ndcg}        ",end='\r')
    print(f"Testing :{epoch_msg}:recall:{epRecall/len(test_data_loader.dataset)} ndcg:{epNdcg/len(test_data_loader.dataset)}                                            ")
    epochRecall.append({param.args.epoch:epRecall/len(test_data_loader.dataset)})
    epochNdcg.append({param.args.epoch:epNdcg/len(test_data_loader.dataset)})
    filename = f"{'%s' % (datetime.datetime.now())}.{param.args.dataset}.{param.args.topk}.log.json:"
    
    with open(filename, "w") as fp:
        fp.write(json.dumps({param.args.dataset:{"Loss":epochLoss,"Recall":epochRecall,"Ndcg":epochNdcg}}))
    