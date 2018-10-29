import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim

class lstm(nn.Module):
	def __init__(self,embed_dim,hidden_dim,batchsize,maxwords=1640):
		super(lstm,self).__init__()
	
		self.embedding_layer = nn.Embedding(maxwords+1,embed_dim)
		self.lstm = nn.LSTM(embed_dim,hidden_dim,batch_first=True)


	def forward(self,x1,x2):
		q1_emb = self.embedding_layer(x1)
		q2_emb = self.embedding_layer(x2)
		q1_out,_ = self.lstm(q1_emb,None)
		q2_out,_ = self.lstm(q2_emb,None)

		manhattan_dis = torch.exp(-torch.sum(torch.abs(q1_out[:,-1,:]-q2_out[:,-1,:]),dim=1,keepdim=True))

		return manhattan_dis


