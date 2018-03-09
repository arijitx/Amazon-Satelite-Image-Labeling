from load_data import DataGen
from cnn2rnn import EncoderCNN,DecoderRNN
import argparse
from tqdm import tqdm

from torchvision import transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import CrossEntropyLoss

import numpy as np 



epochs=10
learing_rate=0.0001
batch_size=128
hidden_size=256
emb_size=256
output_size=19
CUDA_DEVICE=2
validation_per_batch=5

train_data=DataGen('processed_data/train_ims.npy','processed_data/train_labels.npy',batch_size=batch_size)
val_data=DataGen('processed_data/val_ims.npy','processed_data/val_labels.npy',batch_size=batch_size)

enc=EncoderCNN(emb_size,CUDA_DEVICE).float()
dec=DecoderRNN(emb_size,hidden_size,output_size,CUDA_DEVICE).float()
criterion=nn.CrossEntropyLoss()
criterion_val=nn.CrossEntropyLoss()

print('> Models Created ! ')

params = list(dec.parameters()) + list(enc.linear.parameters()) + list(enc.bn.parameters())
optimizer = torch.optim.Adam(params, lr=learing_rate)
total_loss=0

for e in range(epochs):
	train_data.reset()
	for bi in train_data.n:
		#get batch 
		imb,lbb,lnb=train_data.get_batch()

		#load variables to gpu
		imb=Variable(torch.from_numpy(imb),requires_grad=False,volatile=True).float().cuda(CUDA_DEVICE)
		lbb=Variable(torch.from_numpy(lbb)).cuda(CUDA_DEVICE)
		lnb=Variable(torch.from_numpy(lnb)).cuda(CUDA_DEVICE)

		#pack_padded_Seq Len for target
		lens=lnb.cpu().data.numpy().flatten().astype('int').tolist()
		targets = pack_padded_sequence(lbb, lens, batch_first=True)[0]

		#encoder- decoder forward
		context=enc.forward(imb)
		output=dec.forward(lbb,context,lnb)

		#zerograd enc dec
		enc.zero_grad()
		dec.zero_grad()

		#compute loss
		loss=criterion(output,targets)

		#update weights
		loss.backward()
		optimizer.step()

		# validation
		val_data.idx=0
		val_loss=0
		total_loss+=loss
		# print('> Epoch :',e+1,' Batch :',bi+1,' Loss :',loss.data[0])
		if (bi+1)%validation_per_batch==0:
			for j in val_data.n:
				vim,vlb,vln=val_data.get_batch()
				vim=Variable(torch.from_numpy(vim),requires_grad=False,volatile=True).float().cuda(CUDA_DEVICE)
				vlb=Variable(torch.from_numpy(vlb)).cuda(CUDA_DEVICE)
				vln=Variable(torch.from_numpy(vln)).cuda(CUDA_DEVICE)

				ctx=enc(vim)
				val_labels,val_opt=dec.infer(ctx,val_data.labels_max_size)
				# val_opt=Variable(val_opt).cuda(CUDA_DEVICE)
				vln=vln.cpu().data.numpy().flatten().astype('int').tolist()
				vlb=pack_padded_sequence(vlb,vln,batch_first=True)[0]
				vpred=pack_padded_sequence(val_opt,vln,batch_first=True)[0]
				val_loss+=criterion(vpred,vlb)
			val_loss=val_loss/max(val_data.n)+1
			total_loss=total_loss/validation_per_batch
			print('> Epoch :',e+1,' Batch :',bi+1,' Loss :',total_loss.data[0],' Val Loss :',val_loss.data[0])

