from load_data import DataGen
from cnn2rnn import EncoderCNN,DecoderRNN
import argparse
from tqdm import tqdm

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



epochs=1
learing_rate=0.001
batch_size=32
hidden_size=256
emb_size=256
output_size=19

data_gen=DataGen('data/ims.npy','data/labels.npy',batch_size=batch_size)

enc=EncoderCNN(emb_size).float()
dec=DecoderRNN(emb_size,hidden_size,output_size).float()
criterion=nn.CrossEntropyLoss()
print('> Models Created ! ')

params = list(dec.parameters()) + list(enc.linear.parameters())
optimizer = torch.optim.Adam(params, lr=learing_rate)

for e in range(epochs):
    for bi in data_gen.n:
    	#get batch 
        imb,lbb,lnb=data_gen.get_batch()

        #load variables to gpu
        imb=Variable(torch.from_numpy(imb)).float().cuda(2)
        lbb=Variable(torch.from_numpy(lbb)).cuda(2)
        lnb=Variable(torch.from_numpy(lnb)).cuda(2)

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

       	print('> Epoch :',e+1,' Batch :',bi+1,' Loss :',loss.data[0])
