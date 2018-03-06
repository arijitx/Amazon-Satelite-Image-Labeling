import numpy as np 


def padd_labels(labels):
	new_labels=[]
	lengths=np.zeros((labels.shape[0],1))
	max_size=0
	PADD_TOKEN=17
	SOS_TOKEN=18
	EOS_TOKEN=19
	for i in range(labels.shape[0]):
		max_size=max(labels[i].shape[0],max_size)
	for i in range(labels.shape[0]):
		lengths[i]=labels[i].shape[0]
		padded_label=np.concatenate((np.array([SOS_TOKEN]),labels[i],np.array([EOS_TOKEN]),np.ones(max_size-labels[i].shape[0])*PADD_TOKEN))
		new_labels.append(padded_label)
	padded_labels=np.array(new_labels).reshape((len(new_labels),max_size+2))
	return padded_labels,lengths


lzp,lengths=padd_labels(np.load('data/labels.npy'))
print(lzp.shape)
print(lzp[:5])