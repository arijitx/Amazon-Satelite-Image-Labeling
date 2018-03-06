import numpy as np 
import math 

def padd_labels(labels):
	new_labels_train=[]
	new_labels_target=[]
	lengths=np.zeros((labels.shape[0],1))
	max_size=0
	PADD_TOKEN=17
	# SOS_TOKEN=18
	EOS_TOKEN=18
	for i in range(labels.shape[0]):
		max_size=max(labels[i].shape[0],max_size)
	for i in range(labels.shape[0]):
		lengths[i]=labels[i].shape[0]+1
		padded_label_train=np.concatenate((labels[i],np.ones(max_size-labels[i].shape[0])*PADD_TOKEN,[EOS_TOKEN]))
		padded_label_target=np.concatenate((labels[i],np.ones(max_size-labels[i].shape[0])*PADD_TOKEN))
		new_labels_train.append(padded_label_train)
		new_labels_target.append(padded_label_target)

	sorted_lengths_i=lengths.flatten().argsort()

	padded_labels_train=np.array(new_labels_train).reshape((len(new_labels_train),max_size+1))
	padded_labels_target=np.array(new_labels_target).reshape((len(new_labels_target),max_size))
	lengths=lengths[sorted_lengths_i[::-1]]
	padded_labels_train=padded_labels_train[sorted_lengths_i[::-1]]
	padded_labels_target=padded_labels_target[sorted_lengths_i[::-1]]
	return padded_labels_train.astype('int'),lengths.astype('int')

def load_labels(label_path):
	return padd_labels(np.load(label_path))

def load_images(images_path):
	images=np.load(images_path)
	ims=images.shape
	images=images/255.
	images=images.reshape((ims[0],ims[3],ims[1],ims[2]))
	return images

def one_hot_labels(labels,vocab_size):
	one_hot_encoded=[]
	for i in range(labels.shape[0]):
		a=labels[i]
		a=a.astype('int')
		b=np.zeros((labels[i].shape[0],vocab_size))
		b[np.arange(int(labels[i].shape[0])),a]=1.
		one_hot_encoded.append(b)
	one_hot_encoded=np.array(one_hot_encoded)
	one_hot_encoded=one_hot_encoded.reshape((one_hot_encoded.shape[0],labels[0].shape[0],vocab_size))
	return one_hot_encoded

class DataGen():
	def __init__(self,im_path,labels_path,batch_size=256):
		self.im=load_images(im_path)
		print('> Loading Images Done .')
		self.labels,self.lengths=load_labels(labels_path)
		self.batch_size=batch_size
		self.idx=0
		self.n=range(math.floor(self.im.shape[0]/self.batch_size))

	def get_batch(self):
		im_batch=self.im[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
		lbl_batch=self.labels[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
		lens_batch=self.lengths[self.idx*self.batch_size:(self.idx+1)*self.batch_size]
		self.idx+=1
		return im_batch,lbl_batch,lens_batch

# train,target,lens=load_labels('data/labels.npy')
# print(train[1])
# print(target[1])
# print(lens)