import cv2
import os
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import json


def parse_args():
	parser = argparse.ArgumentParser(description = "No Desc")
	parser.add_argument("--images", type = str, required = True)
	parser.add_argument("--labels", type = str, required = True)
	parser.add_argument("--output", type = str, required = True)
	args = parser.parse_args()
	return args

def load_data(im_path,labels_path):
	# Load images to Numpy array
	X=[]
	im_shape=cv2.imread(im_path+"/"+os.listdir(im_path)[0]).shape
	files=[]
	for f in tqdm(os.listdir(im_path),total=len(os.listdir(im_path))):
		im=cv2.imread(im_path+'/'+f)
		if im is not None:
			# im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
			im=cv2.resize(im,(224,224))
			files.append(f.split('.')[0])
			X.append(im)
	X=np.array(X)
	im_shape=[224,224,im_shape[2]]
	X=X.reshape(X.shape[0],im_shape[0],im_shape[1],im_shape[2])
	print('> Loading Images Done ')
	print('> Discarded Images :',len(os.listdir(im_path))-len(files))
	# Load lables
	labels_list=[]
	label_to_id={}

	def put_to_dict(x):
		resp=[]
		for e in x:
			if e not in labels_list:
				labels_list.append(e)
				label_to_id[e]=len(labels_list)-1
			resp.append(label_to_id[e])
		return np.array(resp)

	labels=pd.read_csv(labels_path)
	labels['tags']=labels['tags'].apply(lambda x:x.split())

	f_names=pd.DataFrame(files)
	f_names.columns=['image_name']

	labels['tags']=labels['tags'].apply(put_to_dict)
	labels=pd.merge(f_names,labels)

	Y=np.array(labels['tags'].values)
	return X,Y,label_to_id

def main():
	args=parse_args()
	X,Y,l_dict=load_data(args.images,args.labels)
	np.save(args.output+'/'+'ims.npy',X)
	np.save(args.output+'/'+'labels.npy',Y)
	with open(args.output+'/'+'label_dict.json', 'w') as fp:
		json.dump(l_dict, fp)
	print('> Done and Saved to file')

if __name__=='__main__':
	main()