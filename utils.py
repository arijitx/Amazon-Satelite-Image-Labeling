import cv2
import torch
import numpy as np

def draw_image(attn,ims,preds):
	for i in range(attn.size(0)):
		im=cv2.imread('dataset/test/images/'+ims[i]+'.jpg')
		attn_i=attn[i].cpu().data.numpy()
		imz=np.ones((224,224*(len(preds[i].split(' '))+1),3))
		imz[0:224,0:224,:]=cv2.resize(im,(224,224))

		for j,pred in enumerate(preds[i].split(' ')):
			min_at=np.min(attn_i[j])
			max_at=np.max(attn_i[j])
			attn_i[j]=(attn_i[j]-min_at)/(max_at-min_at)
			attn_i[j]*=255
			attn_ij=np.reshape(attn_i[j],(7,7))
			attn_viz=cv2.resize(attn_ij,(224,224))
			imz[0:224,(j+1)*224:(j+2)*224,:]=cv2.cvtColor(attn_viz,cv2.COLOR_GRAY2BGR)
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(imz,pred,((j+1)*224+10,180), font, 1,(255,255,255),1,cv2.LINE_AA)

		cv2.imwrite('results/'+ims[i]+'.jpg',imz)