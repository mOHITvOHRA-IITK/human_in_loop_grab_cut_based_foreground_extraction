import cv2
import numpy as np
import os
from os import path
from functions import*




	

data_folder_path = './images/'
all_images_list = os.listdir(data_folder_path)


save_folder_path = './foreground_mask/'



for file in all_images_list:
	cmp_img_path = data_folder_path + file
	img = cv2.imread(cmp_img_path)
	# img2 = cv2.resize(img, (192,256))
	# cv2.imwrite(cmp_img_path, img2)
	mask = generate_mask(img, True)
	cmp_mask_path = save_folder_path + file
	cv2.imwrite(cmp_mask_path, mask)
	
	
print ('***************************************')
print ('***************Thank you***************')
print ('***************************************')


