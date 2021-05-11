import cv2
import numpy as np
import os
from os import path
from functions import*




	

data_folder_path = './images/'
all_images_list = os.listdir(data_folder_path)

save_folder_path = './foreground_mask/'
final_image_folder_path = './final_image/'


for file in all_images_list:
	cmp_img_path = data_folder_path + file
	print ('reading file', cmp_img_path)
	img = cv2.imread(cmp_img_path)

	mask = generate_mask(img, True)
	cmp_mask_path = save_folder_path + file
	cv2.imwrite(cmp_mask_path, mask)

	only_foreground_img = generate_final_image(img, mask)
	only_foreground_img = cv2.resize(only_foreground_img, (192,256))
	cmp_img_path = final_image_folder_path + file
	cv2.imwrite(cmp_img_path, only_foreground_img)
	
	
print ('***************************************')
print ('***************Thank you***************')
print ('***************************************')


