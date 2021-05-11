import cv2
import numpy as np


all_foreground_pixels = []
all_background_pixels = []

target_image_region = None

foreground_region_mask = None
foreground_region_selected = False

background_region_mask = None
background_region_selected = False

exit_region_mask = None
exit_button_selected = False

process_region_mask = None
process_button_selected = False

eraser_region_mask = None
eraser_button_selected = False

def get_mouse_click(event,x,y,flags,param):
	global all_foreground_pixels, all_background_pixels, foreground_region_mask, foreground_region_selected, background_region_mask, background_region_selected, exit_region_mask, exit_button_selected, process_region_mask, process_button_selected, target_image_region, eraser_region_mask, eraser_button_selected

	if event == cv2.EVENT_LBUTTONDOWN:

		
		if ( (x > foreground_region_mask[2]) & (x < foreground_region_mask[3]) & (y > foreground_region_mask[0]) & (y < foreground_region_mask[1]) ):
			foreground_region_selected = True
			background_region_selected = False
			exit_button_selected = False
			process_button_selected = False
			eraser_button_selected = False


		if ( (x > background_region_mask[2]) & (x < background_region_mask[3]) & (y > background_region_mask[0]) & (y < background_region_mask[1]) ):
			foreground_region_selected = False
			background_region_selected = True
			exit_button_selected = False
			process_button_selected = False
			eraser_button_selected = False


		if ( (x > process_region_mask[2]) & (x < process_region_mask[3]) & (y > process_region_mask[0]) & (y < process_region_mask[1]) ):
			foreground_region_selected = False
			background_region_selected = False
			exit_button_selected = False
			process_button_selected = True
			eraser_button_selected = False


		if ( (x > exit_region_mask[2]) & (x < exit_region_mask[3]) & (y > exit_region_mask[0]) & (y < exit_region_mask[1]) ):
			foreground_region_selected = False
			background_region_selected = False
			exit_button_selected = True
			process_button_selected = False
			eraser_button_selected = False


		if ( (x > eraser_region_mask[2]) & (x < eraser_region_mask[3]) & (y > eraser_region_mask[0]) & (y < eraser_region_mask[1]) ):
			foreground_region_selected = False
			background_region_selected = False
			exit_button_selected = False
			process_button_selected = False
			eraser_button_selected = True



		if ( (x > target_image_region[2]) & (x < target_image_region[3]) & (y > target_image_region[0]) & (y < target_image_region[1]) ):

			if foreground_region_selected:
				all_foreground_pixels.append((x,y))

			if background_region_selected:
				all_background_pixels.append((x,y))

			
			if eraser_button_selected:
				for i in range(len(all_foreground_pixels)):
					x_diff = x - all_foreground_pixels[i][0]
					y_diff = y - all_foreground_pixels[i][1]

					dis = np.sqrt(x_diff*x_diff + y_diff*y_diff)
					if dis < 10:
						del all_foreground_pixels[i]
						break


				for i in range(len(all_background_pixels)):
					x_diff = x - all_background_pixels[i][0]
					y_diff = y - all_background_pixels[i][1]

					dis = np.sqrt(x_diff*x_diff + y_diff*y_diff)
					if dis < 10:
						del all_background_pixels[i]
						break






	
        

def generate_mask(img, visualize = False):
	global all_foreground_pixels, all_background_pixels, foreground_region_mask, foreground_region_selected, background_region_mask, background_region_selected, exit_region_mask, exit_button_selected, process_region_mask, process_button_selected, target_image_region, eraser_region_mask, eraser_button_selected
	all_foreground_pixels = []
	all_background_pixels = []

	foreground_region_mask = None
	foreground_region_selected = False

	background_region_mask = None
	background_region_selected = False

	exit_region_mask = None
	exit_button_selected = False

	process_region_mask = None
	process_button_selected = False

	eraser_region_mask = None
	eraser_button_selected = False

	cv2.namedWindow('annotation')
	cv2.setMouseCallback('annotation',get_mouse_click)

	h,w,_ = img.shape
	big_frame = np.zeros([h,np.int(1.4*w),3], np.uint8)
	big_frame[0:h, 0:w, :] = img

	big_frame[np.int(0.1*h):np.int(0.2*h), np.int(1.1*w):np.int(1.3*w),:] = [0, 255, 0]  # Foreground region
	big_frame[np.int(0.25*h):np.int(0.35*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 0, 0]  # background region
	big_frame[np.int(0.4*h):np.int(0.5*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 255, 255]  # Process region
	big_frame[np.int(0.55*h):np.int(0.65*h), np.int(1.1*w):np.int(1.3*w),:] = [0, 0, 255]  # Exit region
	big_frame[np.int(0.7*h):np.int(0.8*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 0, 255]  # Eraser region

	target_image_region = [0,h,0,w]
	foreground_region_mask = [np.int(0.1*h), np.int(0.2*h), np.int(1.1*w), np.int(1.3*w)]
	background_region_mask = [np.int(0.25*h), np.int(0.35*h), np.int(1.1*w), np.int(1.3*w)]
	process_region_mask = [np.int(0.4*h), np.int(0.5*h), np.int(1.1*w), np.int(1.3*w)]
	exit_region_mask = [np.int(0.55*h), np.int(0.65*h), np.int(1.1*w), np.int(1.3*w)]
	eraser_region_mask = [np.int(0.7*h), np.int(0.8*h), np.int(1.1*w), np.int(1.3*w)]

	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect = (1,1,img.shape[1]-1,img.shape[0]-1)
	cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
	mask2 = 255*np.where((mask==2)|(mask==0),0,1).astype('uint8')
	
	org_img = img.copy()


	while(exit_button_selected == False):

		img = org_img.copy()
		user_input_mask = 127*np.ones(img.shape[:2],np.uint8)

		for i in range(len(all_foreground_pixels)):
			cv2.circle(img, all_foreground_pixels[i], 10, (0,255,0), -1)
			cv2.circle(user_input_mask, all_foreground_pixels[i], 10, 255, -1) # sure foreground


		for i in range(len(all_background_pixels)):
			cv2.circle(img, all_background_pixels[i], 10, (255,0,0), -1)
			cv2.circle(user_input_mask, all_background_pixels[i], 10, 0, -1) # sure background

		
		big_frame = np.zeros([h,np.int(1.4*w),3], np.uint8)
		big_frame[0:h, 0:w, :] = img

		big_frame[np.int(0.1*h):np.int(0.2*h), np.int(1.1*w):np.int(1.3*w),:] = [0, 255, 0]  # Foreground region
		big_frame[np.int(0.25*h):np.int(0.35*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 0, 0]  # background region
		big_frame[np.int(0.4*h):np.int(0.5*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 255, 255]  # Process region
		big_frame[np.int(0.55*h):np.int(0.65*h), np.int(1.1*w):np.int(1.3*w),:] = [0, 0, 255]  # Exit region
		big_frame[np.int(0.7*h):np.int(0.8*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 0, 255]  # Eraser region


		if process_button_selected:
			
			mask[user_input_mask == 0] = 0
			mask[user_input_mask == 255] = 1
			local_mask, bgdModel, fgdModel = cv2.grabCut(org_img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
			mask2 = 255*np.where((local_mask==2)|(local_mask==0),0,1).astype('uint8')


		cv2.imshow('foreground_mask', mask2)
		cv2.imshow('annotation', big_frame)
		k = cv2.waitKey(10)

	return mask2

	




def generate_final_image(img, mask):
	background_pixel_loc = mask[:,:] == 0
	only_foreground_img = img.copy()
	only_foreground_img[background_pixel_loc,:] = [255,255,255]
	cv2.imshow('img', img)
	cv2.imshow('mask', mask)
	cv2.imshow('only_foreground_img', only_foreground_img)
	cv2.waitKey(0)

	return only_foreground_img





