import cv2
import time
import numpy as np
import os
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset
from skimage.morphology import skeletonize




image_saved_path = '/target_images/'
if os.path.isdir(os.getcwd() + image_saved_path) == False:
	os.mkdir(os.getcwd() + image_saved_path)


process_image_path = '/process_images/'
if os.path.isdir(os.getcwd() + process_image_path) == False:
	os.mkdir(os.getcwd() + process_image_path)


num_classes = 7
model = networks.init_model('resnet101', num_classes=num_classes, pretrained=None)
state_dict = torch.load(os.getcwd() + '/weights/' + 'exp-schp-201908270938-pascal-person-part.pth')['state_dict']
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.cuda()
model.eval()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette



def process_image():
	
	input_size = [512, 512]
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])])
	dataset = SimpleFolderDataset(root=os.getcwd() + image_saved_path, input_size=input_size, transform=transform)
	dataloader = DataLoader(dataset)

	palette = get_palette(num_classes)

	with torch.no_grad():
		for idx, batch in enumerate(tqdm(dataloader)):
			image, meta = batch
			img_name = meta['name'][0]
			c = meta['center'].numpy()[0]
			s = meta['scale'].numpy()[0]
			w = meta['width'].numpy()[0]
			h = meta['height'].numpy()[0]

			output = model(image.cuda())
			upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
			upsample_output = upsample(output[0][-1][0].unsqueeze(0))
			upsample_output = upsample_output.squeeze()
			upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

			logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
			parsing_result = np.argmax(logits_result, axis=2)
			parsing_result_path = os.path.join(os.getcwd() + process_image_path, img_name[:-4] + '.png')
			output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
			output_img.putpalette(palette)
			output_img.save(parsing_result_path)



def get_body_skeleton(human_seg_coloured_mask):

	human_seg_binary_mask = 0*human_seg_coloured_mask
	x =  ((human_seg_coloured_mask[:,:,0] > 0) | (human_seg_coloured_mask[:,:,1] > 0) | (human_seg_coloured_mask[:,:,2] > 0))
	human_seg_binary_mask[x, :] = [255, 255, 255]

	skeleton = skeletonize(human_seg_binary_mask)
	x2 = (skeleton[:,:,0] > 0) | (skeleton[:,:,1] > 0) | (skeleton[:,:,2] > 0)
	skeleton[x2] = [255, 255, 255]
	skeleton = np.array(skeleton, np.uint8)

	background_part = 255*np.ones(human_seg_coloured_mask.shape, np.uint8)
	background_part[x, :] = [0, 0, 0]
	kernel = np.ones((5,5),np.uint8)
	background_part = cv2.erode(background_part, kernel ,iterations = 1)
	return skeleton, background_part




all_foreground_pixels = []
all_foreground_radius_list = []

all_background_pixels = []
all_background_radius_list = []

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

circle_region_mask = None
circle_radius = 10



def get_mouse_click(event,x,y,flags,param):
	global all_foreground_pixels, all_background_pixels, foreground_region_mask, foreground_region_selected, background_region_mask, background_region_selected, exit_region_mask, exit_button_selected, process_region_mask, process_button_selected, target_image_region, eraser_region_mask, eraser_button_selected, circle_radius, circle_region_mask, all_foreground_radius_list, all_background_radius_list

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


		if ( (x > circle_region_mask[2]) & (x < circle_region_mask[3]) & (y > circle_region_mask[0]) & (y < circle_region_mask[1]) ):
			circle_radius += 1



		if ( (x > target_image_region[2]) & (x < target_image_region[3]) & (y > target_image_region[0]) & (y < target_image_region[1]) ):

			if foreground_region_selected:
				all_foreground_pixels.append((x,y))
				all_foreground_radius_list.append(circle_radius)

			if background_region_selected:
				all_background_pixels.append((x,y))
				all_background_radius_list.append(circle_radius)

			
			if eraser_button_selected:
				for i in range(len(all_foreground_pixels)):
					x_diff = x - all_foreground_pixels[i][0]
					y_diff = y - all_foreground_pixels[i][1]

					dis = np.sqrt(x_diff*x_diff + y_diff*y_diff)
					if dis < circle_radius:
						del all_foreground_pixels[i]
						del all_foreground_radius_list[i]
						break


				for i in range(len(all_background_pixels)):
					x_diff = x - all_background_pixels[i][0]
					y_diff = y - all_background_pixels[i][1]

					dis = np.sqrt(x_diff*x_diff + y_diff*y_diff)
					if dis < circle_radius:
						del all_background_pixels[i]
						del all_background_radius_list[i]
						break


	if event == cv2.EVENT_RBUTTONDOWN:
		if ( (x > circle_region_mask[2]) & (x < circle_region_mask[3]) & (y > circle_region_mask[0]) & (y < circle_region_mask[1]) ):
			circle_radius -= 1

			if circle_radius < 1:
				circle_radius = 1


def step_1_grabcut_intialization_with_bbox(img):
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	rect = (1,1,img.shape[1]-1,img.shape[0]-1)
	cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
	return mask, bgdModel, fgdModel



def step_2_get_targte_object_pixels(img, mask, bgdModel, fgdModel, file):
	cmp_targte_img_path = os.getcwd() + image_saved_path + file
	cv2.imwrite(cmp_targte_img_path, img)
	process_image()
	os.remove(cmp_targte_img_path) 

	cmp_human_seg_coloured_mask_path = os.getcwd() + process_image_path + file.split('.')[0] + '.png'
	human_seg_coloured_mask = cv2.imread(cmp_human_seg_coloured_mask_path)
	skeleton, background_part = get_body_skeleton(human_seg_coloured_mask)

	user_input_mask = 127*np.ones(img.shape[:2],np.uint8)
	forground_pixels =  (skeleton[:,:,0] > 0) 
	user_input_mask[forground_pixels] = 255 # sure foreground

	background_pixels =  (background_part[:,:,0] > 0) 
	user_input_mask[background_pixels] = 0 # sure background

	# cv2.imshow('human_seg_coloured_mask', human_seg_coloured_mask)
	# cv2.imshow('skeleton', skeleton)
	# cv2.imshow('background_part', background_part)
	# cv2.imshow('user_input_mask', user_input_mask)
	# cv2.waitKey(0)

	mask[user_input_mask == 0] = 0
	mask[user_input_mask == 255] = 1
	mask, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

	return mask, bgdModel, fgdModel

	

def step_3_user_marking_for_grab_cut(img, mask, bgdModel, fgdModel, mask2):
	global all_foreground_pixels, all_background_pixels, foreground_region_mask, foreground_region_selected, background_region_mask, background_region_selected, exit_region_mask, exit_button_selected, process_region_mask, process_button_selected, target_image_region, eraser_region_mask, eraser_button_selected, circle_radius, circle_region_mask, all_foreground_radius_list, all_background_radius_list
	all_foreground_pixels = []
	all_foreground_radius_list = []

	all_background_pixels = []
	all_background_radius_list = []

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

	circle_region_mask = None
	circle_radius = 10

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
	cv2.circle(big_frame, (np.int(1.2*w), np.int(0.9*h)), circle_radius, (255,255,0), -1)
	# print ('circle center', (np.int(0.9*h), np.int(1.2*w)))
	

	target_image_region = [0,h,0,w]
	foreground_region_mask = [np.int(0.1*h), np.int(0.2*h), np.int(1.1*w), np.int(1.3*w)]
	background_region_mask = [np.int(0.25*h), np.int(0.35*h), np.int(1.1*w), np.int(1.3*w)]
	process_region_mask = [np.int(0.4*h), np.int(0.5*h), np.int(1.1*w), np.int(1.3*w)]
	exit_region_mask = [np.int(0.55*h), np.int(0.65*h), np.int(1.1*w), np.int(1.3*w)]
	eraser_region_mask = [np.int(0.7*h), np.int(0.8*h), np.int(1.1*w), np.int(1.3*w)]
	circle_region_mask = [np.int(0.85*h), np.int(0.95*h), np.int(1.1*w), np.int(1.3*w)]


	org_img = img.copy()


	while(exit_button_selected == False):

		img = org_img.copy()
		user_input_mask = 127*np.ones(img.shape[:2],np.uint8)

		for i in range(len(all_foreground_pixels)):
			cv2.circle(img, all_foreground_pixels[i], all_foreground_radius_list[i], (0,255,0), -1)
			cv2.circle(user_input_mask, all_foreground_pixels[i], all_foreground_radius_list[i], 255, -1) # sure foreground


		for i in range(len(all_background_pixels)):
			cv2.circle(img, all_background_pixels[i], all_background_radius_list[i], (255,0,0), -1)
			cv2.circle(user_input_mask, all_background_pixels[i], all_background_radius_list[i], 0, -1) # sure background

		
		big_frame = np.zeros([h,np.int(1.4*w),3], np.uint8)
		big_frame[0:h, 0:w, :] = img

		big_frame[np.int(0.1*h):np.int(0.2*h), np.int(1.1*w):np.int(1.3*w),:] = [0, 255, 0]  # Foreground region
		big_frame[np.int(0.25*h):np.int(0.35*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 0, 0]  # background region
		big_frame[np.int(0.4*h):np.int(0.5*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 255, 255]  # Process region
		big_frame[np.int(0.55*h):np.int(0.65*h), np.int(1.1*w):np.int(1.3*w),:] = [0, 0, 255]  # Exit region
		big_frame[np.int(0.7*h):np.int(0.8*h), np.int(1.1*w):np.int(1.3*w),:] = [255, 0, 255]  # Eraser region
		cv2.circle(big_frame, (np.int(1.2*w), np.int(0.9*h)), circle_radius, (255,255,0), -1)



		if process_button_selected:
			
			mask[user_input_mask == 0] = 0
			mask[user_input_mask == 255] = 1
			local_mask, bgdModel, fgdModel = cv2.grabCut(org_img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
			mask2 = 255*np.where((local_mask==2)|(local_mask==0),0,1).astype('uint8')


		cv2.imshow('foreground_mask', mask2)
		cv2.imshow('annotation', big_frame)
		k = cv2.waitKey(10)

	return mask2




def get_foreground_mask(img, file):
	
	mask, bgdModel, fgdModel = step_1_grabcut_intialization_with_bbox(img)

	mask, bgdModel, fgdModel = step_2_get_targte_object_pixels(img, mask, bgdModel, fgdModel, file)
	mask2 = 255*np.where((mask==2)|(mask==0),0,1).astype('uint8')

	mask2 = step_3_user_marking_for_grab_cut(img, mask, bgdModel, fgdModel, mask2)

	return mask2




def generate_final_image(img, mask):

	h,w,_ = img.shape
	concat_img = np.zeros([h,np.int(2*w),3], np.uint8)
	concat_img[0:h, 0:w, :] = img
	concat_img[0:h, w:2*w, 0] = mask
	concat_img[0:h, w:2*w, 1] = mask
	concat_img[0:h, w:2*w, 2] = mask


	background_pixel_loc = mask[:,:] == 0
	only_foreground_img = img.copy()
	only_foreground_img[background_pixel_loc,:] = [255,255,255]


	cv2.imshow('concat_img', concat_img)
	cv2.waitKey(0)

	return only_foreground_img, concat_img


