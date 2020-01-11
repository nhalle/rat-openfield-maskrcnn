# detect rats in photos with mask rcnn model
from os import listdir
from os import mkdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.model import load_image_gt
from mrcnn.utils import compute_ap
from mrcnn.utils import Dataset
from images_to_video import video_to_images
from images_to_video import images_to_video
from datetime import datetime
from progress.bar import ChargingBar

# class that defines and loads the rat dataset
class RatDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir):
		# define one class
		self.add_class("dataset", 1, "rat")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]

			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'

			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	#load a video to analyze
	def load_video(self, video_name, video_path, max_frames=300):
		dataset_dir = './'+ video_name
		try:
			mkdir(dataset_dir)
		except OSError:
			print ("Creation of the directory %s failed" % dataset_dir)
		else:
			print ("Successfully created the directory %s " % dataset_dir)

		video_to_images(video_path, dataset_dir, max_frames)

		#define one class
		self.add_class("dataset", 1, "rat")

		# define data locations
		images_dir = dataset_dir + '/'
		lst = listdir(images_dir)
		lst.sort()
		# find all images
		for filename in lst:
			# extract image id
			image_id = filename[:-4]

			img_path = images_dir + filename

			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path)

		return max_frames



	def load_frames(self, images_dir):
		# define one class
		self.add_class("dataset", 1, "rat")
		# define data locations

		lst = listdir(images_dir)
		lst.sort()
		# find all images
		for filename in lst:
			# extract image id
			image_id = filename[:-4]

			img_path = images_dir + filename

			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path)
	# load all bounding boxes for an image
	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename)
		boxes = list()
		# extract each bounding box
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('rat'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "rat_cfg"
	# number of classes (background + rat)
	NUM_CLASSES = 1 + 1
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

# plot predictions
def plot_predicted(dataset, model, cfg, n_images):

	results_dir = './results' + datetime.now().strftime('%Y%m%d_%H%M%S')
	print(results_dir)
	try:
	    mkdir(results_dir)
	except OSError:
	    print ("Creation of the directory %s failed" % results_dir)
	else:
	    print ("Successfully created the directory %s " % results_dir)

	images_dir = results_dir + '/images'
	mkdir(images_dir)

	#save the center for each rat
	plot_center = []

	center_x = []
	center_y = []
	# load images
	bar = ChargingBar('Processing', max=n_images)
	for i in range(n_images):
		# load the image
		image = dataset.load_image(i)

		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]

		# plot raw pixel data
		pyplot.imshow(image)

		ax = pyplot.gca()

		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			#calulate midpoint
			midx, midy = (x1+x2)/2, (y1+y2)/2

			#print("this is (x,y) midx midy  (" + str(midx) + ',' + str(midy) + ")")
			#Add midpoint to the plot_center list
			center_x.append(midx)
			center_y.append(midy)
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box to image
			ax.add_patch(rect)

		pyplot.axis('off')
		# keep image paths ordered for video processing
		if(i < 10):
			save_image_path = images_dir+ '/0000' + str(i) + '.jpeg'
		elif(i<100):
			save_image_path = images_dir+ '/000' + str(i) + '.jpeg'
		elif(i<1000):
			save_image_path = images_dir+ '/00' + str(i) + '.jpeg'
		elif(i<10000):
			save_image_path = images_dir+ '/0' + str(i) + '.jpeg'

		pyplot.savefig(save_image_path, bbox_inches='tight')
		pyplot.clf()
		bar.next()

	# show the figure
	bar.finish()

	pyplot.scatter(center_x,center_y)
	path_dir = results_dir + '/path.png'
	pyplot.savefig(path_dir)
	pyplot.clf()

	#set path for video
	pathIn = images_dir + '/'
	pathOut = results_dir+ '/test_video.avi'
	# set frames per second
	fps = 5
	# Convert images to video
	images_to_video(pathIn,pathOut,fps)
	print("Check directory: " + results_dir)
	return results_dir

# calculates the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = mean(APs)
	return mAP
