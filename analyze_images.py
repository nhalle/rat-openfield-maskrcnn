"""
Analyze images
Method for prexisting frames
type: python analyze_images <images_directory>

ex:
python analyze_images ./ratvideo
"""
from images_to_video import *
from rat_maskrcnn import *
import sys

image_path = sys.argv[1]
# load test dataset
test_set = RatDataset()

frames = test_set.load_frames(image_path)

test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model_path = 'mask_rcnn_rat_cfg_0005.h5'
model.load_weights(model_path, by_name=True)
#
# plot predictions for test dataset
results_path = plot_predicted(test_set, model, cfg, 300)

pathIn = results_path + '/images/'
#set paths and fps for images to video conversion
pathOut = results_path + '/test_video.avi'
# set frames per second
fps = 5
# Convert images to video
images_to_video(pathIn,pathOut,fps)
