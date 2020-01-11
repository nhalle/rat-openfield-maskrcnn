"""
analyze_video.py takes in 3 parameters video path, video name (will be used to create a
directory to store frames), and the number of max
the user would like to analyze.

ex:
python analyze_video.py <video_path> <video_name> <nframes>
python analyze_video.py ./videos/video1.mp4 openfield1 20

"""

from images_to_video import *
from rat_maskrcnn import *
import sys
import shutil
# define path of video
#video_path = './videos/openfield2.mp4'
video_path = sys.argv[1]
# define video name
#video_name = 'openfield2'
video_name = sys.argv[2]
# define number of videos
#nframes  = 20
nframes = int(sys.argv[3])
# load test dataset
test_set = RatDataset()

frames = test_set.load_video(video_name, video_path, nframes)
# test_set.load_raw_dataset('ratvideo')
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
results_path = plot_predicted(test_set, model, cfg, nframes)

frames_dir = "./" + video_name
shutil.rmtree(frames_dir)
