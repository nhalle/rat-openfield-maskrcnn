from images_to_video import *
from rat_maskrcnn import *


# load the test dataset
train_set = RatDataset()
train_set.load_dataset('rat')
train_set.prepare()
print('Test: %d' % len(train_set.image_ids))

test_set = RatDataset()
test_set.load_dataset('rattest')
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model path
model_path = 'mask_rcnn_rat_cfg_0005.h5'
# load model weights
model.load_weights(model_path, by_name=True)

# evaluate model on test dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)

# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)
