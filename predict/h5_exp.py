import h5py
import json 
import numpy as np

hdf = h5py.File('/home/phn501/plot-finder/dataset/features/hsv_training_feature_imagenet.h5', 'r')


with open("/home/phn501/plot-finder/dataset/features/plotreel_video_names.json") as f:
    videos = json.loads(f.read())

for video_name in videos['train_keys']:
    frame_features = np.array(hdf[video_name + '/gtscore'])

    # gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
    print(frame_features)