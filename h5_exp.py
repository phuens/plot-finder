import h5py
import json 

hdf = h5py.File('/home/phuntsho/Desktop/plot-finder/plot-finder/2048_feature.h5', 'r')


with open("/home/phuntsho/Desktop/plot-finder/plot-finder/predict/extratced_feature/plotreel_video_names.json") as f:
    videos = json.loads(f.read())

for video in videos['test_keys']: 
    print(video)