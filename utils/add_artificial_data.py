import pandas as pd 
import numpy as np 
import random 
import csv
import os

import data_split 


class AddData: 
    def __init__(self, data, insertion_count, csv_dir, save_dir) -> None:
        self.insertcount = insertion_count
        self.year        = data["year"]
        self.trial       = data["trial"]
        self.date        = data["date"]
        self.camera      = data["camera"]
        self.range_      = data["range_"]
        self.class_      = data["class_"]
        self.position    = data["position"]
        self.img_path    = str(self.year)+"/"+self.trial+"/"+self.date+"/"+self.camera+"/"+self.range_
        self.score       = data["score"]
        self.files2process   = data["files2process"]
        print(data["trial"])
        self.NEW_IMG_DIR    = f"/home/phn501/plot-finder/dataset/images/{self.year}/{self.trial}/{self.date}/{self.camera}/{self.range_}"
        self.CSV_DIR        = csv_dir
        self.SAVE_DIR       = save_dir

        self.create_dir()
        self.images = self.get_images()

    def create_dir(self): 
        if not os.path.exists(self.SAVE_DIR):
            os.mkdir(self.SAVE_DIR)


    def get_images(self):
        ''' read the images to be injected and return them in a list format ''' 
        images = [] 
        print(self.NEW_IMG_DIR)
        for file in os.listdir(self.NEW_IMG_DIR): 
            if file.endswith('.JPG'):
                images.append(file)
        
        images.sort()
        return images


    def insert_aritifical_data(self): 
        """ insert images from other dataset and save it as csv """

        for file in os.listdir(self.CSV_DIR): 
            data = [] 
            # generate new random for each file so you have diversity of the image insertion
            inject_index = set(random.sample(range(1, 800), self.insertcount))
            img_index    = random.sample(range(0,100), self.insertcount)
            counter = 0
            if file.endswith('.csv') and file in self.files2process: 
                with open(os.path.join(self.CSV_DIR, file), 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    for index, row in enumerate(reader):

                        # inject the new data 
                        if index in inject_index: 
                            data.append([self.year, self.trial, self.date, self.camera, 
                                        self.range_, self.images[img_index[counter]], 
                                        self.class_, self.position, self.img_path, 
                                        self.score])
                            counter += 1

                        #copy the old data 
                        data.append(row)
                
                name    = os.path.join(self.SAVE_DIR, f"{self.insertcount}_artificial_{file}")
                df      = pd.DataFrame(data)

                df.to_csv(name, index=False, header=False)
                print(f"processed: {file}")


INSERTION_COUNT = [2] # number of rows with artificial data
CSV_DIR        = "/home/phn501/plot-finder/dataset/score_data/"
SAVE_DIR    = "/home/phn501/plot-finder/dataset/score_data/"

for count in INSERTION_COUNT: 
    for data in data_split.info["validation"]: 
        val_csv_dir     = os.path.join(CSV_DIR, "validation")
        val_save_dir    = os.path.join(SAVE_DIR, "artificial_validation/", str(count))
        inserter = AddData(data, count, val_csv_dir, val_save_dir)
        inserter.insert_aritifical_data()


    # for data in data_split.info["train"]: 
    #     csv_dir     = os.path.join(CSV_DIR,"train")
    #     save_dir    = os.path.join(SAVE_DIR, "artificial_train/", str(count))
    #     inserter = AddData(data, count, csv_dir, save_dir)
    #     inserter.insert_aritifical_data()
        
