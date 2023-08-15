import os 
import pandas as pd
from typing import List
directory = '/home/phn501/plot-finder/dataset/range_wise/test' 
outputdir = '/home/phn501/plot-finder/dataset/range_wise/test'


def assign_score(df:pd.DataFrame) -> List[int]: 
	'''
		assign importance score for every frame
		param: 
			df: pandas dataframe of the csv file

		return: 
			imp_score: list of importance score of each image
	'''

	row_count	= len(df)
	imp_score 	= [0]*row_count
	
	for index, row in df.iterrows():
		if row['class'] == 1: 
			# set the current frame as 1
			imp_score[index] = 1.0

			for i in range(1, 4):
				# next element: value range will be 0.25, 0.5, 0.75
				if index+i < row_count: 
					imp_score[index+i] = max(imp_score[index+i], 1 - i/4)  

				# previous element: value range will be 0.25, 0.5, 0.75
				if 0 <= index-i: 
					imp_score[index-i] = max(imp_score[index-i], 1 - i/4)  

	return imp_score



def get_files(directory: str): 
	'''
		iterate over the root folder and append impotance score and resave the csv files
		param: 
			directory(str) : root directory containing all the csv files
	'''
	if not os.path.exists(outputdir): 
		os.mkdir(outputdir)

	for i, file in enumerate(os.listdir(directory)): 
		if file.endswith('.csv'):
			print(file)
			file_path   = os.path.join(directory, file)

			df 	= pd.read_csv(file_path)
			# df 	= df.shift(periods=1, axis=1)
			# df 	= df.drop('year', axis=1)
			# df  = df.reset_index().rename(columns={'index': 'year'})

			imp_score 	= assign_score(df)
			
			df['score'] = imp_score
			

			output_file = os.path.join(outputdir, file)
			df.to_csv(output_file, index=False)
				

get_files(directory)


