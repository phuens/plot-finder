import os
import csv
import pandas as pd
import sklearn.metrics as metrics 
import numpy as np


def read_file(filename):
	file_dataframe = pd.read_csv(filename, header=0)
	return file_dataframe


def naive_remove_duplicates(data):
	'''
		In a pair of consecutive images with label 1, set the last image label to 0
		args:
			data(pandas.DataFrame)
	'''
	prediction = data['predicted'].copy()
	for index in range(1, len(prediction)):
		if prediction[index] == 1:
			prediction[index] = (prediction[index] + prediction[index-1]) % 2

	return prediction



def calculate_metrics(targets, predicted): 
	accuracy = (predicted == targets).sum() / len(predicted)

	f1 = metrics.f1_score(targets, predicted)
	precision = metrics.precision_score(targets, predicted)
	recall = metrics.recall_score(targets, predicted)
	bal_acc = metrics.balanced_accuracy_score(targets, predicted)

	print(f"f1: {round(f1,6)}, precision: {round(precision,6)}, recall: {round(recall,6)}, accuracy:{round(accuracy,6)}, bal_acc: {round(bal_acc,6)} \n")       
	return f1, precision, recall



def probability_based_removal(data):
	'''
		select the image with the highest probability score from the 3 consecutive occuring images with class label 1
		args:
			data(pandas.DataFrame)
	'''
	
	prob_based = data['predicted'].copy()
	for index in range(len(prob_based)-1):
		if prob_based[index] == 1: 
			if index+1 < len(prob_based) and data['softmax_1'][index] < data['softmax_1'][index+1]: 
				prob_based[index] = 0
			
			if index+2 < len(prob_based) and data['softmax_1'][index] < data['softmax_1'][index+2]: 
				prob_based[index] = 0
			
			if index-1 >= 0 and data['softmax_1'][index] < data['softmax_1'][index-1]: 
				prob_based[index] = 0
			
			if (index-2) >= 0 and data['softmax_1'][index] < data['softmax_1'][index-2]: 
				prob_based[index] = 0   

			# print(data['image'][index], "-->", prob_based[index])
	return prob_based


def process_data(folder_name=None):
	print(folder_name)
	folder_name = "ood"
	dir = "/home/phn501/plot-finder/predict/results/unprocessed-files/"+folder_name
	name, f1, precision, recall = [], [], [], []
	for file in os.listdir(dir):
		if file.endswith(".csv") and file != "results.csv": 
			print(file, "\n")
			filename 	= os.path.join(dir, file)
			file_df 	= read_file(filename)
			file_df 	= file_df.sort_values('name')
			file_df 	= file_df.reset_index()

			prob_based  			= probability_based_removal(file_df)			
			file_df['prob_based'] 	= prob_based
			target 					= file_df["target"]
			
			f_one, prec, rec = calculate_metrics(target, prob_based)
			name.append(file)
			f1.append(f_one)
			precision.append(prec)
			recall.append(rec)

			filename 	= "/home/phn501/plot-finder/predict/results/post-processed-files/"+str(folder_name)+"/"+file
			file_df 	= file_df[["name", "target", "predicted", "prob_based"]]
			
			file_df.to_csv(filename, index=False)

			# os.remove("/home/phn501/plot-finder/predict/results/unprocessed/"+file)
	print(f"f1: {np.mean(f1)}, precision: {np.mean(precision)}, recall: {np.mean(recall)}")
	df = pd.DataFrame(zip(name, f1, precision, recall), columns=["name", "f1", "precision", "recall"])
	df.to_csv("/home/phn501/plot-finder/predict/results/result/ood_results.csv", index=False)

process_data("ood")
