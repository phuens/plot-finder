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
	return f1



def probability_based_removal(data):
	'''
		select the image with the highest probability score from the 3 consecutive occuring images with class label 1
		args:
			data(pandas.DataFrame)
	'''

	prob_based = data['predicted'].copy()
	softmax_based = data['predicted'].copy()
	for index in range(len(prob_based)-1):
		if (prob_based[index] == 1) and (prob_based[index+1] == 1):

			#based on the linear output of the two classes
			greater_prob = data['prob_1'][index] > data['prob_1'][index+1]
			prob_based[index] = int(greater_prob)
			prob_based[index+1] = int(not greater_prob)

			# based on the softmax activation of the ouput
			greater_prob = data['softmax_1'][index] > data['softmax_1'][index+1]
			softmax_based[index] = int(greater_prob)
			softmax_based[index+1] = int(not greater_prob)
				   

	return prob_based, softmax_based



def process_data():
	dir = "/home/phuntsho/Desktop/plot-finder/plot-finder/predict/result/csv"
	softmax_f1, naive_f1, prob_f1 = [], [], []
	for file in os.listdir(dir):
		if file.endswith(".csv"): 
			filename = os.path.join(dir, file)
			file_df 	= read_file(filename)
			
			# file_df 	= file_df.drop('misc', axis=1)
			naive_based = naive_remove_duplicates(file_df)
			prob_based, softmax_based  = probability_based_removal(file_df)
			file_df['naive_based'] 	= naive_based
			file_df['prob_based'] 	= prob_based
			file_df['softmax_based'] 	= softmax_based
			
			print(f"-----------{file.upper()}-----------")
			target = file_df["target"]
			print("ORIGINAL")
			naive_f1.append(calculate_metrics(target, file_df["predicted"]))

			print("Naive based")
			naive_f1.append(calculate_metrics(target, naive_based))

			print("Prob based")
			prob_f1.append(calculate_metrics(target, prob_based))

			print("softmax based")
			softmax_f1.append(calculate_metrics(target, softmax_based))

			filename = "/home/phuntsho/Desktop/plot-finder/plot-finder/predict/result/csv/post-processed/"+file
			file_df.to_csv(filename)
			print("=========================================================================\n")
	
	print("softmax f1:", round(sum(softmax_f1)/len(softmax_f1), 6))
	print("naive f1: ", round(sum(naive_f1)/len(naive_f1), 6))
	print("prob f1:", round(sum(prob_f1)/len(prob_f1), 6))
process_data()