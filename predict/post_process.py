import os
import csv
import pandas as pd
import sklearn.metrics as metrics 


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

	print(f"f1: {f1}, precision: {precision}, recall: {recall}, accuracy:{accuracy}, bal_acc: {bal_acc}")       
	return accuracy, f1, precision, recall, bal_acc



def probability_based_removal(data):
	'''
		select the image with the highest probability score from the 3 consecutive occuring images with class label 1
		args:
			data(pandas.DataFrame)
	'''

	prediction = data['predicted'].copy()
	for index in range(len(prediction)-1):
		if (prediction[index] == 1) and (prediction[index+1] == 1):
			# print(data['1_prob'][index], data['1_prob'][index+1])
			# make the element with greater prob 1 and the other 0
			greater_prob = data['1_prob'][index] > data['1_prob'][index+1]
			prediction[index] = int(greater_prob)
			prediction[index+1] = int(not greater_prob)
				   

	return prediction



def process_data():
	filename="/home/phuntsho/Desktop/plot-finder/plot-finder/predict/result/6099445_emergence.csv"
	file_df 	= read_file(filename)
	# file_df 	= file_df.drop('misc', axis=1)
	naive_based = naive_remove_duplicates(file_df)
	prob_based  = probability_based_removal(file_df)
	file_df['naive_based'] 	= naive_based
	file_df['prob_based'] 	= prob_based
	
	target = file_df["target"]

	print("Naive based")
	calculate_metrics(target, naive_based)

	print("Prob based")
	calculate_metrics(target, prob_based)

	file_df.to_csv('processed.csv')
	
process_data()