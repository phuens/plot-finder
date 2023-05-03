import os
import csv
import pandas as pd


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
	filename='./NUE2_flower.csv'
	file_df 	= read_file(filename)
	file_df 	= file_df.drop('misc', axis=1)
	naive_based = naive_remove_duplicates(file_df)
	prob_based  = probability_based_removal(file_df)
	file_df['naive_based'] 	= naive_based
	file_df['prob_based'] 	= prob_based
	print(file_df.head(5))
	file_df.to_csv('processed.csv')
	
process_data()