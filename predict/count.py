import pandas as pd 
import os 
import sklearn.metrics as metrics 


def count(): 
	filedir = "/home/phn501/plot-finder/predict/results/post-processed-files/"
	count = 0
	NAME, MISSES, FALSE_PRED, INCONSISTENT = [], [], [], []
	for file in os.listdir(filedir):
		miss, false_pred, inconsistent = [], [], [] 
		if file.endswith(".csv"): 
			print(file)
			df = pd.read_csv(filedir+file)
			df = df.sort_values(by="name")
			df = df.reset_index()
			for index in range(len(df["name"])-1): 
				if df["target"][index] == 1: 
					if df["prob_based"][index] == 0 and df["prob_based"][index+1] == 0 and  df["prob_based"][index-1] == 0: 
						miss.append(df["name"][index])

				if df["prob_based"][index] == 1:
					if (index - 1) >= 0 and (index+1) < len(df["name"]): 
						if df["target"][index+1] == 1 or  df["target"][index-1] == 1: 
							inconsistent.append(df["name"][index])

						if df["target"][index+1] == 0 and  df["target"][index-1] == 0 and df["target"][index]==0: 
							false_pred.append(df["name"][index])

			MISSES.append(len(miss))
			FALSE_PRED.append(len(false_pred))
			INCONSISTENT.append(len(inconsistent))
			NAME.append(file)

			# print(f"Misses: {len(miss)}, false prediction: {len(false_pred)}, inconsistent: {len(inconsistent)} \n\n")
		
	df = pd.DataFrame(zip(NAME, MISSES, INCONSISTENT, FALSE_PRED), columns=["name", "misses", "inconsistent", "false pred"])
	df.to_csv("./results/result/counted.csv", index=False)



count()
