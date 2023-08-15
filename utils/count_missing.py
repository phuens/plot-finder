import pandas as pd 
import os 


filedir = "/home/phn501/plot-finder/predict/result/csv/post-processed/"
count = 0
NAME, MISSES, FALSE_PRED, INCONSISTENT = [], [], [], []
for file in os.listdir(filedir):
	miss, false_pred, inconsistent = [], [], [] 
	filename = file.split("_")[0]
	if file.endswith(".csv") and filename == "50": 
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

		print(f"Misses: {len(miss)}, false prediction: {len(false_pred)}, inconsistent: {len(inconsistent)} \n\n")
	
df = pd.DataFrame(zip(NAME, MISSES, INCONSISTENT, FALSE_PRED), columns=["name", "misses", "inconsistent", "false pred"])
df.to_csv("count.csv", index=False)


