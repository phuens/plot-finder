import pandas as pd 
import os 
import sklearn.metrics as metrics 


SOURCE_DIR = "/home/phn501/plot-finder/predict/results/unprocessed-files/original"

def calc_metrics(targets, predicted):
    accuracy = round(((predicted == targets).sum() / len(predicted)),5)
    f1          = round(metrics.f1_score(targets, predicted), 5)
    precision   = round(metrics.precision_score(targets, predicted), 5)
    recall      = round(metrics.recall_score(targets, predicted), 5)
    bal_acc     = round(metrics.balanced_accuracy_score(targets, predicted), 5)
    
    return f1, precision, recall


def count_data(data): 
	filedir = SOURCE_DIR
	count = 0
	MISSES, FALSE_PRED, INCONSISTENT = [], [], []
	df = data
	for index in range(len(df["name"])-1): 
		if df["target"][index] == 1: 
			if df["prob_based"][index] == 0 and df["prob_based"][index+1] == 0 and  df["prob_based"][index-1] == 0: 
				MISSES.append(df["name"][index])

		if df["prob_based"][index] == 1:
			if (index - 1) >= 0 and (index+1) < len(df["name"]): 
				if df["target"][index+1] == 1 or  df["target"][index-1] == 1: 
					INCONSISTENT.append(df["name"][index])

				if df["target"][index+1] == 0 and  df["target"][index-1] == 0 and df["target"][index]==0: 
					FALSE_PRED.append(df["name"][index])

	return len(MISSES), len(INCONSISTENT), len(FALSE_PRED)


def relax_f1():
	filedir = SOURCE_DIR
	count = 0
	metric_score = pd.DataFrame(columns=["name", "f1", "precision", "recall", "misses", "inconsistent", "false prediction"])
	for file in os.listdir(filedir):
		miss, false_pred, inconsistent = [], [], [] 
		if file.endswith(".csv"): 
			print(file)
			df = pd.read_csv(os.path.join(filedir, file))
			df = df.sort_values(by="name")
			df = df.reset_index()
			df["new_predicted"] = 0 

			for index in range(len(df["name"])-1): 
				if df["prob_based"][index] == 1:
					if df["target"][index] == 1: 
						df.loc[index, "new_predicted"] = 1

					elif (index - 1) >= 0 and (index+1) < len(df["name"]): 
						if df["target"][index+1] == 1:
							df.loc[index + 1, "new_predicted"] = 1

						elif df["target"][index-1] == 1: 
							df.loc[index - 1, "new_predicted"] = 1

						else: 
							df.loc[index,"new_predicted"] = 1

			df["prob_based"] = df["new_predicted"]
			# df.drop(columns=["new_predicted"], axis=1, inplace=True)
			MISSES, INCONSISTENT, FALSE_PRED = count_data(df)
			f1, precision, recall = calc_metrics(df["target"], df["prob_based"])
			
			metric_score = metric_score.append({
		            "name"     	: file,
		            "f1"        : f1, 
		            "precision" : precision, 
		            "recall"    : recall, 
		            "misses"	: MISSES, 
		            "inconsistent":INCONSISTENT, 
		            "false prediction" : FALSE_PRED
		        }, ignore_index = True)
			
			metric_score.to_csv("/home/phn501/plot-finder/predict/results/relaxed/relaxed_score.csv", index=False)		

relax_f1()