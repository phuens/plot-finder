import random 
import csv
import random
import os
import math 


coverage = 1
root_dir = "/home/phn501/plot-finder/dataset/range_wise/test/"
dest_dir = "/home/phn501/plot-finder/dataset/shuffled/"

if not os.path.exists(dest_dir+str(coverage)):
    os.mkdir(dest_dir+str(coverage))


for file in os.listdir(root_dir): 
    data = []
    with open(root_dir+file, 'r') as infile:
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            data.append(row)
    
    column_titles = data[0]
    data_rows = data[1:]
    
    # Calculate the number of rows to shuffle (one fourth of total rows)
    num_rows = len(data_rows)
    num_rows_to_shuffle = math.ceil(num_rows*coverage)

    # Shuffle the first one fourth of rows
    rows_to_shuffle = data_rows[:num_rows_to_shuffle]
    random.shuffle(rows_to_shuffle)
    data[:num_rows_to_shuffle] = rows_to_shuffle

    # Write the shuffled data back to a new CSV file
    with open(dest_dir+str(coverage)+"/"+file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(column_titles)
        for row in data:
            csv_writer.writerow(row)

