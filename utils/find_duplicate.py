import os 
import csv

source_dir = "/home/phn501/plot-finder/dataset/shuffled/1"

for file in os.listdir(source_dir): 
    if file.endswith(".csv"):
        print(file)
        data = []
        file_path = os.path.join(source_dir, file)
        images = {}
        with open(file_path, "r") as csv_file: 
            file_read = csv.reader(csv_file)
            columns = next(file_read)
        
            for item in file_read: 
                if item[5] not in images: 
                    data.append(item)
                    images[item[5]] = 1
                
                else: 
                    images[item[5]] += 1
                    print(f"image duplicate: {item[5]}")
        
        
        dest_file = os.path.join(source_dir, file)
        with open(dest_file, 'w') as f:
            
            # using csv.writer method from CSV package
            write = csv.writer(f)
            
            write.writerow(columns)
            write.writerows(data)

   