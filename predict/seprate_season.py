import os 
import pandas as pd 
import csv 

nue1_emerge = ("G0350686.JPG", "G0361484.JPG")
nue1_vigor  = ("G0330671.JPG", "G0341427.JPG")
nue1_canopy = ("G0404590.JPG", "G0415401.JPG")
nue1_flower = ("G0733247.JPG", "G0743987.JPG")

nue2_emerge = ("G0151556.JPG", "G0162501.JPG")
nue2_vigor  = ("G0548465.JPG", "G0559351.JPG")
nue2_canopy = ("G0623471.JPG", "G0634428.JPG")
nue2_flower = ("G0846515.JPG", "G0857392.JPG")

filename = "predict/result/5921530.csv"
info = [[],[],[],[],[],[],[],[]]

def reformat_row(index,row): 
    name         = row[0]
    predicted    = row[2]
    target       = row[3]

    probs = row[1].replace('[', '').replace(']', '')
    probs = probs.strip()
    probs = probs.split(' ')
    prob1, prob2 = float(probs[0]), float(probs[-1])

    info[index].append([name, target, predicted, prob1, prob2])    


def create_csv(): 
    names = ["emergence", "vigor", "canopy", "flower"]
    for i in range(len(info)):
        trial = i // 4 + 1
        season = names[i%4]
        file_name = f"NUE{trial}_{season}.csv"

        with open("predict/seasons/"+file_name,'w') as f:
            info[i].sort()
            f.write('name,target,predicted,0_prob,1_prob\n')
            for row in info[i]:
                for x in row:
                    f.write(str(x) + ',')
                f.write('\n')
        f.close()



def process_file():     
    with open(filename) as csv_file:
        next(csv_file)
        data = csv.reader(csv_file, delimiter = ',')
        for index, row in enumerate(data): 
            if nue1_emerge[0] <= row[0] <= nue1_emerge[1]: 
                reformat_row(0, row)

            elif nue1_vigor[0] <= row[0] <= nue1_vigor[1]: 
                reformat_row(1, row)
            
            elif nue1_canopy[0] <= row[0] <= nue1_canopy[1]: 
                reformat_row(2, row)
            
            elif nue1_flower[0] <= row[0] <= nue1_flower[1]: 
                reformat_row(3, row)
            
            elif nue2_emerge[0] <= row[0] <= nue2_emerge[1]: 
                reformat_row(4, row)

            elif nue2_vigor[0] <= row[0] <= nue2_vigor[1]: 
                reformat_row(5, row)
            
            elif nue2_canopy[0] <= row[0] <= nue2_canopy[1]: 
                reformat_row(6, row)
            
            elif nue2_flower[0] <= row[0] <= nue2_flower[1]: 
                reformat_row(7, row)
            else: 
                print(f"{row[0]} cannot be put in any category!")

        print(f"Processed {index} rows!")




process_file()
create_csv()