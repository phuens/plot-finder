from train import training
from predict import predict
import sys


if __name__ == '__main__':
    print(sys.argv[1])
    test = False if sys.argv[1] == 'train' else True
    
    if not test:
        training.run()
    else: 
        predict.run()

