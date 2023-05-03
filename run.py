from train import training
from predict import predict, extract_feature
import sys


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        training.run()
    
    elif sys.argv[1] == 'test':
        predict.run()

    else:
        extract_feature.run()