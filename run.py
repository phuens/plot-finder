from train import training
from predict import predict, extract_feature
from configs import get_config
import random


if __name__ == '__main__':

    config      = get_config()
    identifier  = random.randint(0, 10000000)
    config["model"]["identifier"] = identifier

    if  config["type"]["work"] == 'train':
        training.run(config)
    

    elif config["type"]["work"] == 'test':
        predict.run(config)


    else:
        extract_feature.run(config)