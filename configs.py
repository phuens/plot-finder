import argparse
import torch 

from utils import load_config


def get_config():
    """ Parse configurations with argparse """


    config = load_config("config.yml")

    parser = argparse.ArgumentParser()

    # Directiory
    parser.add_argument('--train_csv', type=str, default=config["dataset"]["train_csv"], help='train csv')

    parser.add_argument('--test_csv', type=str, default=config["dataset"]["test_csv"], help='test csv')

    parser.add_argument('--validation_csv', type=str, default=config["dataset"]["validation_csv"], help='validation csv')

    parser.add_argument('--root_img_dir', type=str, default=config["dataset"]["root_img_dir"], help='root image directory')

    # Augmentation
    parser.add_argument('--augment', type=str, default=config["augmentation"]["augment"], help="True or False: whether to use augmentation")

    parser.add_argument('--width', type=int, default=config["augmentation"]["width"], help='widht of the image')

    parser.add_argument('--height', type=int, default=config["augmentation"]["height"], help="height of the image")

    parser.add_argument('--rotation', type=float, default=config["augmentation"]["rotation"], help="range of rotation")

    parser.add_argument('--horizontal_flip', type=float, default=config["augmentation"]["horizontal_flip"], help='probability of horizontal flip')

    parser.add_argument('--vertical_flip', type=float, default=config["augmentation"]["vertical_flip"], help='probability of vertical flip')

    parser.add_argument('--zoom_range', type=float, default=config["augmentation"]["zoom_range"], help='zoom range')
    
    parser.add_argument('--shear_range', type=float, default=config["augmentation"]["shear_range"], help='shear range')
    
    parser.add_argument('--gaussian', type=float, default=config["augmentation"]["gaussian"], help='gaussian blur')


    # Model
    parser.add_argument('--name', type=str, default=config["model"]["name"], help='[inception_v3, resnet50, resent18, convnext]')
    
    parser.add_argument('--optimizer', type=str, default=config["model"]["optimizer"], help='[Adam, SGD]')

    parser.add_argument('--loss', type=str, default=config["model"]["loss"], help='loss function to use')
    
    parser.add_argument('--scheduler', type=str, default=config["model"]["scheduler"], help='[cycliclr_exp_range, cycliclr_triangle, cosineannealing, cosine_onecyclelr]')


    parser.add_argument('--lr', type=float, default=config["model"]["lr"], help='learning rate')

    parser.add_argument('--batch', type=int, default=config["model"]["batch"], help='batch size')

    parser.add_argument('--classes', type=int, default=config["model"]["classes"], help='number of classes')

    parser.add_argument('--seed', type=int, default=config["model"]["seed"], help='randomizer seed')
    
    parser.add_argument('--epochs', type=int, default=config["model"]["epochs"], help='number of epochs')

    parser.add_argument('--save_model', type=str, default=config["model"]["save_model"], help='folder to save the current model')
    
    parser.add_argument('--momentum', type=float, default=config["model"]["momentum"], help='momentum for optmizer')

    parser.add_argument('--identifier', type=int, default=config["model"]["identifier"], help='unique id for current model')

    parser.add_argument('--class_weight', type=float, default=config["model"]["class_weight"], help='weightage for sampling')


    # Test
    parser.add_argument('--name', type=str, default=config["test"]["name"], help='name of the saved model to load')

    parser.add_argument('--model_path', type=str, default=config["test"]["model_path"], help='path for the saved model to use')


    # Wandb
    parser.add_argument('--use', type=str, default=config["wandb"]["use"], help='use wandb or not')

    parser.add_argument('--wandb_group', type=str, default=config["wandb"]["wandb_group"], help='experiment group name')
    
    parser.add_argument('--note', type=str, default=config["wandb"]["note"], help='note about the experiment')

    # Work
    parser.add_argument('--work', type=str, default=config["type"]["work"], help='whether to train, test or extract features!')



    return set_param(parser, config)


def set_param(parser, config): 
    # Parse the arguments
    args = parser.parse_args()
    
    # Modify the configuration values
    config['dataset']['train_csv']      = args.train_csv
    config['dataset']['test_csv']       = args.test_csv
    config['dataset']['validation_csv'] = args.validation_csv
    config['dataset']['root_img_dir']   = args.root_img_dir


    config['augmentation']['augment']           = False if str(args.augment) in ["False", "false", "F", "f", 0] else True
    config['augmentation']['width']             = args.width
    config['augmentation']['height']            = args.height
    config['augmentation']['rotation']          = args.rotation
    config['augmentation']['horizontal_flip']   = args.horizontal_flip
    config['augmentation']['vertical_flip']     = args.vertical_flip
    config['augmentation']['zoom_range']        = args.zoom_range
    config['augmentation']['shear_range']       = args.shear_range
    config['augmentation']['gaussian']          = args.gaussian

    config['model']['name']         = args.name
    config['model']['optimizer']    = args.optimizer
    config['model']['loss']         = args.loss
    config['model']['scheduler']    = args.scheduler
    config['model']['lr']           = args.lr
    config['model']['batch']        = args.batch
    config['model']['classes']      = args.classes
    config['model']['seed']         = args.seed
    config['model']['epochs']       = args.epochs
    config['model']['save_model']   = args.save_model
    config['model']['momentum']     = args.momentum
    config['model']['identifier']   = args.identifier
    config['model']['class_weight'] = args.class_weight

    config['test']['name']       = args.name
    config['test']['model_path'] = args.model_path


    config['wandb']['use']          = False if str(args.wandb_use) in ["False", "false", "F", "f", 0] else True
    config['wandb']['wandb_group']  = args.run_group
    config['wandb']['note']         = args.note

    return config
