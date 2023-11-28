import os
import warnings
warnings.filterwarnings('ignore')

import log
from train import train_model_gtnn, train_model_w_competence, init_model
from  evaluate import  eval_best_model
from torch_geometric.data import DataLoader
import datetime
import variables
import argparse
import utils

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

#@profile
def main():
    import time
    start = datetime.datetime.now()
    
    parser = argparse.ArgumentParser('Interface for GTNN framework')

    '''
    for rbf, set following arguments:
        training_type = regular
        rbf = gau/lap/lin/cos/sec/qua
        eta = choose between 0-1
        competence = equal_dist
        metric_order = ADQaQd
        approach = none

    for CCL, set the following arguments:
        training_type = regular
        rbf = none
        eta = 0
        competence = plato
        sum_of_all_metric = True # default value is False
        metric_order = A
        approach = loss_based

    for No-CL, set the following arguments:
        training_type = regular
        rbf = none
        eta = 0
        competence = none
        approach = none

    '''

    parser.register('type', bool, utils.str2bool)  # add type keyword to registries

    parser.add_argument('--dataset', type=str, default='pgr', help='dataset name - pgr, omim, cora, arxiv')

    # model hyperparameters
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 regularization weight')
    
    parser.add_argument('--curriculum_length', type=int, default=100, help="length of curriculum") #epochs
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers")
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--device', type=str, default="cuda:0", help="gpu-device")

    # model configuration

    parser.add_argument('--training_type', type=str, default="regular", help="type of training as regular, curriculum (sl), curriculum with trend (sl_trend) ")
    parser.add_argument('--seed', type=int, default=9, help="seed")
    parser.add_argument('--model_type', type=str, default="sage", help="model type")


    # dataset arguments
    parser.add_argument('--add_additional_feature', type=bool, default=True, help="to add or not to add additional feature")
    parser.add_argument('--approach', type=str, default="none", help="none, softmax, mixture_individual, mixture_total, loss_based , complexity_score_based")

    # metric order and loss creteria
    parser.add_argument('--metric_order', type=str, default= "A", help= " choose for ascending order:A/D/Qa/Qd/ADQaQd")
    parser.add_argument('--loss_creteria', type=str, default= 'max', help= "loss_creteria: min/max")
    parser.add_argument('--use_k_means', type=bool, default= False, help= "T/F")
    parser.add_argument('--add_random', type=bool, default= False, help= "T/F")
    parser.add_argument('--evaluate_test_per_epoch', type=bool, default= False, help= "T/F to calculate the results for test (if True) after every epoch")
    parser.add_argument('--error_split', type=str, default= 'train', help= "train/val")
    parser.add_argument('--rbf', type=str, default= "gau", help= "none/gau/lap/lin/cos/sec/qua")
    parser.add_argument('--eta', type=float, default= "0.5", help= "choose between 0-1")
    
    parser.add_argument('--sum_of_all_metric', type=bool, default= False, help= "T/F")
    
    #for competence update
    parser.add_argument('--competence', type=str, default= "plato", help= "plato/leaf/equal_dist")
    parser.add_argument('-a', type=float, default= 0.5, help= "choose between 0-1")

    args = parser.parse_args()
    utils.fix_seed(args.seed)

    train_set, val_set, test_set = utils.load_datasets(args)
    
    model_loc = "{}/{}_best.pth".format(variables.dir_model, args.dataset)
    print(model_loc)
    
    if False and os.path.exists(model_loc): # to load best model
        log_filename = log.create_log(args)
        print(args)
        print('Loading already trained model')
        model = init_model(args)
    else:
        log_filename = log.create_log(args)
        print(args)
        if args.approach == "none":
            if args.evaluate_test_per_epoch:
                model = train_model_gtnn(train_set, test_set,  args)
            else:
                model = train_model_gtnn(train_set, val_set,  args)
        else:
            if args.evaluate_test_per_epoch:
                model = train_model_w_competence(train_set, test_set,  args)
            else:
                model = train_model_w_competence(train_set, val_set,  args)
            
    test_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    train_data_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)
    val_data_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    data_loaders = [test_data_loader, train_data_loader, val_data_loader]
    file_name_suffix = ["_test_prediction", "_train_prediction", "_val_prediction"]
    data_split_name = ["test", "train", "val"]

    for f, D, d in zip(file_name_suffix, data_loaders, data_split_name):
        p, r, f1, predictions, predictions_proba = eval_best_model(args, model, D, d)
        prediction_file = log_filename.replace(".txt", f"{f}.txt")
        prediction_writer = open(prediction_file, "w")
        for pred,proba in zip(predictions, predictions_proba):
            prediction_writer.write("{}\t{}".format(pred.item(), proba.item()))
            prediction_writer.write("\n")
            prediction_writer.flush()

    end = datetime.datetime.now()
    diff = end - start
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    print(f'hours = {hours}, minutes = {minutes}, seconds = {seconds}')



if __name__ == "__main__":
    main()
