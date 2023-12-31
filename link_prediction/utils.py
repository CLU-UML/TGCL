
import variables
import pickle
import torch
import numpy as np
import random
import os

def load_datasets(args):

    dataset = args.dataset

    if args.dataset == "pgr":
        dataloader_loc = variables.dir_data + "/data/{}_train_test_val_doc2vec_v2.pkl".format(dataset)
        train_set, test_set, val_set = pickle.load(open(dataloader_loc, "rb"))
    else:
        dataloader_loc = variables.dir_data + "/data/{}_train_test_val_doc2vec.pkl".format(dataset)
        train_set, val_set, test_set = pickle.load(open(dataloader_loc, "rb"))

    print("Train size = {}, val size = {}, test size = {}".format(len(train_set), len(val_set), len(test_set)))
    return train_set, val_set, test_set


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    random.seed(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def get_model_name(args):
    dataset = args.dataset
    lr = args.lr
    L2 = args.l2
    add_additional_feature = args.add_additional_feature
    train_type = args.training_type
    curriculum_length = args.curriculum_length
    model_type = args.model_type
    seed = args.seed
    approach = args.approach
    lc = args.loss_creteria
    mo = args.metric_order
    km = args.use_k_means
    random = args.add_random
    eval_test = args.evaluate_test_per_epoch
    error_split = args.error_split
    rbf = args.rbf
    eta = args.eta
    sum_of_all_metric = args.sum_of_all_metric
    competence = args.competence
    a = args.a
    
    #time_stamp = str(datetime.datetime.now())
    model_var_order = [
        dataset,
        lr,
        L2,
        add_additional_feature,
        train_type,
        curriculum_length,
        seed,
        model_type,
        approach,
        lc,
        mo,
        km,
        random,
        eval_test,
        error_split,
        rbf,
        eta,
        sum_of_all_metric,
        competence, 
        a
        #time_stamp
        #batch_counter
    ]

    model_name = "{}_{}_{}_addF_{}_tt_{}_T_{}_seed_{}_{}_{}_{}_{}_km_{}_r_{}_et_{}_es_{}_rbf_{}_{}_soam_{}_comp_{}_a_{}_c".format(
        *model_var_order)

    return model_name


def save_the_best_model(model, epoch, optimizer, performance, args):
    model_name = get_model_name(args)
    checkpoint = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  "performance": performance}

    torch.save(checkpoint, variables.dir_model + '/{}_best.pth'.format(model_name))

    print("model saved.")


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
