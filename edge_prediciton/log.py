import sys
import variables
import datetime


def create_log(args):
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
    
    time_stamp = str(datetime.datetime.now()).replace(" ", "_")
    log_var_order = [
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
        time_stamp,
        error_split,
        rbf,
        eta,
        sum_of_all_metric,
        competence,
        a
        
    ]

    log_filename = "../logs" + "/{}_{}_{}_addF_{}_{}_e_{}_s_{}_{}_{}_{}_{}_km_{}_r_{}_{}_{}_{}_{}_eta_{}_{}_c_{}_a_{}.txt".format(
        *log_var_order)

    #sys.stdout = sys.stderr = open(log_filename, 'w')
    sys.stdout.flush()
    
    return log_filename


def create_log_old(args):
    dataset = args.dataset
    lr = args.lr
    L2 = args.l2
    add_additional_feature = args.add_additional_feature
    embedding_type = args.embedding_type
    neg_x = args.neg_x
    sl_lambda = args.sl_lambda
    fusion_type = args.fusion_type
    #batch_counter = args.batch_counter
    train_type = args.training_type

    mode = args.mode
    curriculum_length = args.curriculum_length
    alpha = args.alpha

    prev_k_loss = args.prev_k_loss
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
    
    time_stamp = str(datetime.datetime.now())
    log_var_order = [
        dataset,
        lr,
        L2,
        add_additional_feature,
        embedding_type,
        fusion_type,
        train_type,
        mode,
        neg_x,
        curriculum_length,
        sl_lambda,
        alpha,
        prev_k_loss,
        seed,
        model_type,
        approach,
        lc,
        mo,
        km,
        random,
        eval_test,
        time_stamp,
        error_split,
        rbf,
        eta,
        sum_of_all_metric,
        competence,
        a
        
    ]

    log_filename = variables.dir_logs + "/{}_{}_{}_addF_{}_{}_f_{}_tt_{}_m_{}_neg_x_{}_T_{}_sl_L_{}_a_{}_pkl_{}_seed_{}_{}_{}_{}_{}_km_{}_r_{}_et_{}_{}_es_{}_rbf_{}_{}_soam_{}_comp_{}_a_{}_c.txt".format(
        *log_var_order)

    sys.stdout = sys.stderr = open(log_filename, 'w')
    sys.stdout.flush()
    
    return log_filename
