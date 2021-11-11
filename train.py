from tqdm import tqdm
import numpy as np
import argparse
import os
from pathlib import Path
import tensorflow as tf
from networks import Cov1DModel, single_loss
from datasets import ACCORDDataset, ACCORDDataloader
from utils import Params, cal_acc, generate_classifier, increment_dir, cal_classifier_acc

def get_args():
    parser = argparse.ArgumentParser(description='CWRU data training.')
    parser.add_argument('--cfg', type=str, default='./cfg/accord.yml', help='The path to the configuration file.')
    parser.add_argument('--log-dir', type=str, default='./log', help='The directory of log files.')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--k', type=int, default=10, help='K-fold cross validation.')
    parser.add_argument('--optim', type=str, default='adam', help='The optimizer to use (Adam or SGD).')
    parser.add_argument('--model', type=str, default='cnn', help='1DCNN/Transformer.')

    return parser.parse_args()

def train(**kwargs):
    m = kwargs.get('m', 'cnn')
    faults_classifiers = kwargs.get('faults_classifiers', None)
    epochs = kwargs.get('epochs', 1)
    m1 = kwargs.get('m1', 0.1)
    m2 = kwargs.get('m2', 0.1)
    batch_size = kwargs.get('batch_size', 8)
    time_steps = kwargs.get('time_steps', 30)
    channels = kwargs.get('channels', 76)
    optim_type = kwargs.get('optim_type', 'adam')
    threshold = kwargs.get('threshold', None)
    acc_path = kwargs.get('acc_path', None)

    accs = {}
    for k in faults_classifiers.keys():
        if m == 'cnn':
            model = Cov1DModel()
            model.build(input_shape=(batch_size*2, time_steps, channels))
        
        faults_classifiers[k]['model'] = model

        if optim_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        elif optim_type == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        elif optim_type == 'sgd':
            optimizer = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.8, nesterov=True)

        faults_classifiers[k]['optimizer'] = optimizer

    best_avg_acc = 0
    best_accs = None
    for epoch in range(epochs):
        for k in faults_classifiers.keys():
            train_loader = faults_classifiers[k]['train_loader']
            process = tqdm(enumerate(train_loader), total=train_loader.gen_len())
            train_losses = []
            train_accs = []
            pos_scores = []
            neg_scores = []
            for step, data in process:
                pos_data, neg_data = data['pos_data'], data['neg_data']
                batch = tf.concat([pos_data, neg_data], 0)

                loss = 0.0 #stepごとにlossを初期化
                with tf.GradientTape() as t:
                    pred = tf.zeros([batch_size*2, 0, 1])
                    pred = faults_classifiers[k]['model'](batch)
                    loss = single_loss(pred, m1, m2)
                    train_losses.append(loss)
                    acc, pos_sum, neg_sum = cal_acc(mode='train', threshold=threshold, pred=pred)
                    train_accs.append(acc)
                    pos_scores.append(pos_sum)
                    neg_scores.append(neg_sum)
                
                d = t.gradient(loss, faults_classifiers[k]['model'].trainable_weights)
                faults_classifiers[k]['optimizer'].apply_gradients(zip(d, faults_classifiers[k]['model'].trainable_weights))

                postfix = 'Fault: {0}, Step: {1:4d}, Train loss: {2:.4f}, Train acc: {3:.4f}, Positive score: {4:.4f}, Negative score: {5:.4f}'.format(k, step+1, sum(train_losses)/len(train_losses), sum(train_accs)/len(train_accs), sum(pos_scores)/len(pos_scores), sum(neg_scores)/len(neg_scores))
                process.set_postfix_str(postfix)

            # Eval
            val_losses = []
            val_accs = []
            pos_scores = []
            neg_scores = []
            test_dataloader = faults_classifiers[k]['test_loader']
            process = tqdm(enumerate(test_dataloader), total=test_dataloader.gen_len())
            for step, data in process:
                pos_data, neg_data = data['pos_data'], data['neg_data']
                batch = tf.concat([pos_data, neg_data], 0)
                pred = faults_classifiers[k]['model'](batch)
                loss = single_loss(pred, m1, m2)
                val_losses.append(loss)
                acc, pos_sum, neg_sum = cal_acc(mode='train', threshold=threshold, pred=pred)
                val_accs.append(acc)
                pos_scores.append(pos_sum)
                neg_scores.append(neg_sum)

                postfix = 'Fault: {0}, Step: {1:4d}, Val loss: {2:.4f}, Val acc: {3:.4f}, Positive score: {4:.4f}, Negative score: {5:.4f}'.format(k, step+1, sum(val_losses)/len(val_losses), sum(val_accs)/len(val_accs), sum(pos_scores)/len(pos_scores), sum(neg_scores)/len(neg_scores))
                process.set_postfix_str(postfix)

        cur_accs = cal_classifier_acc(fault_flags, faults_classifiers, threshold)
        accs = []
        for fault_flag, acc in cur_accs.items():
            results_str = 'Epoch: {0:4d}, fault: {1}: {2:.4f}\n'.format(epoch+1, fault_flag, acc)
            print(results_str)
            with open(acc_path, 'a') as f:
                f.write(results_str)
            accs.append(acc)
        avg_acc = sum(accs) / len(accs)
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_accs = cur_accs

    return best_accs


if __name__ == '__main__':
    args = get_args()
    cfg, log_dir, name, epochs, batch_size, k, optim_type, m = \
        args.cfg, args.log_dir, args.name, args.epochs, args.batch_size, args.k, args.optim, args.model
    
    params = Params(cfg)
    data_dir, fault_flags, time_steps, channels, ab_range, threshold, m1, m2, use_normal = \
        params.data_dir, params.fault_flags, params.time_steps, params.channels, \
        params.ab_range, params.threshold, params.m1, params.m2, params.use_normal
    
    log_dir = increment_dir(Path(log_dir) / 'exp', name)
    print('Log directory: {}'.format(log_dir))
    weight_dir = os.path.join(log_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    results_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    dataset = ACCORDDataset(data_dir=data_dir, 
                            fault_flags=fault_flags,
                            ab_range=ab_range,
                            time_steps=time_steps,
                            channels=channels,
                            k=k,
                            use_normal=use_normal)


    final_accs = {}
    for val_idx in range(k):
        print('Start {0}-Fold Cross-Validation: {1}'.format(k, val_idx+1))
        faults_classifiers = generate_classifier(fault_flags, dataset, val_idx, batch_size)
        acc_path = os.path.join(results_dir, 'avg_acc.txt')
        if m == 'cnn':
            accs = train(m=m, 
                        faults_classifiers=faults_classifiers, 
                        epochs=epochs, 
                        m1=m1, 
                        m2=m2, 
                        batch_size=batch_size, 
                        time_steps=time_steps, 
                        channels=channels, 
                        optim_type=optim_type, 
                        threshold=threshold, 
                        acc_path=acc_path)
        for fault_flag, acc in accs.items():
            if fault_flag in final_accs.keys():
                final_accs[fault_flag].append(acc)
            else:
                final_accs[fault_flag] = [acc]
        break
    print('\nFinal evaluating...\n')
    final_acc_path = os.path.join(results_dir, 'final_avg_acc.txt')
    for fault_flag, v in final_accs.items():
        avg_acc = sum(v) / len(v)
        results_str = '{0}: {1:.4f}\n'.format(fault_flag, avg_acc)
        print(results_str)
        with open(final_acc_path, 'a') as f:
            f.write(results_str)