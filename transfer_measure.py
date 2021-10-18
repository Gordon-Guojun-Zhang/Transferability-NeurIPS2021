import json
import argparse
import numpy as np
from domainbed.lib import misc
from domainbed import algorithms
import torch
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed import datasets
import torch.nn.functional as F
import copy
import torch.nn as nn
import os
from domainbed.misc_t import argmax, argmin, round_l, loss_gap, loss_gap_batch, distance, proj, loss_acc
import pickle
import sys
from domainbed import hparams_registry



class Transfer_hparam():
    ''' hyperparameters for sup_{|h' - h| leq delta} (max_i loss_i(h) - min_j loss_j(h))'''
    def __init__(self, batch_size=64, delta=2.0, num_epochs=30, optim = 'SGD'):
       dict_ = {}
       dict_['delta'] = delta
       dict_['num_epochs'] = num_epochs
       dict_['optimizer'] = optim
       dict_['batch_size'] = batch_size
       self.dict_ = dict_
    
    def __repr__(self):
        return self.dict_


def save_checkpoint(checkpoint_dir, filename, algorithm, epoch):
    save_dict = {'next_epoch': epoch, 
                 'model_dict': algorithm.cpu().state_dict()}
    torch.save(save_dict, os.path.join(checkpoint_dir, filename))


    
def local_classifier(args, dir_to_save, num_domains, in_splits, out_splits, model, device, transfer_hparam, hparams, start_epoch, optimizer='Adam', seed=0):
    ''' compute sup_{|h' - h| leq delta} (max_i loss_i(h) - min_j loss_j(h))'''
    ''' this is the test phase so we using both training and testing data of each env'''
    delta = transfer_hparam.dict_['delta']
    num_epochs = transfer_hparam.dict_['num_epochs']
    batch_size = transfer_hparam.dict_['batch_size']
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env, _ in in_splits])
    steps_per_epoch = int(steps_per_epoch + 1)
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=hparams['lr'], weight_decay = hparams['weight_decay'])
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.01, momentum=0.9)
    '''projection, compute max_index and min_index, load data, compute grad '''
    ''' the following loader is for evaluation'''
    env_in_loaders = [InfiniteDataLoader(dataset=env, weights=env_weights,
        batch_size=batch_size, num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)]
    env_out_loaders = [FastDataLoader(
        dataset=env, batch_size=batch_size, num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)]
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    print('save to directory: ', dir_to_save)
    path = os.path.join(dir_to_save, 'delta_' + str(delta) + '_seed_' + str(seed) + '.jsonl')
    if start_epoch == 0 and os.path.exists(path):
        os.remove(path)

    ''' computing of sup starts here'''
    model.to(device)
    learned_featurizer = copy.deepcopy(model.featurizer)
    learned_classifier = copy.deepcopy(model.classifier)
    if delta == 0.0:
        num_epochs = 1
    if start_epoch >= num_epochs:
        return

    for epoch in range(start_epoch, num_epochs):
        model.to(device)
        test_minibatches_iterator = zip(*env_in_loaders)
        # train on the training data from each domain
        for step in range(steps_per_epoch):
            minibatches_device = [(x.to(device), y.to(device)) for x,y in next(test_minibatches_iterator)]
            loss_gap_ = loss_gap_batch(num_domains, minibatches_device, model, device)  # compute the loss_gap_ 
            optimizer.zero_grad()
            loss_gap_.backward()
            optimizer.step()
            model.classifier = proj(delta, model.classifier, learned_classifier)
        if not args.out_only:
            loss_all_in, acc_all_in = loss_acc(num_domains, env_in_loaders, model, device)
            max_loss_in, min_loss_in = max(loss_all_in), min(loss_all_in)
            max_acc_in, min_acc_in = max(acc_all_in), min(acc_all_in)
        loss_all_out, acc_all_out = loss_acc(num_domains, env_out_loaders, model, device)
        max_loss_out, min_loss_out = max(loss_all_out), min(loss_all_out)
        max_acc_out, min_acc_out = max(acc_all_out), min(acc_all_out)
        max_index, min_index = argmax(acc_all_out), argmin(acc_all_out)

        if not args.out_only:
            print(f"seed: {seed} epoch: {epoch} step: {step} distance: {distance(model.classifier, learned_classifier).item():4f}, max_index: {max_index}, min_index: {min_index}, loss gap: {-loss_gap_.item():4f} losses_in: {[int(loss*10000)/10000 for loss in loss_all_in]} accs_in: {[int(acc*10000)/10000 for acc in acc_all_in]}, losses_out: {[int(loss*10000)/10000 for loss in loss_all_out]} accs_out: {[int(acc*10000)/10000 for acc in acc_all_out]}, max loss_in: {max_loss_in:4f}, min loss_in: {min_loss_in:4f}, max_acc_in: {max_acc_in:4f}, min_acc_in: {min_acc_in:4f}, max loss_out: {max_loss_out:4f}, min loss_out: {min_loss_out:4f}, max_acc_out: {max_acc_out:4f}, min_acc_out: {min_acc_out:4f}")
        else:
            print(f"seed: {seed} epoch: {epoch} step: {step} distance: {distance(model.classifier, learned_classifier).item():4f}, max_index: {max_index}, min_index: {min_index}, loss gap: {-loss_gap_.item():4f}, losses_out: {[int(loss*10000)/10000 for loss in loss_all_out]} accs_out: {[int(acc*10000)/10000 for acc in acc_all_out]}, max loss_out: {max_loss_out:4f}, min loss_out: {min_loss_out:4f}, max_acc_out: {max_acc_out:4f}, min_acc_out: {min_acc_out:4f}")
        distance_ = distance(model.classifier, learned_classifier).item()
        print(f'saving to checkpoint')
        '''safe save'''
        dict_ = {}
        if not args.out_only:
            dict_['loss_in'] = loss_all_in
            dict_['acc_in'] = acc_all_in
        dict_['loss_out'] = loss_all_out
        dict_['acc_out'] = acc_all_out
        dict_['max_index'] = max_index
        dict_['min_index'] = min_index
        dict_['epoch'] = epoch
        dict_['distance'] = distance_
        with open(path, 'a') as f:
            f.write(json.dumps(dict_, sort_keys=True) + '\n')
        save_checkpoint(args.checkpoint_dir, 'model_temp.pkl', model, epoch + 1)
        check_path = os.path.join(args.checkpoint_dir, 'model.pkl')
        temp_path = os.path.join(args.checkpoint_dir, 'model_temp.pkl')
        os.replace(temp_path, check_path)
        
         
if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print('device: ', device)

    parser = argparse.ArgumentParser(description="Evaluate Transferability")
    parser.add_argument('--data_dir', type=str, default='domainbed/datasets/')
    parser.add_argument('--algorithm', type=str, default='ERM')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--output_dir', type=str, default="results_transfer")
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--adv_epoch', type=int, default=10)
    parser.add_argument('--d_steps_per_g', type=int, default=10)
    parser.add_argument('--train_delta', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_only', type=bool, default=True)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_transfer/')
    args = parser.parse_args()

    delta = args.delta
    seed = args.seed
    adv_epoch = args.adv_epoch
    print('Args: ')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')

    if args.algorithm == 'Transfer':
        data_dir_pure = os.path.join(args.dataset, args.algorithm + '_' + str(args.d_steps_per_g) + '_' + str(args.train_delta))
        #print('data_dir_pure: ', data_dir_pure)
    else:
        data_dir_pure = os.path.join(args.dataset, args.algorithm)

    data_dir_saved = os.path.join('results', data_dir_pure)
    data_dir_to_save = os.path.join('results_transfer', data_dir_pure)
    os.makedirs(data_dir_to_save, exist_ok=True)

    sys.stdout = misc.Tee(os.path.join(data_dir_to_save, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(data_dir_to_save, 'err.txt'))

    model = torch.load(data_dir_saved + '/model.pkl')
    algorithm_dict = model['model_dict']
    if args.algorithm == 'GroupDRO':
        algorithm_dict['q'] = torch.tensor([])
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    input_shape = model['model_input_shape']
    num_classes = model['model_num_classes']
    #num_domains = model['model_num_domains'] + len(args.test_envs) 
    #hparams = model['model_hparams']
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    '''dataset'''
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    data_dir = args.data_dir
    alg_label = args.algorithm
    dataset_label = args.dataset
    data_dir = data_dir + dataset_label

    in_file = os.path.join(data_dir, 'in.pickle')
    out_file = os.path.join(data_dir, 'out.pickle')


    print("data loading")
    with open(in_file, 'rb') as f_in:
        in_splits = pickle.load(f_in)
    with open(out_file, 'rb') as f_out:
        out_splits = pickle.load(f_out)
    print("data loaded")

    num_domains = len(in_splits)
    print('num_domains: ', num_domains)

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env, _ in in_splits])
    print("main batch size: ", hparams['batch_size'])
    batch_size = hparams['batch_size']
    print('step per epochs: ', steps_per_epoch)
    print('lengths: ', [len(env) for env, _ in (in_splits + out_splits)])
    print('lengths/batch: ', [len(env)/batch_size for env, _ in in_splits])

    # if you want to include out_split, just do in_splits + out_splits
    
    transfer_hparam = Transfer_hparam(batch_size=batch_size, delta=delta, num_epochs=adv_epoch)
    
    print("delta: ", transfer_hparam.dict_['delta'])
    dir_to_save = os.path.join('results_transfer', data_dir_pure)
    print("seed: ", seed)
    algorithm = algorithm_class(input_shape, num_classes, num_domains, hparams)
    '''checkpoint'''
    check_path = os.path.join(args.checkpoint_dir, 'model.pkl')
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if os.path.exists(check_path):
        checkpoint_dict = torch.load(check_path)
        start_epoch = checkpoint_dict['next_epoch']
        adv_epoch = adv_epoch - start_epoch
        print(f'loading model...starting from epoch {start_epoch} and run for {adv_epoch} epochs')
        algorithm_dict = checkpoint_dict['model_dict']
        if args.algorithm == 'GroupDRO':
            algorithm_dict['q'] = torch.tensor([])
        algorithm.load_state_dict(algorithm_dict)
    else:
        start_epoch = 0
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    print("algorithm loaded")
    local_classifier(args, dir_to_save, num_domains, in_splits, out_splits, algorithm, device, transfer_hparam, hparams, start_epoch, optimizer='Adam', seed=seed)
