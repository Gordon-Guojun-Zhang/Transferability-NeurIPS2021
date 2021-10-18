import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader, InfiniteSubDataLoader, FastSubDataLoader
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default='domainbed/datasets/')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=1000,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoint/")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--d_steps_per_g', type=int, default=30)
    parser.add_argument('--train_delta', type=float, default=0.5)
    parser.add_argument('--lr_d', type=float, default=1e-3, help='step size for the maximization optimizer')
    parser.add_argument('--lr', type=float, default=5e-5, help='step size for the minimization optimizer')
    args = parser.parse_args()


    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None


    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))


    if args.dataset in ['PACS', 'OfficeHome']:
        hparams['batch_size'] = 32
    elif args.dataset == 'WILDSFMoW':
        hparams['batch_size'] = 16
    if args.algorithm == 'Transfer': 
        hparams['d_steps_per_g'] = args.d_steps_per_g
        hparams['delta'] = args.train_delta
        hparams['lr_d'] = args.lr_d

    if args.dataset == 'RotatedMNIST' and args.algorithm == 'Transfer':
        hparams['lr'] = 0.01

    if args.algorithm == 'Transfer':
        output_dir = os.path.join(args.output_dir, args.dataset, args.algorithm + '_' + str(args.d_steps_per_g) + '_' + str(args.train_delta))
    else:
        output_dir = os.path.join(args.output_dir, args.dataset, args.algorithm)


    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(output_dir, 'err.txt'))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    data_dir = args.data_dir
    alg_label = args.algorithm
    dataset_label = args.dataset
    data_dir = data_dir + dataset_label
    
    in_file = os.path.join(data_dir, 'in.pickle')
    out_file = os.path.join(data_dir, 'out.pickle')

    with open(in_file, 'rb') as f_in:
        in_splits = pickle.load(f_in)
    with open(out_file, 'rb') as f_out:
        out_splits = pickle.load(f_out)

    
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    if args.dataset in ['OfficeHome', 'WILDSFMoW']:   
        ''' if the dataset is large, evaluate on the test set only'''
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for env, _ in out_splits]


        eval_weights = [None for _, weights in out_splits]
        eval_loader_names = ['env{}_out'.format(i)
            for i in range(len(out_splits))]
    else:
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for env, _ in (in_splits + out_splits)]

        eval_weights = [None for _, weights in (in_splits + out_splits)]
        eval_loader_names = ['env{}_in'.format(i)
            for i in range(len(in_splits))]
        eval_loader_names += ['env{}_out'.format(i)
            for i in range(len(out_splits))]


    print('num of domains: ', len(eval_loaders))
    print('lengths: ', [len(env) for env, _ in in_splits])

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    checkpoint_dir = args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    check_path = os.path.join(args.checkpoint_dir, 'model.pkl')
    print('check path: ', check_path)

    if os.path.exists(check_path):
        checkpoint_dict = torch.load(check_path)
        start_step = checkpoint_dict['step']
        print(f"loading from checkpoint,  start_step: {start_step}")
        algorithm_dict = checkpoint_dict['model_dict']
        algorithm.load_state_dict(algorithm_dict)
    else:
        start_step = 0
        
    if start_step == 0:
        print('Args:')
        for k, v in sorted(vars(args).items()):
            print('\t{}: {}'.format(k, v))
        print('output_dir: ', output_dir)
        print('HParams:')
        for k, v in sorted(hparams.items()):
            print('\t{}: {}'.format(k, v))

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS

    def save_checkpoint(output_dir, filename, step):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "step": step, 
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(output_dir, filename))


    last_results_keys = None

    epochs_path = os.path.join(output_dir, 'results.jsonl')
    if os.path.exists(epochs_path):
        os.remove(epochs_path)

    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if step % 100 == 0:
            print(f"step: {step}")
                
        algorithm.to(device)
        if args.algorithm == 'Transfer' and (args.dataset in ['OfficeHome, WILDSFMoW']):
            step_vals = algorithm.update_second(minibatches_device, None)
        else:
            step_vals = algorithm.update(minibatches_device, None)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                #loss = misc.loss(algorithm, loader, device)
                results[name+'_acc'] = acc

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)    
            })

            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            checkpoint_vals = collections.defaultdict(lambda: [])

            #if args.save_model_every_checkpoint:

            if step % args.checkpoint_freq == 0 and step != 0:
                print("saving to checkpoint", 'checkpoint dir: ', args.checkpoint_dir, 'output dir: ', output_dir)
                save_checkpoint(args.checkpoint_dir, f'model_temp.pkl', step)
                check_path = os.path.join(args.checkpoint_dir, 'model.pkl')
                temp_path = os.path.join(args.checkpoint_dir, 'model_temp.pkl')
                os.replace(temp_path, check_path)
                save_checkpoint(output_dir, f'model_step{step}.pkl', step)

    save_checkpoint(output_dir, 'model.pkl', step)

    with open(os.path.join(output_dir, 'done'), 'w') as f:
        f.write('done')
