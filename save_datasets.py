'''save the datasets so that they do change afterwards'''
import argparse
import pickle
from domainbed.lib import misc
import os
import torch
from domainbed import datasets
from domainbed import hparams_registry

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str, default='domainbed/datasets/')
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--output_dir', type=str, default="domainbed/datasets/")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    data_dir_saved = os.path.join('results', args.dataset, args.algorithm)
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        if env_i == 5:
            break
        print('env_i', env_i)
        seed = 0
        hold_out_frac = 0.2
        out, in_ = misc.split_dataset(env, int(len(env)*hold_out_frac), seed=0)
        in_splits.append((in_, None))
        out_splits.append((out, None))
    output_dir = args.output_dir + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_in = os.path.join(output_dir, 'in.pickle')
    output_out = os.path.join(output_dir, 'out.pickle')
    with open(output_in, 'wb') as output:
        pickle.dump(in_splits, output)
    with open(output_out, 'wb') as output:
        pickle.dump(out_splits, output)
