# Transferability for domain generalization

This repo is for evaluating and improving transferability in domain generalization, based on our paper [Quantifying and Improving Transferability in Domain Generalization](https://arxiv.org/abs/2106.03632) (NeurIPS 2021). The code is adapted from [the DomainBed suite](https://github.com/facebookresearch/DomainBed).

* python version: 3.6
* pytorch version: 1.7.1
* cuda version: 10.2


We aim to achieve two goals:

* measure the transferability between domains
* implement the Transfer algorithm

Currently we support four datasets:

* RotatedMNIST
* PACS
* OfficeHome
* WILDS-FMoW

To get started, first obtain a datasplit of a dataset. For example, if the dataset is RotatedMNIST, we run:
```sh
python save_datasets.py --dataset=RotatedMNIST
```

The next step is to run the training algorithm. For example, if we want to train ERM:
```sh
python -m train --algorithm=ERM --dataset=RotatedMNIST
```

The repo also supports the training of Transfer algorithm. For instance, if we want to train Transfer on RotatedMNIST with 30 steps per inner loop with projection radius 10.0:
```sh
python -m train --algorithm=Transfer --dataset=RotatedMNIST \
--output_dir="results" \
--steps=8000 \
--lr=0.01 \
--lr_d=0.01 \
--d_steps_per_g=30 \
--train_delta=10.0
```

Finally we could run evaluation after the training process. For example, if we want to evaluate ERM with delta=2.0:

```sh
python transfer_measure.py --algorithm=ERM --delta=2.0 --adv_epoch=10 --seed=0
```

Similarly, if we run:
```sh
python -m transfer_measure \
--d_steps_per_g=30 \
--train_delta=10.0 \
--algorithm=Transfer \
--dataset=RotatedMNIST \
--delta=2.0 \
--adv_epoch=10 \
--seed=0
```
We could evaluate the Transfer algorithm. 


## License

This source code is released under the MIT license, included [here](LICENSE).

## Citation
Comments are welcome! Please use the following bib if you use our code in your research:
```
@article{zhang2021quantifying,
  title={Quantifying and improving transferability in domain generalization},
  author={Zhang, Guojun and Zhao, Han and Yu, Yaoliang and Poupart, Pascal},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={10957--10970},
  year={2021}
}
```
