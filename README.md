# Transferability for domain generalization

The code is adapted from [the DomainBed suite](https://github.com/facebookresearch/DomainBed)

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
