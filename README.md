# beeva-poc-pytorch-for-recsys
Proof of Concept with PyTorch for Recommender Systems at BEEVA

### Instructions

```
docker run -i -t -p 8888:8888 -v $PWD/notebooks /opt/notebooks continuumio/anaconda /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && mkdir /opt/notebooks && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --allow-root
```
Then run spotlight_experiment.ipynb notebook

### Results

*Infrastructure: laptop 16GB, 4 processors Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz*

#### Movielens 1M

| Dataset preparation | Algorithm | Parameters | MAP | F1 score | training time
| --- | --- | -----------| ---- | --- | ---
| split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.079 | 0.057 | 61s
| transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.068 | 0.038 | 63s
| transpose & split 2/10 | ImplicitFactorizationModel | n_iter=10, loss='bpr'| 0.063 | 0.038 |


#### Movielens 10M

| Dataset preparation | Algorithm | Parameters | MAP | F1 score | training time
| --- | --- | -----------| ---- | --- | ---
| split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| x | x | >3h
| transpose & split 2/10 | ImplicitFactorizationModel | n_iter=10, loss='bpr'| x | x |


