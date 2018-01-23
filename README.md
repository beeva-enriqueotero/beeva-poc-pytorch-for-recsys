# beeva-poc-pytorch-for-recsys
Proof of Concept with PyTorch-based [Spotlight](https://github.com/maciejkula/spotlight) library for Recommender Systems at BEEVA

### Instructions

```
docker run -i -t -p 8888:8888 -v $PWD/notebooks:/opt/notebooks continuumio/anaconda /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --allow-root"
```
Then open `spotlight_experiment.ipynb` notebook in your browser

### Results

*Infrastructure: laptop 16GB, 4 processors Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz*

#### Movielens 1M

| Dataset preparation | Algorithm | Parameters | MAP | F1 score | training time
| --- | --- | -----------| ---- | --- | ---
| split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.079+-0.0005 | 0.057+-0.0005 | 60 +-1s
| transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.063+-0.0005 | 0.037+-0.001 | 64.5 +-1.5s
| transpose & split 2/10 | ImplicitFactorizationModel | n_iter=10, loss='bpr'| 0.068 | 0.038 |


#### Movielens 10M

| Dataset preparation | Algorithm | Parameters | MAP | F1 score | training time
| --- | --- | -----------| ---- | --- | ---
| split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.076 | 0.059 | 3h33:09
| transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.040 | 0.025 |2h:39:41


