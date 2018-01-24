# beeva-poc-pytorch-for-recsys
Proof of Concept with PyTorch-based [Spotlight](https://github.com/maciejkula/spotlight) library for Recommender Systems at BEEVA

## Spotlight

### Instructions

* Scenario 1 (CPU): laptop 16GB, 4 processors Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz, spotlight=0.1.3

```
# Instructions for infrastructure 1
docker run -i -t -p 8888:8888 -v $PWD/notebooks:/opt/notebooks continuumio/anaconda /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --allow-root"
```
* Scenario 2 (GPU):  AWS p2.x (1 gpu nvidia Tesla K80). Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-f1d51489), NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10, spotlight=0.1.3
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# Miniconda3 default installation
export PATH=/home/ubuntu/miniconda3/bin:$PATH
conda install jupyter -y
source activate $HOME/miniconda3
python -m ipykernel install --user --name py3-conda --display-name "Python3 conda"
jupyter notebook --ip='*' --allow-root
# If disk space problems: rm -rf /home/ubuntu/src/cntk
```
Then open `spotlight_experiment.ipynb` notebook in your browser

### Results

#### Movielens 1M

| Scenario | Dataset preparation | Algorithm | Parameters | MAP | F1 score | training time
| --- | --- | --- | -----------| ---- | --- | ---
| 1 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.079+-0.0005 | 0.057+-0.0005 | 60 +-1s
| 1 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.063+-0.0005 | 0.037+-0.001 | 64.5 +-1.5s
| 1 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=10, loss='bpr'| 0.068 | 0.038 |
| --- | --- | --- | -----------| ---- | --- | ---
| 2 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'|  |  | 
| 2 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'|  |  | 19s 


#### Movielens 10M

| Scenario | Dataset preparation | Algorithm | Parameters | MAP | F1 score | training time
| --- | --- | --- | -----------| ---- | --- | ---
| 1 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.076 | 0.059 | 3h33:09s
| 1 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.040 | 0.025 |2h:39:41s
| --- | --- | --- | -----------| ---- | --- | ---
| 2 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.077 +-0.001 | 0.059 +-0.001 | 5:08.5s +-0.5s
| 2 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.038+-0.002 | 0.023+-0.002 | 5:05.5s +-0.5s

### Conclusions
* Execution on scenario 2 (GPU) is around 40x faster for Movielens10M not transposed.

### Issues
* Not compatible with P3 instances and NVIDIA containers: Spotlight 0.1.3 requires CUDA 8 but Tesla V100 requires CUDA9 for optimal performance. Error. `Found GPU0 Tesla V100-SXM2-16GB which requires CUDA_VERSION >= 8000 [...]`
* Not able to run nvidia-docker2 on Ubuntu P2 instances with Deep Learning AMI Ubuntu Oct 2017. `Error: requirement error: unsatisfied condition: cuda >= 9.0\`
