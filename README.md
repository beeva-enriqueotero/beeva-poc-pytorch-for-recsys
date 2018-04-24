# beeva-poc-pytorch-for-recsys
Proof of Concept with PyTorch-based [Spotlight](https://github.com/maciejkula/spotlight) library for Recommender Systems at BEEVA

## Spotlight

### Instructions

* Scenario 1 (CPU, dockerized): laptop 16GB, 4 processors Intel(R) Core(TM) i5-4210U CPU @ 1.70GHz, spotlight=0.1.3

```
git clone https://github.com/beeva-enriqueotero/beeva-poc-pytorch-for-recsys
cd beeva-poc-pytorch-for-recsys/
docker run -i -t -p 8888:8888 -v $PWD/notebooks:/opt/notebooks continuumio/anaconda /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --allow-root"
```

* Scenario 1b (CPU, dockerized): AWS c5.4xlarge. ubuntu/images/hvm-ssd/ubuntu-xenial-16.04-amd64-server-20180109 (ami-1ee65166) Hard Disk 50GB, spotlight=0.1.3, pytorch 0.3.0

```
git clone https://github.com/beeva-enriqueotero/beeva-poc-pytorch-for-recsys
cd beeva-poc-pytorch-for-recsys/
sudo apt update
sudo apt install docker.io -y
sudo docker run -i -t -p 8888:8888 -v $PWD/notebooks:/opt/notebooks continuumio/anaconda /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --allow-root"
```

* Scenario 2 (GPU, dockerized): AWS p2.x (1 gpu nvidia Tesla K80). Ubuntu AMI (50 GB hard disk). NVIDIA-SMI 390.12. cuda-9.1 (host). Conda: 4.4.7-py36_0
```
# Install CUDA
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda -y

# Install docker-ce
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce -y --allow-unauthenticated

# Install nvidia-docker2
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo pkill -SIGHUP dockerd

# Build image
git clone https://github.com/beeva-enriqueotero/beeva-poc-pytorch-for-recsys
cd beeva-poc-pytorch-for-recsys/
sudo docker build . -t myspotlight

# Run container with --runtime=nvidia
sudo docker run -i -t --runtime=nvidia -p 8888:8888 -v $PWD/notebooks:/opt/notebooks myspotlight /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --allow-root"
# If disk space problems: sudo rm -rf /var/cache/apt
```
* Scenario 2b (GPU, no dockerized):  AWS p2.x (1 gpu nvidia Tesla K80). Deep Learning AMI Ubuntu Linux - 2.4_Oct2017 (ami-f1d51489), NVIDIA Driver 375.66, CUDA 8.0, libcudnn.so.5.1.10, spotlight=0.1.3
```
# Install Miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# Miniconda3 default installation
export PATH=/home/ubuntu/miniconda3/bin:$PATH

# Install jupyter
conda install jupyter -y
source activate $HOME/miniconda3
python -m ipykernel install --user --name py3-conda --display-name "Python3 conda"

# Get notebook and launch jupyter
git clone https://github.com/beeva-enriqueotero/beeva-poc-pytorch-for-recsys
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
| 1 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='adaptive_hinge'| 0.082 | 0.056 | 73.5s
| 1 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='adaptive_hinge'| 0.061 | 0.031 | 79.5s
| 1 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='hinge'| 0.063 | 0.039 | 81.7s
| 1 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='hinge'| 0.052 | 0.024 | 73.8s
| 1 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='pointwise'| 0.064 | 0.037 | 61.8s
| 1 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='pointwise'| 0.051 | 0.022 | 68.9s
| 1b | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'|  0.076 | 0.054 | 30.8s
| 1b | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.066 | 0.036 | 35.3s
| --- | --- | --- | -----------| ---- | --- | ---
| 2b | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.078 +-0.001  | 0.055 +-0.002 | 19s +-0.5s
| 2b | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.064 +-0.003 | 0.037 +-0.001 | 16.5s +-0.5s 


#### Movielens 10M

| Scenario | Dataset preparation | Algorithm | Parameters | MAP | F1 score | training time
| --- | --- | --- | -----------| ---- | --- | ---
| 1 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.076 | 0.059 | 3h33:09s
| 1 | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.040 | 0.025 |2h:39:41s
| 1b | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.077  | 0.058 | 2h44:14s
| --- | --- | --- | -----------| ---- | --- | ---
| 2 | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.077 | 0.059 | 5:20s
| 2b | split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.077 +-0.001 | 0.059 +-0.001 | 5:08.5s +-0.5s
| 2b | transpose & split 2/10 | ImplicitFactorizationModel | n_iter=3, loss='bpr'| 0.038+-0.002 | 0.023+-0.002 | 5:05.5s +-0.5s

### Conclusions
* Scores on transposed datasets are significantly lower than scores on original datasets.
* Execution on scenario 2 (GPU) is around 40x faster for Movielens10M not transposed.
* There is no difference in MAP between transposing the dataset and regrouping the items by user after the predictions

### Issues
* Not compatible with P3 instances and NVIDIA containers: Spotlight 0.1.3 requires CUDA 8 but Tesla V100 requires CUDA9 for optimal performance. Error. `Found GPU0 Tesla V100-SXM2-16GB which requires CUDA_VERSION >= 8000 [...]`
* Not able to run nvidia-docker2 on Ubuntu P2 instances with Deep Learning AMI Ubuntu Oct 2017. `Error: requirement error: unsatisfied condition: cuda >= 9.0\`
