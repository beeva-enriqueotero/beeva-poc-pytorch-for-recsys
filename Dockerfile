FROM nvidia/cuda:8.0-cudnn7-runtime
RUN apt update && \
	apt install wget -y && \ 
	apt install bzip2 && \
	/usr/bin/wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
	bash ~/miniconda.sh -b -p /opt/conda/
ENV PATH="/opt/conda/bin:$PATH"
