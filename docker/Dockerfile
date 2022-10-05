ARG CUDA_VERSION=11.1
ARG CUDNN_VERSION=8
ARG OS_VERSION=16.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${OS_VERSION}

WORKDIR workspace

# basic
RUN apt-get update && \
    apt-get install -y --no-install-recommends && \
    apt-get install -y build-essential python-dev python3-dev && \
    apt-get install -y git wget sudo curl openssh-server openssh-client bash-completion command-not-found \
    vim htop tmux zsh rsync bzip2 zip unzip patch time make cmake locales locales-all libgtk2.0-dev libgl1-mesa-glx python3-tk \
    ninja-build libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python-openssl libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# python3.6 pip3.6 并修改指向
RUN wget https://www.python.org/ftp/python/3.6.8/Python-3.6.8.tgz && \
    tar -xzvf Python-3.6.8.tgz && \
    cd Python-3.6.8 && \
    ./configure --enable-shared --prefix=/usr/local/python --with-ssl  && \
    make -j 10 && \
    make install && \
    cd .. && rm -rf Python-3.6.8 Python-3.6.8.tgz && \
    rm /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/local/python/bin/python3.6 /usr/bin/python && \
    ln -s /usr/local/python/bin/python3.6 /usr/bin/python3 && \
    ln -s /usr/local/python/bin/pip3.6 /usr/bin/pip && \
    ln -s /usr/local/python/bin/pip3.6 /usr/bin/pip3

ENV LD_LIBRARY_PATH=/usr/local/python/lib:$LD_LIBRARY_PATH

RUN mv /usr/bin/lsb_release /usr/bin/lsb_release_bak && \
    pip install -U pipenv setuptools


# install torch
RUN pip install --upgrade pip && pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# env
RUN echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=/usr/local/python/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc