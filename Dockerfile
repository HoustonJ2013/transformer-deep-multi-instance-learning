ARG WORK_DIR=none 
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt update && apt install -y less nano git 

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    black \
    flake8 \
    h5py \ 
    isort \
    jupyter \
    jupyterlab \
    pandas \
    pre-commit \
    scikit-learn \
    torchmetrics==0.10.3 \
    transformers \
    transformers-interpret \
    xformers==0.0.18 \
    typing_extensions==4.7.1 

RUN git config --global --add safe.directory /app