ARG WORK_DIR=none 
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt update && apt install -y less nano git 

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    black \
    flake8 \
    isort \
    jupyter \
    jupyterlab \
    pandas \
    pre-commit \
    torchmetrics==0.10.3 \
    transformers \
    transformers-interpret \
    xformers==0.0.18 

RUN git config --global --add safe.directory /app