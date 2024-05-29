FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    jupyter \
    jupyterlab \
    pandas \
    torchmetrics==0.10.3 \
    transformers \
    transformers-interpret \
    xformers==0.0.18 

