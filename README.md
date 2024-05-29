# transformer-deep-multi-instance-learning

## Motivation
Many real world use cases fall into the category of multi-instance learnings, such as medical image analysis, insurance claim photo analysis. In those use cases, the user may have multiple images taken/uploaded as input, and one final single decision as output. For examples, a customer who filed a accident claims usually upload many images, including the shot of the car, insurance policy, ids and other documents, to the insurance server as evidence, and the insurance decision are made based on one or more of those images. The instances can be in different modalities, such as videos, audios, images, and even strctured data. 

The transfomer-based foudational models make it easy to have comprehensive representation of the instances, such [DINO v2](https://arxiv.org/abs/2304.07193) for image embedding, [Bert/Roberta](https://arxiv.org/abs/1907.11692) for text embedding. Inspired by the work [Attention-based deep multiple instance learning](https://arxiv.org/pdf/1802.04712), we explore transformer architecture as a way to map the multiple instances into one embedding (CLS token embedding) and provide an easy to replace code for other experiments. 


## Environment 
A docker file is provided to manage the enviroment for this repo. You need to have GPU on your PC, and cuda 11.7+ and the GPU driver installed. 
```
make env
```