# EDAL: Energy-driven Active Learning
This is the source code accompanying the paper **[*EDAL: Energy-driven Active Learning*]**

## Informations
- OS : rocky linux 8.8
- python : 3.8.10
- CUDA : 11.4
- NVIDIA Driver version : 470.82.01
- GPU : NVIDIA Geforce a10 (24GB)

## Requirements
```python
pip install -r requirements.txt
```
The related version for PyTorch and related library (e.g. torchvision) is based on cu113. Please adjust it according to the experimental environment.

## Executing code
```python
python main.py --data Cifar10-SVHN \
               --oodRatio 0.6 \
               --method VOS \
               --acqType energy_basedScore \
               --isEstimateVk True \
               --query_opt middle \
               --isInit True \
               --ai 10
```
**Explanation about arguments**
- --data: type of dataset; Cifar10-SVHN is *Cross* dataset, Cifar10 is *Split* dataset.
- --oodRatio: ratio of OOD data in Pool dataset; 0.2, 0.4, 0.6.
- --method: if we use *EDAL*(Energy), VOS; if we use Entropy, AcqOnly.
- --acqType: if we use *EDAL*(Energy), energy_basedScore; if we use Entropy, MAX_ENTROPY.
- --isEstimateVk: extimate Vk per ai.
- --query_opt: origin, top, middle, bottom, random; refer to graph.
- --isInit: Initialize raw model per ai.
- --ai: active learning iteration.
![query_opt](./README_image/Query%20set.png)