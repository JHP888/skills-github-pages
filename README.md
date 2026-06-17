# LDCNet: Language-dominant Dual-space Collaboration Network for Multimodal Sentiment Analysis

<p float="left"><img src="https://img.shields.io/badge/python-v3.9+-red"> <img src="https://img.shields.io/badge/pytorch-v2.6+-blue">

## The motivation.

![image-20260617142943298](README.assets/image-20260617142943298.png)

## Main Contributions

Our main contributions can be summarized as follows:

- **Proposed Framework:** We propose a Language-dominant Dual-space Collaboration Network (LDCNet) for multimodal sentiment analysis, which decomposes multimodal representations into modality-invariant and modality-specific spaces to reduce modality heterogeneity.
- **Adaptive Cross-Modality Interaction:** We design an Adaptive Cross-Modality Interaction (ACMI) module to enhance semantic consistency among modality-invariant representations and alleviate redundancy and conflicts in shared multimodal features.
- **Hierarchical Language Gate:** We propose a Hierarchical Language Gate (HLG) module to construct language-guided hyper-modality representations, enabling audio and visual modalities to provide complementary cues under the guidance of language.
- **Experimental Validation:** Extensive experiments on MOSI and MOSEI demonstrate that LDCNet outperforms representative state-of-the-art methods, while ablation studies verify the effectiveness of each proposed component.

## The Framework

![image-20260617142108306](README.assets/image-20260617142108306.png)
The framework of LDCNet. 


## Usage

### Prerequisites
- Python 3.9.19
- CUDA 12.2

### Installation
- Create a conda environment. Please make sure you have installed conda before.
```
conda create -n LDCNet python==3.9.19
```
- Activate the built LDCNet environment.
```
conda activate LDCNet
```
- Install Pytorch with CUDA
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```
### Datasets
Data files (containing processed MOSI, MOSEI datasets) can be downloaded from [here](https://drive.google.com/drive/folders/1BBadVSptOe4h8TWchkhWZRLJw8YG_aEi?usp=sharing). 
You can first build and then put the downloaded datasets into `./dataset` directory and revise the path in `./config/config.json`. For example, if the processed the MOSI dataset is located in `./dataset/MOSI/aligned_50.pkl`. Please make sure "dataset_root_dir": "./dataset" and "featurePath": "MOSI/aligned_50.pkl".
Please note that the meta information and the raw data are not available due to the privacy of YouTube content creators. For more details, please follow the [official website](https://github.com/ecfm/CMU-MultimodalSDK) of these datasets.

### Run the Codes
- Training

You can first set the training dataset name in `./train.py` as "mosei" or "mosi", and then run:
```
python train.py
```
By default, the trained model will be saved in `./pt` directory. You can change this in `train.py`.

- Testing

You can first set the testing dataset name in `./test.py` as "mosei" or "mosi", and then test the trained model:
```
python test.py
```

