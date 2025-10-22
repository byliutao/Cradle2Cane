<p align="center">
  <h1 align="center">From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging</h1>
  <h3 align='center'>NeurIPS 2025</h3>
  <div align="center">
      <a href='https://arxiv.org/abs/2506.20977'><img src='https://img.shields.io/badge/arXiv-2501.04440-brown.svg?logo=arxiv&logoColor=white'></a>
      <a href='https://github.com/byliutao/Cradle2Cane'><img src='https://img.shields.io/badge/Github-page-yellow.svg?logo=Github&logoColor=white'></a>
      <a href='https://huggingface.co/byliutao/Cradle2Cane'><img src='https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface'></a>
  </div>
  <p align='center'>
      If you find our work helpful, please consider giving us a ⭐!
  </p>
</p>


<!-- ## ⭐️Highlights -->

<!-- ![highlight](docs/highlight.png) -->


## 📋Introduction

<h3 align='left'>
    Abstract
</h3>

<div>
  Face aging has become a crucial task in computer vision, with applications ranging from entertainment to healthcare. However, existing methods struggle with achieving a realistic and seamless transformation across the entire lifespan, especially when handling large age gaps or extreme head poses. <span style="color:orange;">The core challenge lies in balancing <i>age accuracy</i> and <i>identity preservation</i>—what we refer to as the <i>Age-ID trade-off</i></span>. Most prior methods either prioritize age transformation at the expense of identity consistency or vice versa. In this work, we address this issue by proposing a <i>two-pass</i> face aging framework, named <b>Cradle2Cane</b>, based on few-step text-to-image (T2I) diffusion models. The first pass focuses on solving <i>age accuracy</i> by introducing an adaptive noise injection (<b>AdaNI</b>) mechanism. This mechanism is guided by including prompt descriptions of age and gender for the given person as the textual condition. <span style="color:orange;">Also, by adjusting the noise level, we can control the strength of aging while allowing more flexibility in transforming the face.</span> However, identity preservation is weakly ensured here to facilitate stronger age transformations. In the second pass, we enhance <i>identity preservation</i> while maintaining age-specific features by conditioning the model on two identity-aware embeddings (<b>SVR-ArcFace</b> and <b>Rotate-CLIP</b>). This pass allows for denoising the transformed image from the first pass, ensuring stronger identity preservation without compromising the aging accuracy. Both passes are <i>jointly trained in an end-to-end way</i>. Extensive experiments on the CelebA-HQ test dataset, evaluated through Face++ and Qwen-VL protocols, show that <b>Cradle2Cane</b> outperforms existing face aging methods in age accuracy and identity consistency. Additionally, <b>Cradle2Cane</b> demonstrates superior robustness when applied to in-the-wild human face images, where prior methods often fail. This significantly broadens its applicability to more diverse and unconstrained real-world scenarios.
</div>



## 🛠️ Usage

### 1. Infer


```sh
git clone https://github.com/byliutao/Cradle2Cane

conda create --name cradle2cane python=3.10 -y
conda activate cradle2cane
pip install -r config/requirement.txt

# Download models
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download stabilityai/sdxl-turbo --local-dir models/sdxl-turbo/
huggingface-cli download --resume-download madebyollin/sdxl-vae-fp16-fix --local-dir models/sdxl-vae-fp16-fix/
huggingface-cli download --resume-download openai/clip-vit-large-patch14 --local-dir models/clip-vit-large-patch14/
huggingface-cli download --resume-download byliutao/Cradle2Cane --local-dir models/


# infer
python infer.py
``` 



### 2. Train

Download the `ffhq 512×512` dataset from the [link](https://www.kaggle.com/datasets/chelove4draste/ffhq-512x512) and put the files to `$DATAROOT`.  
Download the `json` directory from the [link](https://github.com/DCGM/ffhq-features-dataset/tree/master/json) and put it under `$DATAROOT`.  
Download the `ffhq-dataset-v2.json` directory from the [link](https://drive.google.com/file/d/16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA/view) and put it under `$DATAROOT`.  
The directory structure should look like:

```
$DATAROOT
|-- dataset
|   |-- ffhq512  # contains images :*.png
|   |-- json  # contains images :*.json
|   |-- ffhq-dataset-v2.json
```

```

# preprocess ffhq512 dataset
python -m lib.utils.ffhq_process

# check you config in train.sh first
bash config/train.sh
```




## 🚀Results

### 1. Weakly supervised Model 


### 2. Fully supervised Model (base on RSAR)




## 📘Citation

If you find our paper or benchmark helpful for your research, please consider citing our paper and giving this repo a star ⭐. Thank you very much!

```bibtex
@inproceedings{
anonymous2025from,
title={From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging},
author={Anonymous},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=E1eVGJ5RYG}
}
```


## License

Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. Any commercial use should get formal permission first. 

