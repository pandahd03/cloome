# Contrastive learning of image- and structure-based representations in drug discovery

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.9-red.svg)](https://pytorch.org/get-started/previous-versions/)
[![DOI](https://zenodo.org/badge/506966028.svg)](https://zenodo.org/badge/latestdoi/506966028)


This repository contains the code for training and testing CLOOME, which is developed on the code basis of [CLOOB](https://github.com/ml-jku/cloob). Weights for the trained CLOOME can be downloaded here: https://huggingface.co/anasanchezf/cloome/tree/main.
Embeddings of all images in the CellPainting dataset can be found here: https://huggingface.co/anasanchezf/cloome/blob/main/image_embeddings.zip

The data used for training is the subset of treated cells from the Cell Painting dataset (http://gigadb.org/dataset/100351; see download instructions here https://github.com/gigascience/paper-bray2017). Pre-processed images from this dataset can be found at https://ml.jku.at/software/cellpainting/dataset/. 

![plot](cloome_fig.png)

## Abstract
Contrastive learning for self-supervised representation learning has brought a strong improvement to many application areas, such as computer vision and natural language processing. With the availability of large collections of unlabeled data in vision and language, contrastive learning of language and image representations has shown impressive results. The contrastive learning methods CLIP and CLOOB have demonstrated that the learned representations are highly transferable to a large set of diverse tasks when trained on multi-modal data from two different domains. In life sciences, similar large, multi-modal datasets comprising both cell-based microscopy images and chemical structures of molecules are available.

However, contrastive learning has not yet been used for this type of multi-modal data, although this would allow to design cross-modal retrieval systems for bioimaing and chemical databases. In this work, we present a such a contrastive learning method, the retrieval systems, and the transferability of the learned representations. Our method, Contrastive Learning and leave-One-Out-boost for Molecule Encoders (CLOOME), is based on both CLOOB and CLIP and comprises an encoder for microscopy data, an encoder for chemical structures and a contrastive learning objective, which produce rich embeddings of bioimages and chemical structures. 

On the benchmark dataset ”Cell Painting”, we demonstrate that the embeddings can be used to form a retrieval system for bioimaging and chemical databases. We also show that CLOOME learns transferable representations by performing linear probing for activity prediction tasks. Furthermore, the image embeddings can identify new cell phenotypes, as we show in a zero-shot classification task. 

The paper can be found here: [here](https://www.nature.com/articles/s41467-023-42328-w).


## Geting started
You can now easily encode microscopy images with CLOOME pretrained models following these steps.

```bash
# Clone repository and swtich into the directory to work with example files and config
git clone https://github.com/ml-jku/cloome
cd cloome

# Install 
pip install git+https://github.com/ml-jku/cloome
```

```python
import os
from cloome.model import CLOOME
from huggingface_hub import hf_hub_download

# get checkpoint from Hugging Face
FILENAME = "cloome-bioactivity.pt"
REPO_ID = "anasanchezf/cloome"
ckpt = hf_hub_download(REPO_ID, FILENAME)

config = "src/training/model_configs/RN50.json"
images = [os.path.join("example", "images", f"{channel}.tif") for channel in ["Mito", "ERSyto", "ERSytoBleed", "Ph_golgi", "Hoechst"]]


encoder = CLOOME(ckpt, config)
img_embeddings = encoder.encode_images(images)
```

## Funding
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie Actions, grant agreement “Advanced machine learning for Innovative Drug Discovery (AIDD)” No 956832”


This is a big test