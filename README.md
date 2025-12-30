# UTContrast
Official PyTorch implementation of the paper UTContrast: Decoupled Two-Path Contrastive Learning with Uncertainty-aware for Medical Image Segmentation
## Architecture
<img width="1252" height="631" alt="image" src="https://github.com/user-attachments/assets/04906dfd-4ff1-46e6-8c20-97cb03b750f0" />

## Data preparation
### Binary Segmentation
• To verify the performance and general applicability of our UTContrast in the field of medical image segmentation, we conducted experiments on six challenging public datasets: Kvasir-SEG, Kvasir-Sessile, GlaS, ISIC-2016, and ISIC-2017, covering subdivision tasks across three modalities.
### Detailed information of the six datasets.

| Dataset      | Modality                      | Anatomic Region | Segmentation Target       | Data Volume |
|:------------:|:-----------------------------:|:---------------:|:-------------------------:|:-----------:|
| Kvasir-SEG   | endoscope                     | colon           | polyp                     | 1000        |
| ClinicDB     | endoscope                     | colon           | polyp                     | 612         |
| ColonDB      | endoscope                     | colon           | polyp                     | 380         |
| GlaS         | whole-slide image (WSI)       | colorectum      | gland                     | 165         |
| ISIC-2016    | dermoscope                    | skin            | malignant skin lesion     | 1279        |
| ISIC-2017    | dermoscope                    | skin            | malignant skin lesion     | 2750        |

For Kvasir-SEG, we followed the official recommendation, splitting the data into 880/120 for training and validation. For CVC-ClinicDB, we adopted a split of 490/61/61 for training, validation, and testing. For CVC-ColonDB, we used a split of 304/38/38 for training, validation, and testing. For GlaS, we employed the official split of 85/80 for training and validation. For ISIC-2016, we utilized the official split of 900/379 for training and validation. For ISIC-2017, we also followed the official recommendation, using a split of 2000/150/600 for training,
validation, and testing.
### Multi-class Segmentation
• Synapse Multi-organ dataset: Sign up in the [official Synapse website](https://www.synapse.org/Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the [TransUNet's](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using or download the [preprocessed](https://drive.google.com/file/d/1wvmw8DVyDKr5sOAFn5zUpfhbK4Vxjze4/view) data and save in the './data/synapse/' folder. 

• ACDC dataset: Download the preprocessed [ACDC](https://drive.google.com/file/d/1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc/view) dataset from Google Drive and move into './data/ACDC/' folder.
## Download weights
Pretrained weights for the encoder(RWKV-UNet) can be downloaded at (https://drive.google.com/drive/folders/1odF_NK5wYRkE0C3w9eoLUQEVbxefj66e?usp=sharing).

Checkpoints for UTContrast(RWKV-UNet) can be downloaded at (https://drive.google.com/file/d/1wxorQrdlgf6DLx4uUufbb8nNXO94vRcQ/view?usp=sharing).

## Acknowledgements
This code base uses certain code blocks and helper functions from [RWKV-UNet](https://github.com/juntaoJianggavin/RWKV-UNet) and [ConDSeg](https://github.com/Mengqi-Lei/ConDSeg).
@inproceedings{lei2025condseg,
  title={ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement},
  author={Lei, Mengqi and Wu, Haochen and Lv, Xinhua and Wang, Xin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={4571--4579},
  year={2025}
}
