# UTContrast
Official PyTorch implementation of the paper UTContrast: Decoupled Two-Path Contrastive Learning with Uncertainty-aware for Medical Image Segmentation
# Data preparation:
## Binary Segmentation
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
• Synapse Multi-organ dataset: Sign up in the [official Synapse website](https://www.synapse.org/Synapse:syn3193805/wiki/89480) and download the dataset. Then split the 'RawData' folder into 'TrainSet' (18 scans) and 'TestSet' (12 scans) following the [TransUNet's](https://github.com/Beckschen/TransUNet/blob/main/datasets/README.md) lists and put in the './data/synapse/Abdomen/RawData/' folder. Finally, preprocess using or download the [preprocessed](https://drive.google.com/file/d/1wvmw8DVyDKr5sOAFn5zUpfhbK4Vxjze4/view) data and save in the './data/synapse/' folder. 

• ACDC dataset: Download the preprocessed [ACDC](https://drive.google.com/file/d/1CruCQ-jjvA97BX-LIYwXaRMLmp3DN9zc/view) dataset from Google Drive and move into './data/ACDC/' folder.
# Download weights:
Pretrained weights for the encoder(RWKV-UNet) can be downloaded at (https://drive.google.com/drive/folders/1odF_NK5wYRkE0C3w9eoLUQEVbxefj66e?usp=sharing).

Checkpoints for UTContrast(RWKV-UNet) can be downloaded at (https://drive.google.com/drive/folders/19y_8Mzmw5u6Bg-iVfmh6-vRCBDdy149_?usp=sharing).

# Acknowledgements
This code base uses certain code blocks and helper functions from [RWKV-UNet](https://github.com/juntaoJianggavin/RWKV-UNet) and [ConDSeg](https://github.com/Mengqi-Lei/ConDSeg).
