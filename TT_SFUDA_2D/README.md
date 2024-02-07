
## Requirement.

Download the [Dataset](https://drive.google.com/drive/folders/1Gd8Xfhvm2B-YwEm4YqqHYnk5DpRC8bkc?usp=sharing) and [Source-trained Model Weights](https://drive.google.com/drive/folders/1AoWojSlOMhHLeHLCgDWzbeOR4ME7a9Xq?usp=sharing) as mentioned previously. 

## Train the model.

In --source provide the source-trained model name (hrf_unet,rite_unet,chase_unet) and in --target provide to the target domain name (rite,hrf,chase). 
```
CUDA_VISIBLE_DEVICES=1 python tt_sfuda_2d.py --source hrf_unet --target rite 
```
