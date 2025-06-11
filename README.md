# MSTAR: Box-free Multi-query Scene Text Retrieval with Attention Recycling

This repository is the official implementation of this paper.


## Requirements
1. Create conda env:
```setup
conda create -n mstar python=3.8
conda activate mstar
```

2. Install lavis from Pypi:
```setup
pip install salesforce-lavis
```
Optionally, install from source: 
```
cd LAVIS
pip install -e .
```

3. To install requirements:
```setup
pip install -r requirements.txt
```

## Training
1. Prepare the [SynthText_900KDict dataset](https://github.com/lluisgomez/single-shot-str).
2. Prepare the [MLT-5K dataset](https://github.com/lanfeng4659/STR-TDSL).
3. Follow the training annotation files to extract the images from SynthText_900KDict dataset and MLT-5K dataset. Extract the images into the "images" folder.
4. To train the model for scene text retrieval, run:
```eval
bash run_scripts/eval/eval_mstar.sh
```


## Evaluation

1. Place the pretrained weights in the "pretrained" folder.

2. Put the datasets in the "datasets" folder.
> The code supports the evaluation of the [SVT](https://github.com/HCIILAB/Scene-Text-Recognition/blob/master/README.md), [STR](https://github.com/lanfeng4659/STR-TDSL/tree/main), [CTR](https://github.com/lanfeng4659/STR-TDSL/tree/main), [Total-Text](https://github.com/cs-chan/Total-Text-Dataset), [CTW](https://github.com/HCIILAB/Scene-Text-Recognition/blob/master/README.md), [ICDAR15](https://github.com/HCIILAB/Scene-Text-Recognition/blob/master/README.md), [PSTR](https://github.com/Gyann-z/FDP), and MQTR datasets.
1. To evaluate the model on the public word datasets, run:
```eval
bash run_scripts/eval/eval_word.sh
```

2. To evaluate the model on the public multi-query datasets, run:
```eval
bash run_scripts/eval/eval_mstar.sh
```


## Results

Our model achieves the following performance on :

#### Evaluation of Six word retrieval public datasets

| Method                | Venue      | SVT   | STR   | CTR   | Total-Text | CTW  | IC15 | Avg.  | FPS  |
|-----------------------|------------|-------|-------|-------|------------|------|------|-------|------|
| **Box Based**         |            |       |       |       |            |      |      |       |      |
| Mishra et al. [1]     | ICCV'13    | 42.70 | 56.24 | -     | -          | -    | -    | -     | 0.1  |
| Jaderberg et al. [2]  | IJCV'16    | 86.30 | 66.50 | -     | -          | -    | -    | -     | 0.3  |
| Gomez et al. [3]      | ECCV'18    | 83.74 | 69.83 | 41.05 | -          | -    | -    | -     | 43.5 |
| Mafla et al. [4]      | PR'21      | 85.74 | 71.67 | -     | -          | -    | -    | -     | 42.2 |
| TDSL [5]              | CVPR'21    | 89.38 | 77.09 | 66.45 | 74.75      | 59.34| 77.67| 74.16 | 12.0 |
| Wang et al. [6]       | TPAMI'24   | -     | 81.02 | 72.95 | -          | -    | -    | -     | 9.3  |
| Wen et al. [7]        | WSDM'23    | 90.95 | 77.40 | -     | 80.09      | -    | -    | -     | 11.0 |
| FDP-RN50×16 [8]       | ACM MM'24  | 89.63 | 89.46 | -     | 79.18      | -    | -    | -     | 11.8 |
| **Box Free**          |            |       |       |       |            |      |      |       |      |
| BLIP2 (FT) [9]        | PMLR'23    | 88.73 | 85.40 | 45.75 | 77.20      | 82.33| 55.13| 72.42 | 37.2 |
| MSTAR                 | -          | 91.31 | 86.25 | 60.13 | 85.55      | 90.87| 81.21| 82.56 | 14.2 |
| MSTAR (+rerank)       | -          | 91.11 | 86.14 | 65.25 | 86.96      | 92.95| 82.69| 84.18 | 6.9  |

*Comparisons of MAP% on 6 public word retrieval datasets. The best results are shown in **bold**, and the second results are underlined.*

| Method                | Venue      | SVT   | STR   | CTR   | Total-Text | CTW  | IC15 | Avg.  | FPS  |
|-----------------------|------------|-------|-------|-------|------------|------|------|-------|------|
| **Box Based**         |            |       |       |       |            |      |      |       |      |
| ABCNet [10]           | TPAMI'21   | 82.43 | 67.25 | 41.25 | 73.23      | 74.82| 69.28| 68.04 | 17.5 |
| MaskTextspotterV3 [11]| ECCV'20    | 83.14 | 74.48 | 55.54 | 83.29      | 80.03| 77.00| 75.58 | 2.4  |
| Deepsolo [12]         | CVPR'23    | 87.15 | 76.58 | 67.22 | 83.19*     | 87.67*| 82.80*| 80.77 | 10.0 |
| TG-Bridge [13]        | CVPR'24    | 87.23 | 81.30 | 70.08 | 87.11*     | 88.39*| 83.55*| 82.94 | 6.7  |
| **Box Free**          |            |       |       |       |            |      |      |       |      |
| SPTSv2 [14]           | TPAMI'23   | 78.08 | 62.11 | 48.39 | 73.61*     | 83.30*| 66.27*| 68.63 | 7.6  |
| MSTAR                 | -          | 91.31 | 86.25 | 60.13 | 85.55      | 90.87| 81.21| 82.56 | 14.2 |
| MSTAR (+rerank)       | -          | 91.11 | 86.14 | 65.25 | 86.96      | 92.95| 82.69| 84.18 | 6.9  |

*Comparisons with mainstream scene text spotting methods. * indicates finetuning on corresponding training sets. Best results in **bold**, second results underlined.*

### Evaluation on the PSTR dataset
| BLIP2 [1] | TDSL [2] | SigLIP [3] | FDP [4] | MSTAR   |
|----------|----------|------------|---------|---------|
| 85.49    | 89.40    | 89.56      | 92.28   | **95.71** |

*Comparisons on Phrase-level Scene Text Retrieval dataset [4]. Best result in bold.*

### Evaluation on the MQTR dataset

| Method                | Venue      | AVG.  | Word  | Phrase | Combined | Semantic |
|-----------------------|------------|-------|-------|--------|----------|----------|
| **Box Based**         |            |       |       |        |          |          |
| ABCNet [1]            | TPAMI'21   | 24.13 | 26.14 | 15.15  | 36.47    | 18.74    |
| MaskTextSpotter [2]   | ECCV'20    | 32.43 | 46.72 | 27.53  | 29.08    | 26.37    |
| TDSL [3]              | CVPR'21    | 58.25 | 69.11 | 40.83  | 72.71    | 50.36    |
| Deepsolo [4]          | CVPR'23    | 52.04 | 67.54 | 25.68  | 72.14    | 42.79    |
| TG-Bridge [5]         | CVPR'24    | 54.09 | 69.89 | 30.21  | **75.53**| 40.73    |
| **Box Free**          |            |       |       |        |          |          |
| SPTSv2 [6]            | TPAMI'23   | 35.18 | 33.56 | 21.24  | 50.76    | 35.16    |
| BLIP2 [7]             | PMLR'23    | 36.13 | 17.31 | 32.76  | 25.80    | 68.63    |
| SigLIP [8]            | CVPR'23    | 36.06 | 17.81 | 32.88  | 21.81    | 72.23    |
| BLIP2 (FT) [7]        | PMLR'23    | 58.11 | 58.09 | 42.23  | 60.84    | 71.24    |
| MSTAR                 | -          | **66.78** | **73.27** | **44.22** | 74.48 | **75.14** |

*Evaluations of MAP% on MQTR. FT denotes FineTune. The best results are shown in bold.*

