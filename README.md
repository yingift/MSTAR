<div align="center">

# 🌟 【NeurIPS 2025】MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling

<p>
  <a href="https://arxiv.org/abs/2506.10609"><img src="https://img.shields.io/badge/arXiv-2506.10609-b31b1b.svg" alt="arXiv"></a>
  <a href="https://neurips.cc/virtual/2025/poster/"><img src="https://img.shields.io/badge/NeurIPS-2025%20Poster-4b44ce.svg" alt="NeurIPS 2025"></a>
  <a href="https://huggingface.co/stableapuppy/MSTAR"><img src="https://img.shields.io/badge/🤗%20Model-HuggingFace-ffd21e.svg" alt="Model"></a>
  <a href="https://huggingface.co/datasets/stableapuppy/MQTR"><img src="https://img.shields.io/badge/🤗%20Dataset-HuggingFace-ffd21e.svg" alt="Dataset"></a>
  <a href="https://github.com/yingift/MSTAR"><img src="https://img.shields.io/github/stars/yingift/MSTAR?style=social" alt="GitHub Stars"></a>
  <a href="https://github.com/yingift/MSTAR/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

**[Liang Yin](https://github.com/yingift), [Xudong Xie](https://github.com/), [Zhang Li](https://github.com/), [Xiang Bai](https://github.com/), [Yuliang Liu✉](https://github.com/)**

Huazhong University of Science and Technology

<p>
  <em>{liangyin, xdxie, zhangli, xbai, ylliu}@hust.edu.cn</em>
</p>

---

<p>
  <strong>MSTAR</strong> is a <em>box-free</em> approach for scene text retrieval that eliminates costly bounding box annotations while unifying diverse query types (word, phrase, combined, semantic) within a single model.
</p>

### 🏆 Performance Highlights

| | Metric | Value |
|:---:|:---|:---:|
| 📈 | **+6.4% MAP** on Total-Text over previous SOTA (box-free, no box annotation needed!) | **85.55** |
| 📈 | **+8.5% Avg. MAP** on MQTR over all previous methods | **66.78** |
| 📈 | **95.71 MAP** on PSTR, surpassing FDP by **+3.43%** | **95.71** |
| 🚫 | **Zero** bounding box annotations required for training | **Box-Free** |

</div>

## 🔥 News
- **[2026/04]** 💻 Code, weight and dataset are released.
- **[2025/09]** 🎉 MSTAR is accepted by **NeurIPS 2025** as a poster!
- **[2025/06]** 📄 Paper is available on [arXiv](https://arxiv.org/abs/2506.10609).

## ✨ Highlights

- 🚫 **Box-Free** — No bounding box annotations needed, significantly reducing annotation cost.
- 🔄 **Attention Recycling** — Progressive vision embedding shifts attention from salient to insalient regions, capturing fine-grained scene text features.
- 🎯 **Multi-Query Unified** — Seamlessly handles word, phrase, combined, and semantic queries with style-aware instructions.
- 📊 **New Benchmark** — We introduce **MQTR** (Multi-Query Text Retrieval), the first benchmark for evaluating multi-query scene text retrieval.
- 🏆 **State-of-the-Art** — Surpasses previous SOTA by **6.4% MAP** on Total-Text and **8.5% average MAP** on MQTR.

## 🛠️ Installation

### 1. Create Conda Environment

```bash
conda create -n mstar python=3.8
conda activate mstar
```

### 2. Install LAVIS

Install from PyPI:
```bash
pip install salesforce-lavis
```

Or install from source (recommended):
```bash
cd LAVIS
pip install -e .
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📁 Data Preparation

### Training Data

| Dataset | Description | Link |
|---------|-------------|------|
| SynthText_900KDict | Synthetic text images for pre-training | [Download](https://github.com/lluisgomez/single-shot-str) |
| MLT-5K | Multi-lingual scene text dataset | [Download](https://github.com/lanfeng4659/STR-TDSL) |

**Steps:**
1. Download the datasets from the links above.
2. Follow the training annotation files to extract images from SynthText_900KDict and MLT-5K.
3. Place the extracted images into the `images/` folder.

### Evaluation Data

Place the evaluation datasets in the `datasets/` folder. We support the following benchmarks:

| Dataset | Type | Link |
|---------|------|------|
| SVT | Word Retrieval | [Link](https://github.com/HCIILAB/Scene-Text-Recognition/blob/master/README.md) |
| STR | Word Retrieval | [Link](https://github.com/lanfeng4659/STR-TDSL/tree/main) |
| CTR | Word Retrieval | [Link](https://github.com/lanfeng4659/STR-TDSL/tree/main) |
| Total-Text | Word Retrieval | [Link](https://github.com/cs-chan/Total-Text-Dataset) |
| CTW | Word Retrieval | [Link](https://github.com/HCIILAB/Scene-Text-Recognition/blob/master/README.md) |
| ICDAR15 | Word Retrieval | [Link](https://github.com/HCIILAB/Scene-Text-Recognition/blob/master/README.md) |
| PSTR | Phrase Retrieval | [Link](https://github.com/Gyann-z/FDP) |
| **MQTR** (Ours) | Multi-Query Retrieval | [Download]([#](https://huggingface.co/datasets/stableapuppy/MQTR)) |

The expected directory structure:
```
MSTAR/
├── datasets/
│   ├── SVT/
│   ├── STR/
│   ├── CTR/
│   ├── TotalText/
│   ├── CTW/
│   ├── ICDAR15/
│   ├── PSTR/
│   └── MQTR/
├── images/            # Training images
├── pretrained/        # Pretrained weights
│   └── mstar_weights/
├── eval/
├── run_scripts/
│   ├── train/
│   └── eval/
└── ...
```

## 🚀 Training

### Train MSTAR

```bash
bash run_scripts/train/train_mstar.sh
```

The training script uses the LAVIS framework with `blip2_image_text_matching` as the base model. Key training configurations can be modified in the script or the corresponding config files.

## 📊 Evaluation

### 1. Prepare Pretrained Weights

Download the pretrained model weights and place them in the `mstar_weights/` folder:

| Model | Download |
|-------|----------|
| MSTAR (Word) | `mstar_weights/mstar_word1.pth` |

### 2. Evaluate on Word Retrieval Benchmarks

Evaluate on public word-level datasets (SVT, STR, CTR, Total-Text, CTW, ICDAR15):

```bash
bash run_scripts/eval/eval_word.sh
```

You can modify the dataset and parameters in the script. For example, to evaluate on a specific dataset:

```bash
# Set the checkpoint and ViT model
CHK=mstar_weights/mstar_word1.pth
VIT=ft_siglip800_word_hug_rnn

# Evaluate on a specific dataset (e.g., SVT, STR, CTR, TotalText, CTW, ICDAR15)
python eval/evaluate.py --dataset SVT \
    --rerank \
    --top_k_ratio 0.05 \
    --device 'cuda:0' \
    --model_name blip2_image_text_matching \
    --vit $VIT \
    --batch_size 1 \
    --text_prompt 'default' \
    --checkpoint $CHK
```

**Key Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name (`SVT`, `STR`, `CTR`, `TotalText`, `CTW`, `ICDAR15`, `PSTR`, `MQTR`) | `CTW` |
| `--rerank` | Enable reranking for improved performance | flag |
| `--top_k_ratio` | Top-k ratio for reranking candidates | `0.05` |
| `--device` | CUDA device | `cuda:0` |
| `--model_name` | Base model architecture | `blip2_image_text_matching` |
| `--vit` | ViT backbone variant | `ft_siglip800_word_hug_rnn` |
| `--batch_size` | Evaluation batch size | `1` |
| `--text_prompt` | Text prompt style | `default` |
| `--checkpoint` | Path to pretrained weights | - |

### 3. Evaluate on Multi-Query Retrieval (MQTR)

```bash
bash run_scripts/eval/eval_mstar.sh
```

> 💡 **Tip:** Image embeddings are cached in `image_cache/{dataset}/` to speed up repeated evaluations. The cache is automatically rebuilt for each run.

## 📈 Main Results

### 🏅 Word Retrieval on Six Public Datasets (MAP%)

| Method | Venue | SVT | STR | CTR | Total-Text | CTW | IC15 | Avg. | FPS |
|--------|-------|:---:|:---:|:---:|:----------:|:---:|:----:|:----:|:---:|
| *Box-Based Methods* | | | | | | | | | |
| Mishra et al. | ICCV'13 | 42.70 | 56.24 | - | - | - | - | - | 0.1 |
| Jaderberg et al. | IJCV'16 | 86.30 | 66.50 | - | - | - | - | - | 0.3 |
| Gomez et al. | ECCV'18 | 83.74 | 69.83 | 41.05 | - | - | - | - | 43.5 |
| Mafla et al. | PR'21 | 85.74 | 71.67 | - | - | - | - | - | 42.2 |
| TDSL | CVPR'21 | 89.38 | 77.09 | 66.45 | 74.75 | 59.34 | 77.67 | 74.16 | 12.0 |
| Wang et al. | TPAMI'24 | - | 81.02 | 72.95 | - | - | - | - | 9.3 |
| Wen et al. | WSDM'23 | 90.95 | 77.40 | - | 80.09 | - | - | - | 11.0 |
| FDP-RN50×16 | ACM MM'24 | 89.63 | 89.46 | - | 79.18 | - | - | - | 11.8 |
| *Box-Free Methods* | | | | | | | | | |
| BLIP2 (FT) | PMLR'23 | 88.73 | 85.40 | 45.75 | 77.20 | 82.33 | 55.13 | 72.42 | 37.2 |
| **MSTAR** 🌟 | **NeurIPS'25** | **91.31** | **86.25** | 60.13 | **85.55** | **90.87** | 81.21 | **82.56** | 14.2 |
| **MSTAR (+rerank)** 🌟 | **NeurIPS'25** | 91.11 | 86.14 | **65.25** | **86.96** | **92.95** | **82.69** | **84.18** | 6.9 |

> 📊 *Comparisons of MAP% on 6 public word retrieval datasets. **Bold** = best result.*

### 🏅 Comparison with Scene Text Spotting Methods (MAP%)

| Method | Venue | SVT | STR | CTR | Total-Text | CTW | IC15 | Avg. | FPS |
|--------|-------|:---:|:---:|:---:|:----------:|:---:|:----:|:----:|:---:|
| *Box-Based* | | | | | | | | | |
| ABCNet | TPAMI'21 | 82.43 | 67.25 | 41.25 | 73.23 | 74.82 | 69.28 | 68.04 | 17.5 |
| MaskTextSpotterV3 | ECCV'20 | 83.14 | 74.48 | 55.54 | 83.29 | 80.03 | 77.00 | 75.58 | 2.4 |
| Deepsolo | CVPR'23 | 87.15 | 76.58 | 67.22 | 83.19* | 87.67* | 82.80* | 80.77 | 10.0 |
| TG-Bridge | CVPR'24 | 87.23 | 81.30 | 70.08 | 87.11* | 88.39* | 83.55* | 82.94 | 6.7 |
| *Box-Free* | | | | | | | | | |
| SPTSv2 | TPAMI'23 | 78.08 | 62.11 | 48.39 | 73.61* | 83.30* | 66.27* | 68.63 | 7.6 |
| **MSTAR** 🌟 | **NeurIPS'25** | **91.31** | **86.25** | 60.13 | 85.55 | **90.87** | 81.21 | 82.56 | **14.2** |
| **MSTAR (+rerank)** 🌟 | **NeurIPS'25** | 91.11 | 86.14 | **65.25** | 86.96 | **92.95** | 82.69 | **84.18** | 6.9 |

> 📊 *\* indicates finetuning on corresponding training sets. MSTAR achieves competitive performance **without any box annotations and dataset-specific finetuning**.*

### 🏅 Phrase-Level Retrieval on PSTR (MAP%)

| BLIP2 | TDSL | SigLIP | FDP | **MSTAR** 🌟 |
|:-----:|:----:|:------:|:---:|:-----------:|
| 85.49 | 89.40 | 89.56 | 92.28 | **95.71** (+3.43↑) |

### 🏅 Multi-Query Retrieval on MQTR (MAP%)

| Method | Venue | Avg. | Word | Phrase | Combined | Semantic |
|--------|-------|:----:|:----:|:------:|:--------:|:--------:|
| *Box-Based* | | | | | | |
| ABCNet | TPAMI'21 | 24.13 | 26.14 | 15.15 | 36.47 | 18.74 |
| MaskTextSpotter | ECCV'20 | 32.43 | 46.72 | 27.53 | 29.08 | 26.37 |
| TDSL | CVPR'21 | 58.25 | 69.11 | 40.83 | 72.71 | 50.36 |
| Deepsolo | CVPR'23 | 52.04 | 67.54 | 25.68 | 72.14 | 42.79 |
| TG-Bridge | CVPR'24 | 54.09 | 69.89 | 30.21 | **75.53** | 40.73 |
| *Box-Free* | | | | | | |
| SPTSv2 | TPAMI'23 | 35.18 | 33.56 | 21.24 | 50.76 | 35.16 |
| BLIP2 | PMLR'23 | 36.13 | 17.31 | 32.76 | 25.80 | 68.63 |
| SigLIP | CVPR'23 | 36.06 | 17.81 | 32.88 | 21.81 | 72.23 |
| BLIP2 (FT) | PMLR'23 | 58.11 | 58.09 | 42.23 | 60.84 | 71.24 |
| **MSTAR** 🌟 | **NeurIPS'25** | **66.78** (+8.5↑) | **73.27** | **44.22** | 74.48 | **75.14** |

> 📊 *FT = FineTune. MSTAR outperforms all previous methods by a significant margin on the MQTR benchmark.*

## 🔍 Visualization

<p align="center">
  <img src="assets/visualization.png" width="95%">
</p>

## 📝 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{yin2025mstar,
  title={MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling},
  author={Yin, Liang and Xie, Xudong and Li, Zhang and Bai, Xiang and Liu, Yuliang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## 🙏 Acknowledgements

This project is built upon the excellent [LAVIS](https://github.com/salesforce/LAVIS) framework. We thank the authors for their great work.

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

## 📧 Contact

If you have any questions, please feel free to open an issue or contact us at:
- **Liang Yin**: liangyin@hust.edu.cn
- **Yuliang Liu** (Corresponding Author): ylliu@hust.edu.cn
