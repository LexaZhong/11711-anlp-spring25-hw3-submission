# GHINT: GPT-enhanced Hierarchical Interaction Network for Multimodal Clinical Trial Outcome Prediction

## 11711 ANLP Spring 2024 - Homework 3

## HINT (Fork): Reproducing the Hierarchical Interaction Network for Clinical Trial Outcome Prediction

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![GitHub Repo stars](https://img.shields.io/github/stars/futianfan/clinical-trial-outcome-prediction)](https://github.com/futianfan/clinical-trial-outcome-prediction/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/futianfan/clinical-trial-outcome-prediction)](https://github.com/futianfan/clinical-trial-outcome-prediction/network/members)

This repository is a **fork of the original [HINT repository](https://github.com/futianfan/clinical-trial-outcome-prediction)**, which proposes a deep learning method for predicting clinical trial outcomes based on hierarchical interactions among multi-modal data.

## GHINT Framework

We proposed GPT-enhanced HINT (\textbf{GHINT}), which leverages advanced GPTs capable of effectively modeling long-text, clinical-domain-specific data as the protocol encoder in the HINT architecture.

The following figure illustrates the framwork of GHINT where mainly revised the protocol embedding parts in HINT.

<p align="center"><img src="./poster/ghint.png" alt="logo" width="810px" /></p>

## Ηοw-to-use

Run the main training and evaluation script:

For Phase II:

```bash
python experiments.py --base_name phase_II
```

⬇️ The following content is the original readme.
--------------------------------------------------

This repository hosts HINT, a deep learning based method for clinical trial outcome prediction.
The repository can be mainly divided into three parts:

- [`benchmark`](https://github.com/futianfan/clinical-trial-outcome-prediction/tree/main/benchmark) describes the process of curating benchmark dataset named **Trial Outcome Prediction (TOP)** for clinical trial outcome prediction.
- [`HINT`](https://github.com/futianfan/clinical-trial-outcome-prediction/tree/main/HINT) is the Hierarchical Interaction Network, a deep learning based method.
- [`data`](https://github.com/futianfan/clinical-trial-outcome-prediction/tree/main/data) stores processed data.

The following figure illustrates the pipeline of HINT.

<p align="center"><img src="./poster/hint.png" alt="logo" width="810px" /></p>

## Table Of Contents

- [Installation](#installation)
- [Benchmark Data](#benchmark)
- [HINT: Learn and Inference](#hint)
- [Tutorial (Jupyter Notebook)](#tutorial)
- [Contact](#contact)

---

<a name="installation"></a>

## ⚙️ Installation

We build conda environment and uses `conda` or `pip` to install the required packages. See [`conda.yml`](https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/conda.yml) for all the packages.

```bash
conda create -n predict_drug_clinical_trial python==3.7
conda activate predict_drug_clinical_trial
conda install -c rdkit rdkit
pip install tqdm scikit-learn
pip install torch
pip install seaborn
pip install icd10-cm
```

We use following command to activate conda environment.

```bash
conda activate predict_drug_clinical_trial
```

---

<a name="benchmark"></a>

## 📊 Benchmark Data

To standardize the clinical trial outcome prediction, we create a benchmark dataset for Trial Outcome Prediction named TOP, which incorporate rich data components about clinical trials, including drug, disease and protocol (eligibility criteria).
All the scripts are in the folder [`benchmark`](https://github.com/futianfan/clinical-trial-outcome-prediction/tree/main/benchmark).
Please see [`benchmark/README.md`](https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/benchmark/README.md) for details.

---

<a name="hint"></a>

## 🤖 HINT: Learn and Inference

After processing the data, we learn the Hierarchical Interaction Network (HINT) on the following four tasks. The following figure illustrates the pipeline of HINT. All the scripts are available in the folder [`HINT`](https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/HINT).
Please see [`HINT/README.md`](https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/HINT/README.md) for details.

### Prediction results

We add the prediction results in `./results` for all the three phases.

### Trained model

The trained HINT models for all the three phases are available in `./save_model`.

## 📚 Tutorial (jupyter notebook)

- `benchmark`: [`tutorial_benchmark.ipynb`](https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/benchmark/README.md) describes some key components of the data curation process.
- `HINT`: [`tutorial_HINT.ipynb`](https://github.com/futianfan/clinical-trial-outcome-prediction/blob/main/HINT/README.md) is a tutorial to learn and evaluate HINT step by step.

<a name="contact"></a>

## 📞 Contact

Please contact futianfan@gmail.com for help or submit an issue. This is a joint work with [Kexin Huang](https://www.kexinhuang.com/), [Cao(Danica) Xiao](https://sites.google.com/view/danicaxiao/), Lucas M. Glass and [Jimeng Sun](http://sunlab.org/).

## Benchmark Usage Agreement

The benchmark dataset and code (including data collection and preprocessing, model construction, learning process, evaluation), referred as the Works, are publicly available for Non-Commercial Use only at https://github.com/futianfan/clinical-trial-outcome-prediction. Non-Commercial Use is defined as for academic research or other non-profit educational use which is: (1) not-for-profit; (2) not conducted or funded (unless such funding confers no commercial rights to the funding entity) by an entity engaged in the commercial use, application or exploitation of works similar to the Works; and (3) not intended to produce works for commercial use.
