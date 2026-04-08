<a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=ACM%20SIGIR&color=green&logo=arXiv"></a> <a href="https://opensource.org/license/mit"><img src="https://img.shields.io/static/v1?label=License&message=MIT&color=red"></a>

# Context Convergence Improves Answering Inferential Questions

🧠 **Why do LLMs struggle with inferential questions?**
Because not all context is equally helpful—some sentences guide reasoning, others just add noise.

💡 **What if we could measure how useful a sentence is for reasoning?**
This work shows that *convergence*—how much a sentence narrows down possible answers—plays a key role in improving inferential QA.

## 🌟 Overview

Large Language Models (LLMs) are powerful—but they still struggle with **inferential questions** 🤔
(those where answers must be *reasoned*, not directly found).

💡 In this project, we introduce **convergence** as a signal that measures how well a sentence (hint) narrows down possible answers.

### 🔍 What we show:

* ✅ High-convergence sentences → **better QA performance**
* 📊 Convergence > cosine similarity for passage selection
* 🧠 Ordering sentences by convergence → **even better results**

## 🗂️ Repository Structure

```bash
├── dataset                                  # Data preparation and evaluation utilities
│   ├── compute_similarities.py              # Computes cosine similarity scores
│   ├── dataset_final.tar.gz                 # Ready-to-use final dataset for experiments
│   ├── make.sh                              # Rebuilds the dataset pipeline from scratch
│   ├── make_dataset.py                      # Creates dataset with convergence annotations
│   ├── merge.py                             # Merges intermediate outputs into final dataset
│   ├── qa.py                                # Runs QA evaluation pipeline
│
└── experiments                              # Experiment scripts used in the paper
    ├── convergence_vs_cosine.py             # Compares convergence vs cosine similarity
    ├── order.py                             # Tests effect of sentence ordering
```

## 📦 Dataset

📍 Preprocessed dataset included:

```text
dataset/dataset_final.tar.gz
```

* ✅ **Recommended:** Use this directly
* ⚠️ Optional: Rebuild from scratch if needed

### **🧩 What’s inside the dataset?**

The dataset is derived from **hint-based QA data (TriviaHG)** and is designed for **inferential question answering**. Unlike standard QA datasets, the answer must be **inferred by combining hints**, not extracted from a single sentence.

### **📊 Convergence in the dataset**

Each hint has a **convergence score**, measuring how well it narrows down candidate answers:

* 🟢 High → strongly filters incorrect answers
* 🟡 Medium → partially informative
* 🔴 Low → weak or ambiguous

## 🧪 Running Experiments

### ⚖️ Convergence vs Cosine

```bash
python experiments/convergence_vs_cosine.py
```

### 🔢 Sentence Ordering

```bash
python experiments/order.py
```

## 🔁 Reproducibility

You can reproduce the paper in two ways:

### ✅ Option A — Reproduce using the provided dataset

This is the easiest and recommended way.

#### Step 1 — Get the code

Follow the following steps:

```bash
git clone https://github.com/DataScienceUIBK/Context-Convergence-Inferential-QA
cd Context-Convergence-Inferential-QA
pip install termcolor
```

You do **not** need to recreate the dataset for the experiments.

#### Step 2 — Run the experiments

```bash
python experiments/convergence_vs_cosine.py
python experiments/order.py
```

This reproduces the main experimental setup using the prepared data.

### ⚠️ Option B — Rebuild the dataset from scratch

Use this only if you want to regenerate the dataset yourself.

#### Step 1 — Complete setup first

Before rebuilding anything, finish all the following steps:

 

Make sure **HintEval** is installed correctly:
👉 [https://hinteval.readthedocs.io/](https://hinteval.readthedocs.io/)

Then:

  

```bash
git clone https://github.com/DataScienceUIBK/Context-Convergence-Inferential-QA
cd Context-Convergence-Inferential-QA
pip install -r requirements.txt
```

#### Step 2 — Go to the dataset directory

```bash
cd dataset
```

#### Step 3 — Run the full dataset pipeline

```bash
bash make.sh
```

This rebuilds the dataset step by step using the scripts in `dataset/`.

#### Step 4 — Return to the repository root

```bash
cd ..
```

#### Step 5 — Run the experiments on the rebuilt data

```bash
python experiments/convergence_vs_cosine.py
python experiments/order.py
```

### 📌 Notes for exact reproduction

* Use the provided dataset if you want the closest match to the reported results.
* Rebuilding the dataset is mainly for transparency and regeneration.
* Make sure HintEval is installed correctly before rebuilding.
* The experiments in this repository correspond to the two main studies in the paper:

  * convergence vs cosine similarity
  * sentence ordering by convergence

## 🧠 Key Findings

* 🟢 Convergence is a **strong relevance signal**
* 📈 High-convergence passages → **better accuracy**
* ❌ Cosine similarity is **not reliable**
* 🔝 Ordering by convergence improves performance
* 🧭 LLMs prioritize **earlier information**

## 📚 Citation

```bibtex
```

## 📜 License

 

MIT License — see `LICENSE`.
