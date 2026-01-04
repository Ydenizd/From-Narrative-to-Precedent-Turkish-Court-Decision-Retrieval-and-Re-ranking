# From-Narrative-to-Precedent-Turkish-Court-Decision-Retrieval-and-Re-ranking

# LegalBERTTurk – Turkish Legal NLP Experiments

A Turkish legal NLP project that builds domain-adapted language models and evaluates a two-stage retrieval + re-ranking pipeline for court decision retrieval from user-provided narratives. [file:1][file:4]

## Project overview
This repository contains Jupyter notebooks for Turkish legal NLP experiments, including:
- **Domain-adaptive pretraining (MLM)** on a Turkish legal corpus to adapt a Turkish BERT model to legal language. [file:4]
- Fine-tuning / evaluation experiments aligned with defined **research questions (RQs)**. [file:2]
- A two-stage system: **dense retrieval** (high-recall candidate generation) followed by **cross-encoder re-ranking** (precision-focused reordering). [file:1]
- Paragraph-to-decision (case-level) **score aggregation (pooling)** and evidence span highlighting to improve interpretability of retrieved decisions. [file:1]

## What this system does
- **Input:** a case narrative written in everyday or legal language. [file:1]
- **Stage 1 (Dense Retrieval):** retrieves a `topP` candidate pool of paragraphs/decisions with high recall to ensure relevant items enter the candidate set even under imperfect terminology. [file:1]
- **Stage 2 (Re-ranking):** re-orders the candidate pool using a cross-encoder re-ranker and aggregates paragraph scores to decision-level scores. [file:1]
- **Output:** the top-ranked court decisions, optionally with highlighted evidence spans that match the query narrative. [file:1]

## Repository structure
Key notebooks:
- `LegalBertTurk-Final.ipynb`: End-to-end / final experiment notebook. [file:1]
- `MLM_Main.ipynb`: Masked Language Modeling (domain-adaptive pretraining) pipeline. [file:4]
- `RQs-corresponding-experiments.ipynb`: Experiments aligned with research questions. [file:2]

## Domain-adaptive pretraining (MLM)
The MLM pipeline in `MLM_Main.ipynb`:
- Initializes from `dbmdz/bert-base-turkish-cased`. [file:4]
- Tokenizes with `max_length=512`, padding to max length and truncation enabled. [file:4]
- Uses `DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)` for masking. [file:4]
- Trains with checkpointing enabled and supports resuming from the last checkpoint found in the output directory. [file:4]

### Training configuration (from notebook)
Key settings used in the notebook include:
- `per_device_train_batch_size=16`, `gradient_accumulation_steps=4` [file:4]
- `learning_rate=5e-5`, `weight_decay=0.01`, `warmup_steps=1000` [file:4]
- `save_steps=2000`, `save_total_limit=3`, mixed precision when CUDA is available (`fp16=True`). [file:4]

## Research questions (RQs) and evaluation
`RQs-corresponding-experiments.ipynb` contains code for evaluating re-ranking effects and sensitivity analyses such as varying:
- Candidate pool size (`topP`) and its trade-off between quality and compute. [file:2]
- Re-ranker max sequence length settings (e.g., comparisons across different max token limits). [file:2]

## Dataset source
**Legal text / decisions**
- Source: Official portals and publicly available legal decision repositories (collected from public web pages).  
- Collection method: Selenium-based web scraping.

## Notes on reproducibility
- The training and experiments are notebook-driven; paths and artifacts (checkpoints, final model outputs, evaluation CSV/plots) are configured inside notebooks. [file:4][file:2]
- MLM training saves checkpoints and final model directories as specified in `MLM_Main.ipynb`. [file:4]
