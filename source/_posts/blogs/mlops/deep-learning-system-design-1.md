---
title: "Deep Learning System Design - A Checklist (Part I)"
excerpt: "An Overview of how to design a full-stack Deep Learning System"
date: 2024/02/09
categories:
  - Blogs
tags:
  - Deep Learning
  - System Design
layout: post
mathjax: true
toc: true
---

### Introduction

After reviewing many blog posts and working on several DL-based projects myself, I\'ve compiled a list of must-do\'s for a robust, complete Deep Learning System. In general, when we consider a DL system to be "complete", it needs to have the following components:

1. Data
2. Modeling
3. Training & Optimization
4. Experiments
5. Packaging and Deployment
6. Serving
7. Monitoring

I\'ll walk through each step and provide checklists for each of them, detailing rationales and provide examples wherever possible.

### Step 1: Data

#### Data Source

- What is the availability of data?
- What is the size/scale of data?
- Do we have user feedback data?
- Do we use system/operation data (logs? API req/resp?)
- Are there privacy issues?
- A note about logs: Store logs for as long as they are useful, and can discard them when they are no longer relevant for you to debug your current system

#### Data ETL

- What is the data size before/after transformation, this often involves granularity
- What is the data format?
  - JSON
  - CSV (Row-format)
  - Parquet (Column format, Hadoop, AWS Redshift)
  - Row-major vs Column-major
    - Overall, row-major formats are better when you have to do a lot of writes, whereas column-major ones are better when you have to do a lot of column-based reads.
    - Note: `Pandas` is column-major, `NumPy` is row-major by default (if not specified). Access `Pandas DataFrame` rows are faster after we do `df.to_numpy()`
  - Model related:
    - Metadata
    - Training data
    - Monitoring data (sometimes for iterative deployment with model updates)
- Where is the data stored (Cloud? Local? Edge?)
  - Most of the time, it is cloud. Afterall, it costs little for school-level project to store data in AWS S3.
  - Consider spliting app-related dat from model-related data (e.g. WandB vs MongoDB)
- Processing
  - Recall **ACID** and **BASE**
  - Tranactional: OLTP
    1. low latency (often for streaming service)
    2. high availability
    3. transaction won\'t go through if system cannot process it
    4. Often row-major
    5. Eventual consistency
  - Analytical: OLAP
    1. Tolerant to higher query latency (often require trasnformation)
    2. less available: can afford some downtime
    3. delayed operation, but will go through during system overload
    4. Often uses a columnar storage format for better query performance.
    5. Strong consistency

#### Data Routine

- ETL daily routine
- Example: using Airflow

#### Data Quality & Data Validation

- Are the feature information complete? Any missing data?
- Is the training/testing data fully labeled? (can we use self-supervised to do ML-based annotation?)
- Are there data drifts? Are there bias in the data? Packages to detect them?
- Routine to validate data?
- Example: use `pandera` package

```python
import pandera as pa
from azureml.core import Run

run = Run.get_context(allow_offline=True)

if run.id.startswith("OfflineRun"):
    import os

    from azureml.core.dataset import Dataset
    from azureml.core.workspace import Workspace
    from dotenv import load_dotenv

    load_dotenv()

    ws = Workspace.from_config(path=os.getenv("AML_CONFIG_PATH"))

    liko_data = Dataset.get_by_name("liko_data")
else:
    liko_data = run.input_datasets["liko_data"]

df = liko_data.to_pandas_dataframe()

# ---------------------------------
# Include code to prepare data here
# ---------------------------------

liko_data_schema = pa.DataFrameSchema({
    "Id": pa.Column(pa.Int, nullable=False),
    "AccountNo": pa.Column(pa.Bool, nullable=False),
    "BVN": pa.Column(pa.Bool, nullable=True, required=False),
    "IdentificationType": pa.Column(pa.String checks=pa.Check.isin([
        "NIN", "Passport", "Driver's license"
    ]),
    "Nationality": pa.Column(pa.String, pa.Check.isin([
        "NG", "GH", "UG", "SA"
    ]),
    "DateOfBirth": pa.Column(
        pa.DateTime,
        nullable=True,
        checks=pa.Check.less_than_or_equal_to('2000-01-01')
    ),
    "*_Risk": pa.Column(
        pa.Float,
        coerce=True,
        regex=True
    )
}, ordered=True, strict=True)

run.log_table("liko_data_schema", liko_data_schema)
run.parent.log_table("liko_data_schema", liko_data_schema)

# -----------------------------------------------
# Include code to save dataframe to output folder
# -----------------------------------------------

##### Downstream task
liko_data_schema.validate(data_sample)
```

### Step 2: Modeling

#### Model selection

- Start with model suitable for the task -> task categorization
  1.  with/without label/partial label
  2.  numeric/categorical output
  3.  generation/prediction (for generation you need to learn the latent space)
- Baseline selection
  1.  Random Baseline
  2.  Human Heuristic
  3.  Simplest ML model

#### Metric Selection

- What is the task type

    <details>
      <summary>Classification Metrics: Binary Classification</summary>
      <ul>
          <li>Accuracy</li>
          <li>Precision</li>
          <li>Recall</li>
          <li>F1 Score</li>
          <li>Area Under the Receiver Operating Characteristic curve (AUC-ROC)</li>
          <li>Area Under the Precision-Recall curve (AUC-PR)</li>
          <li>True Positive Rate (Sensitivity or Recall)</li>
          <li>True Negative Rate (Specificity)</li>
          <li>False Positive Rate</li>
          <li>False Negative Rate</li>
      </ul>
  </details>

    <details>
        <summary>Classification Metrics: Multi-Class Classification</summary>
        <ul>
            <li>Micro/Macro/Average Precision</li>
            <li>Micro/Macro/Average Recall</li>
            <li>Micro/Macro/Average F1 Score</li>
            <li>Confusion Matrix</li>
            <li>Multi-class Log Loss</li>
            <li>Cohen\'s Kappa</li>
            <li>Jaccard Similarity Score</li>
        </ul>
    </details>

    <details>
        <summary>Regression Metrics</summary>
        <ul>
            <li>Mean Squared Error (MSE)</li>
            <li>Root Mean Squared Error (RMSE)</li>
            <li>Mean Absolute Error (MAE)</li>
            <li>R-squared (Coefficient of Determination)</li>
            <li>Mean Squared Logarithmic Error (MSLE)</li>
            <li>Mean Absolute Percentage Error (MAPE)</li>
            <li>Huber Loss</li>
        </ul>
    </details>

    <details>
        <summary>Clustering Metrics</summary>
        <ul>
            <li>Silhouette Score</li>
            <li>Davies-Bouldin Index</li>
            <li>Calinski-Harabasz Index</li>
            <li>Inertia (within-cluster sum of squares)</li>
            <li>Adjusted Rand Index</li>
            <li>Normalized Mutual Information (NMI)</li>
            <li>Homogeneity, Completeness, and V-Measure</li>
        </ul>
    </details>

    <details>
        <summary>Anomaly Detection Metrics</summary>
        <ul>
            <li>Precision at a given recall</li>
            <li>Area Under the Precision-Recall curve (AUC-PR)</li>
            <li>F1 Score</li>
            <li>Receiver Operating Characteristic curve (ROC)</li>
            <li>Area Under the Receiver Operating Characteristic curve (AUC-ROC)</li>
        </ul>
    </details>

    <details>
        <summary>Natural Language Processing (NLP) Metrics</summary>
        <ul>
            <li>BLEU Score</li>
            <li>ROUGE Score</li>
            <li>METEOR Score</li>
            <li>CIDEr Score</li>
            <li>Perplexity</li>
            <li>Accuracy, Precision, Recall for NER tasks</li>
        </ul>
    </details>

    <details>
        <summary>Ranking Metrics</summary>
        <ul>
            <li>Mean Reciprocal Rank (MRR)</li>
            <li>Normalized Discounted Cumulative Gain (NDCG)</li>
            <li>Mean Average Precision</li>
            <li>Precision at K</li>
            <li>Recall at K</li>
        </ul>
    </details>

    <details>
        <summary>Recommender System Metrics</summary>
        <ul>
            <li>Precision at K</li>
            <li>Recall at K</li>
            <li>Mean Average Precision (MAP)</li>
            <li>Bayesian Personalized Ranking (BPR)</li>
            <li>Root Mean Squared Error (RMSE) for collaborative filtering</li>
        </ul>
    </details>

    <details>
        <summary>Image Segmentation Metrics</summary>
        <ul>
            <li>Intersection over Union (IoU)</li>
            <li>Dice Coefficient</li>
            <li>Pixel Accuracy</li>
            <li>Mean Intersection over Union (mIoU)</li>
            <li>F1 Score</li>
        </ul>
    </details>

    <details>
        <summary>Time Series Forecasting Metrics</summary>
        <ul>
            <li>Mean Absolute Error (MAE)</li>
            <li>Mean Squared Error (MSE)</li>
            <li>Root Mean Squared Error (RMSE)</li>
            <li>Mean Absolute Percentage Error (MAPE)</li>
            <li>Symmetric Mean Absolute Percentage Error (SMAPE)</li>
            <li>Mean Directional Accuracy (MDA)</li>
        </ul>
    </details>

    <details>
        <summary>Reinforcement Learning Metrics</summary>
        <ul>
            <li>Average Reward</li>
            <li>Discounted Sum of Rewards</li>
            <li>Entropy of Policy</li>
            <li>Exploration-Exploitation Tradeoff Metrics</li>
        </ul>
    </details>

- What is the business objective
- Imbalance and Cost Sensitivtiy
- Threshold Selection
- Data Type
- Interpretability
- Robustness

#### (IMPT!) Evaluation methods for Model comparison & Model quality control

1. When drawing conclusion about model performance, consider **Students t-test**
2. Perturbation test (corruption, adversarial attack)
3. Invariance test (Bias removal)
4. Directional Expectation test (Common sense directions. E.g.: rainy season shouldn\'t have much higher temperature than dry season)
5. Model calibration (when standalone probability in the output matters) [see page 10](https://docs.google.com/document/d/1_kxf0xRAushBEepKh57mHubWCkVuS37iAqwmJcYFoJE/edit)
6. Confidence Evaluation (usefulness threshold for each individual prediction)
7. Slice-based Evaluation (model performance on subgroups)

### Step 3: Training & Optimization

- On what platform is the model trained?
- Do we use distribtued training?
- What are the potential issues
  1.  Hardware (GPU memory, inter-GPU communication speed)
  2.  Overfitting/underfitting
  3.  Concept Drift
  4.  training stability (less fluctuations)
  5.  dead neuron
  6.  Local minima
  7.  vanishing/exploding gradients
- How to do debugging
  1.  Start simple and gradually add more components
  2.  (\*)Overfit a single batch: If model can\'t overfit a small amount of data, there\'s something wrong with your implementation.
  3.  Set seed properly
- Is hyperparameter tuning needed? Setup routine for tuning?
- How to optimize the training to make it feasible/efficient/fault tolerant?
  1.  Mixed Precision
  2.  Quantization
  3.  FSDP/DDP/Tensor/Model/Pipeline
  4.  Checkpointing
  5.  Accumulation
  6.  Knowledge Distillation
  7.  PEFT? (LoRA, Prefix Tuning)
- What optimizer do we use? its scheduler?
- What loss do we use?
  - Most of the time it is just same as metrics
  - other scenarios include:
    - Reconstruction loss: mean squared error (MSE) for continuous data or binary cross-entropy for binary data
    - KL Divergence
    - Contrastive Loss: Encourages similarity between augmented versions of the same sample and dissimilarity between different samples. (Siamese Networks / Triplet Loss / SimCLR / Contrastive Divergence Loss (restricted boltzmann machine))

### Step 4: Experiments

Experiment tracking are important, especially when the scale of training is large, and teamwork is involved. There are a lot of tracking tools available out there. `kubeflow`, `mlflow`, `wandb`, `neptune.ai`... you name it. When using these tools, what\'s critical is to consider things to keep track of.

#### Must-have\'s

- **Code**:
  - Preprocessing + training + evaluation scripts,
  - Notebooks for feature engineering
  - Other util codes
- **Environment**:
  - Save the environment configuration files like `Dockerfile` (Docker), `requirements.txt` (pip), `pyproject.toml` (e.g., hatch and poetry), or `conda.yml` (conda).
  - (IMPT) Save Docker images on Docker Hub or your own container repository is always a good practice before running experiment
- **Data**:
  - Saving data versions (as a hash or locations of immutable data resources)
  - You can also use modern data versioning tools like DVC (and save the .dvc files to your experiment tracking tool).
- **Parameters**:
  - Experiment run’s configuration
  - Save parameters used via the command line (e.g., through argparse, click, or hydra)
- **Metrics**:
  - Logging evaluation metrics on train, validation, and test sets for every run.

#### Task-specific

1. Traditional ML

- Model weights
- Evaluation charts (ROC curves, Confusion matrix)
- Prediction distributions

2. Deep Learning

- Model checkpoints (both during and after training, but beware of the cost)
- Gradient norms (to control for vanishing or exploding gradient problems)
- Best/worst predictions on the validation and test set after training
- Hardware resources: CPU/GPU Utility, Memory Utility, Disk I/O, Network Utility, throughput

3. Computer Vision

- Model predictions after every epoch (labels, overlayed masks or bounding boxes)

4. NLP, LLM

- Inference time
- Prompts (in the case of generative LLMs)
- Specific evaluation metrics (e.g., ROUGE for text summarization or BLEU for translation between languages)
- Embedding size and dimensions, type of tokenizer, and number of attention heads (when training transformer models from scratch)
- Feature importance, attention-based, or example-based explanations (see this [overview](https://arxiv.org/pdf/2309.01029.pdf) for specific algorithms and more ideas)

5. Structure Data

- Input data snapshot ( `.head()` on DataFrames if you are using pandas)
- Feature importance (e.g., permutation importance)
- Prediction explanations like SHAP or partial dependence plots (they are all available in DALEX)

6. RL

- Episode info: return, length, intermediate states
- Total environment steps, wall time, steps per second
- Value and police function losses
- Aggregate statistics over multiple environments and/or runs

7. Hyper Optim

- Run score: the metric you are optimizing after every iteration
- Run parameters: parameter configuration tried at each iteration
- Best parameters: best parameters so far and overall best parameters after all runs have concluded
- Parameter comparison charts: there are various visualizations that you may want to log during or after training, like parallel coordinates plot or slice plot (they are all available in Optuna, by the way)

### To be continued...

Usually this is where the school-based projects will end. However, to fully develop a system based on the model and the idea you derive, you still need some engineering skills in building the pipeline for deployment, serving and performance monitoring. Luckily we have a lot of tools for these that do the dirty works for us. Nonetheless, not paying attention to some details in these steps could lead to serious bugs or even cost you thousands of dollars!

Hence to both remind myself and give some suggestions for the readers, I\'ve created a part II for the checklist for these steps. Please check it out if you are interested!
