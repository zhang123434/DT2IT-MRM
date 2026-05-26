# DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling

DT2IT-MRM addresses key limitations in existing multimodal preference data, including textual style bias, limited preference-strength diversity, unreliable preference signals, and noise in open-source preference datasets. The framework integrates:

1. **Debiased Preference Distillation** for constructing single-image preference pairs.
2. **Text-to-Image Preference Reformulation** for converting T2I preference data into discriminative MRM training data.
3. **Iterative Training and Data Curation** for progressively improving both reward models and preference data quality.

---

## Key Components

### 1. Debiased Preference Distillation

The debiased preference distillation pipeline constructs single-image multimodal preference data.

Relevant files:

```text
preference_data_construction/
├── get_pairwise_result.py
├── get_pointwise_result.py
├── prompt_pairwise.py
└── prompt_pointwise.py
````

Main functions:

* Generate or process listwise scoring results.
* Perform pointwise scoring for preference reliability.
* Use structured prompts for multimodal response evaluation.

---

### 2. Text-to-Image Preference Reformulation

DT2IT-MRM reformulates text-to-image preference data into a multimodal pairwise evaluation format that can be used for training discriminative multimodal reward models.

Relevant files:

```text
preference_data_construction/
├── T2I_reformulation.py
````

---

### 3. Iterative Training and Data Curation

The iterative training framework progressively curates noisy open-source multimodal preference data using trained MRMs and MLLM-based re-annotation.

Relevant files:

```text
iterative_training/
├── construct_mrm_scoring_input.py
├── get_first_step_data.py
├── get_second_step_data_and_third_step_input.py
├── get_third_step_data.py
├── inference_qwen3_vl_single_gpu.sh
└── qwen3_vl_single_gpu.py
```

Main functions:

* Construct MRM scoring inputs.
* Perform the first-step preference correction.
* Filter preference pairs using MRM consistency.
* Construct third-step MLLM annotation inputs.
* Merge curated preference data for iterative training.

---

### 4. Reward Model Training and Evaluation

We build DT2IT-MRM on top of Qwen3-VL(default)/Qwen2.5-VL backbones and train discriminative multimodal reward models using LLaMA-Factory.

Relevant files:

```text
script/
├── train/
│   └── demo_rm_qwen3_vl.yaml
└── eval/
    └── demo_rm_qwen3_vl_test.yaml
```

---

## Repository Structure

```text
DT2IT-MRM/
├── iterative_training/              # Iterative preference data curation scripts
├── preference_data_construction/    # Debiased preference distillation scripts, prompts, and text-to-image reformulation scripts
├── script/
│   ├── train/                       # Training configuration files
│   └── eval/                        # Evaluation configuration files
├── compute_metric.py                # compute metric in three benchmarks
└── README.md
```

---

## Installation

```bash
conda create -n dt2it-mrm python=3.11 -y
conda activate dt2it-mrm
bash install.sh
```

---

## Data Preparation

The training data should follow the preference format used by LLaMA-Factory reward modeling.

A typical preference sample is:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "<image>USER_INSTRUCTION"
    }
  ],
  "chosen": {
    "from": "gpt",
    "value": "CHOSEN_RESPONSE"
  },
  "rejected": {
    "from": "gpt",
    "value": "REJECTED_RESPONSE"
  },
  "images": [
    "IMAGE_PATH"
  ]
}
```

---

## Training

An example training config is provided in:

```text
script/train/demo_rm_qwen3_vl.yaml
```

Preparation before training:
* prepare all training image and data(.json file)
* following LLaMA-Factory, configure the `dataset_info.json` file in the `/path/to/data_dir`

Run reward model training with LLaMA-Factory:

```bash
cd LlamaFactory-0.9.4
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
FORCE_TORCHRUN=1 llamafactory-cli train ../script/train/demo_rm_qwen3_vl.yaml
```

Please modify the following fields before training:

```yaml
model_name_or_path: /path/to/Qwen3-VL-8B-Instruct
dataset: train_dataset_name
dataset_dir: /path/to/data_dir
output_dir: /path/to/output_dir
```

---

## Evaluation

An example evaluation config is provided in:

```text
script/eval/demo_rm_qwen3_vl_test.yaml
```

Preparation before evaluation:
* prepare all test image and data(.json file)
* following LLaMA-Factory, configure the `dataset_info.json` file in the `/path/to/data_dir`

Run evaluation with:

```bash
cd LlamaFactory-0.9.4
FORCE_TORCHRUN=1 llamafactory-cli train ../script/train/demo_rm_qwen3_vl_test.yaml
```

Please modify the following fields before training:

```yaml
model_name_or_path: /path/to/checkpoint
dataset: test_dataset_name
dataset_dir: /path/to/data_dir
output_dir: /path/to/output_dir
```

Compute metric:
```bash
python compute_metric.py
```

---

## Preference Data Construction (Initial Preference Pair Construction)

### Listwise Scoring

```bash
python preference_data_construction/get_pairwise_result.py
```

### Pointwise Scoring

```bash
python preference_data_construction/get_pointwise_result.py
```

The corresponding prompt templates are provided in:

```text
preference_data_construction/prompt_pairwise.py
preference_data_construction/prompt_pointwise.py
```

### Text-to-image Preference Reformulation

```bash
python preference_data_construction/T2I_reformulation.py
```

Before running these scripts, please configure:

* input data path
* image root path
* output path
* API key or API endpoint
* model name for preference scoring

---

## Iterative Data Curation

The iterative data curation pipeline consists of multiple steps.

### Construct MRM Scoring Inputs

```bash
python iterative_training/construct_mrm_scoring_input.py
```

### Run MRM Scoring

```bash
bash iterative_training/inference_qwen3_vl_single_gpu.sh
```

### Step 1: Multi-MRM Voting & Label Flipping

```bash
python iterative_training/get_first_step_data.py
```

### Step 2: MRM-based Consistency Check & Prepare Step 3 Input

```bash
python iterative_training/get_second_step_data_and_third_step_input.py
```

### Step 3: Process MLLM annotation result

```bash
python iterative_training/get_third_step_data.py
```

Please edit the input and output paths in each script before running.
