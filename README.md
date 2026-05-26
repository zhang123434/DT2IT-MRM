# DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling

<p align="center">
  <a href="https://arxiv.org/abs/2604.19544"><img src="https://img.shields.io/badge/arXiv-2604.19544-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/zhang123434/DT2IT-MRM"><img src="https://img.shields.io/badge/Code-GitHub-blue.svg" alt="Code"></a>
  <a href="#data-and-model-release"><img src="https://img.shields.io/badge/Data-Coming%20Soon-yellow.svg" alt="Data"></a>
  <a href="#data-and-model-release"><img src="https://img.shields.io/badge/Model-Coming%20Soon-yellow.svg" alt="Model"></a>
</p>

This repository contains the official implementation of **DT2IT-MRM**, a data-centric framework for **Multimodal Reward Modeling (MRM)**.

> **DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling**  
> Zhihong Zhang, Jie Zhao, Xiaojian Huang, Jin Xu, Zhuodong Luo, Xin Liu, Jiansheng Wei, Xuejin Chen  
> [arXiv:2604.19544](https://arxiv.org/abs/2604.19544)

DT2IT-MRM addresses key limitations in existing multimodal preference data, including textual style bias, limited preference-strength diversity, unreliable preference signals, and noise in open-source preference datasets. The framework integrates:

1. **Debiased Preference Distillation** for constructing single-image preference pairs.
2. **Text-to-Image Preference Reformulation** for converting T2I preference data into discriminative MRM training data.
3. **Iterative Training and Data Curation** for progressively improving both reward models and preference data quality.

---

## News

- **[2026/04/21]** Our paper is available on arXiv: [DT2IT-MRM](https://arxiv.org/abs/2604.19544).
- **[Coming Soon]** We will release the processed preference data.
- **[Coming Soon]** We will release the trained DT2IT-MRM checkpoints.

---

## Data and Model Release

We plan to release the following resources for research use:

| Resource | Status | Link |
|---|---|---|
| DT2IT-MRM model checkpoint | Coming soon | `PLACEHOLDER_MODEL_LINK` |
| Processed preference data | Coming soon | `PLACEHOLDER_DATA_LINK` |
| Training configs | Available in this repo | [`script/train`](script/train) |
| Evaluation configs | Available in this repo | [`script/eval`](script/eval) |

> The data and model links are currently placeholders. They will be updated after the release is ready.  
> For datasets derived from existing artifacts, we will only release components permitted by the original licenses and terms of use.

---

## Key Components

### 1. Debiased Preference Distillation

The debiased preference distillation pipeline constructs single-image multimodal preference data with reduced textual style bias and improved preference reliability.

Relevant files:

```text
preference_data_construction/
├── get_pairwise_result.py
├── get_pointwise_result.py
├── prompt_pairwise.py
└── prompt_pointwise.py
````

Main functions:

* Generate or process pairwise/listwise scoring results.
* Perform pointwise scoring for preference reliability.
* Use structured prompts for multimodal response evaluation.

---

### 2. Text-to-Image Preference Reformulation

DT2IT-MRM reformulates text-to-image preference data into a multimodal pairwise evaluation format that can be used for training discriminative multimodal reward models.

The released data processing scripts and formatted datasets will be updated in future releases.

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

We build DT2IT-MRM on top of Qwen-VL backbones and train discriminative multimodal reward models using LLaMA-Factory.

Relevant files:

```text
script/
├── train/
│   └── demo_rm_qwen3_vl.yaml
└── eval/
    └── demo_rm_qwen3_vl_test.yaml
```

The training code is based on:

```text
LlamaFactory-0.9.4/
```

---

## Repository Structure

```text
DT2IT-MRM/
├── LlamaFactory-0.9.4/              # Training framework for reward model training
├── iterative_training/              # Iterative preference data curation scripts
├── preference_data_construction/    # Debiased preference distillation scripts and prompts
├── script/
│   ├── train/                       # Training configuration files
│   └── eval/                        # Evaluation configuration files
└── README.md
```

---

## Installation

```bash
git clone https://github.com/zhang123434/DT2IT-MRM.git
cd DT2IT-MRM
```

Create the environment:

```bash
conda create -n dt2it-mrm python=3.10 -y
conda activate dt2it-mrm
```

Install dependencies:

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers accelerate deepspeed
pip install qwen-vl-utils flash-attn --no-build-isolation
pip install openai httpx pandas tqdm json-repair regex
```

Install LLaMA-Factory:

```bash
cd LlamaFactory-0.9.4
pip install -e .
cd ..
```

> The exact package versions used in our experiments will be updated later.

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

Please replace the following paths in the config files with your local paths:

```yaml
model_name_or_path: /path/to/Qwen3-VL-8B-Instruct
dataset_dir: /path/to/processed_data
output_dir: /path/to/output_model
```

The processed DT2IT-MRM preference data will be released later.

---

## Training

An example training config is provided in:

```text
script/train/demo_rm_qwen3_vl.yaml
```

Run reward model training with LLaMA-Factory:

```bash
cd LlamaFactory-0.9.4

llamafactory-cli train ../script/train/demo_rm_qwen3_vl.yaml
```

Please modify the following fields before training:

```yaml
model_name_or_path: /path/to/base_model
dataset: your_dataset_name
dataset_dir: /path/to/dataset_dir
output_dir: /path/to/output_dir
deepspeed: /path/to/deepspeed_config.json
```

---

## Evaluation

An example evaluation config is provided in:

```text
script/eval/demo_rm_qwen3_vl_test.yaml
```

Run evaluation with:

```bash
cd LlamaFactory-0.9.4

llamafactory-cli train ../script/eval/demo_rm_qwen3_vl_test.yaml
```

Please update the model path, dataset path, and output path in the evaluation config before running.

---

## Preference Data Construction

### Pairwise/Listwise Scoring

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

Before running these scripts, please configure:

* input data path
* image root path
* output path
* API key or API endpoint
* model name for preference scoring

---

## Iterative Data Curation

The iterative data curation pipeline consists of multiple steps.

### Step 1: Construct MRM Scoring Inputs

```bash
python iterative_training/construct_mrm_scoring_input.py
```

### Step 2: Run MRM Scoring

```bash
bash iterative_training/inference_qwen3_vl_single_gpu.sh
```

or

```bash
python iterative_training/qwen3_vl_single_gpu.py
```

### Step 3: First-Step Data Correction

```bash
python iterative_training/get_first_step_data.py
```

### Step 4: Second-Step Filtering and Third-Step Input Construction

```bash
python iterative_training/get_second_step_data_and_third_step_input.py
```

### Step 5: Third-Step Data Construction

```bash
python iterative_training/get_third_step_data.py
```

Please edit the input and output paths in each script before running.

---

## Main Results

DT2IT-MRM achieves strong overall performance across three major multimodal reward model benchmarks:

* VL-RewardBench
* Multimodal RewardBench
* MM-RLHF-RewardBench

It also shows strong effectiveness in downstream applications, including:

* inference-time scaling
* offline reinforcement learning / preference optimization

Please refer to our paper for detailed experimental settings and results.

---

## TODO

* [ ] Release DT2IT-MRM checkpoints.
* [ ] Release processed preference data.
* [ ] Release evaluation outputs.
* [ ] Add full reproduction scripts for benchmark evaluation.
* [ ] Add detailed environment/version information.
* [ ] Add data license and usage documentation.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026dt2itmrm,
  title={DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling},
  author={Zhang, Zhihong and Zhao, Jie and Huang, Xiaojian and Xu, Jin and Luo, Zhuodong and Liu, Xin and Wei, Jiansheng and Chen, Xuejin},
  journal={arXiv preprint arXiv:2604.19544},
  year={2026}
}
```

---

## Acknowledgements

This repository uses LLaMA-Factory for reward model training. We thank the authors of Qwen-VL, LLaMA-Factory, and the open-source multimodal preference datasets and benchmarks used in this work.

---

## License

The code and released artifacts are intended for research use. The license information will be updated before the official release of models and data. For datasets derived from existing sources, please also follow the licenses and terms of use of the original datasets.
[5]: https://arxiv.org/abs/2604.19544 "[2604.19544] DT2IT-MRM: Debiased Preference Construction and Iterative Training for Multimodal Reward Modeling"
[6]: https://github.com/Kwai-YuanQi/MM-RLHF "GitHub - Kwai-YuanQi/MM-RLHF: The Next Step Forward in Multimodal LLM Alignment · GitHub"
