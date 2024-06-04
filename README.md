<img src="imgs/logo.jpg" alt="Logo" align="left" width="105"/>

# Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction

![Python 3.9](https://img.shields.io/badge/python-3.9-DodgerBlue.svg?style=plastic)
![License MIT](https://img.shields.io/badge/License-MIT-DodgerBlue.svg?style=plastic)

This repository contains the official code used in paper "Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction".

Please feel free to contact lyutoon@gmail.com if you have any questions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Attack](#attack)
  - [Analysis](#analysis)
- [Clarification](#clarification)
- [Ethics](#ethics)
- [Cite](#cite)

## Installation

To run the code in this repository, follow these steps:

1. Clone this repository:

   ```sh
   git clone https://github.com/LLM-DRA/DRA.git
   cd DRA
   ```

2. Prepare the conda enviroment  (Python 3.9.18 is recommended):

   ```sh
   conda create -n DRA python==3.9.18
   conda activate DRA
   ```

3. Install the requirements

   ```sh
   pip install -r requirements.txt
   ```

4. Install harmbench

   Please refer to https://huggingface.co/cais/HarmBench-Llama-2-13b-cls

## Usage

### Attack

1. Enter the attack directory which contains the attack code.

   ```sh
   cd src/attack
   ```

2. Modify the local model path in `attack_config.json` based on your own requirements.

   ```json
   {
       "llama2-13b": {
           "model_path": "/models/llama2/llama-2-13b-chat-hf"
           },
       "llama2-70b": {
           "model_path": "/models/Llama-2-70b-chat-hf"
           },
       "zephyr": {
           "model_path": "/models/zephyr-7b-beta"
           },
       "mixtral": {
           "model_path": "/models/Mixtral-8x7B-Instruct-v0.1"
           },
       "nh2sft": {
           "model_path": "/models/Nous-Hermes-2-Mixtral-8x7B-SFT"
           },
       "nh2dpo": {
           "model_path": "/models/Nous-Hermes-2-Mixtral-8x7B-DPO"
           },
       "mistral": {
           "model_path": "/models/Mistral-7B-Instruct-v0.2"
           },
       "vicuna": {
           "model_path": "/models/vicuna/Xorbits/vicuna-13b-v1.5"
           },
       "harmbench": {
           "model_path": "/models/HarmBench-Llama-2-13b-cls"
       }
     }
   ```

   Note: if you want to attack GPT, please first set your openai API key:

   ```sh
   export OPENAI_API_KEY=sk-xxxxxx
   ```

3. Launch attack.

- Debug mode (Only generate the attack prompt with one try)

  ```sh
  python3 attack.py \
  --debug \
  --verbose=True \
  --model=[MODEL] \
  --query=[QUERY]
  ```

  `model`: the target model name, choose from {llama, vicuna, gpt-3.5, gpt-4}

  `query`: provided by user. e.g., "how to rob a bank"

- Attack mode (Attack the model)

  ```sh
  python3 attack.py \
  --model=[MODEL] \
  --start=[START_IDX] \
  --end=[END_IDX] \
  --T=20 \
  --em_t=0.75 \
  --save=[SAVED_FILE, xx.csv]
  ```

  `model`: the target model name, choose from {llama, vicuna, gpt-3.5, gpt-4}

  `start`: the start index of the dataset, from 0 to 120

  `end`: the end index of the dataset, from 0 to 120

  `T`: the maximum retry time while attacking the model

  `em_t`: the threshold for checking the metric **em**. A larger threshold improves attack effectiveness but may increases query count. A smaller threshold reduces queries but may lowers effectiveness.

  `save`:  result save path, in csv format

- Attack batch mode (Batched attack the model)

  ```sh
  python3 attack_batch.py \
  --model=[MODEL] \
  --start=[START_IDX] \
  --end=[END_IDX] \
  --T=20 \
  --bs=8 \
  --em_t=0.75 \
  --save=[SAVED_FILE, xx.json]
  ```
  `model`: the target model name, choose from {llama, vicuna}. Note that GPT is not supported in batch mode.

  `start`: the start index of the dataset, from 0 to 120

  `end`: the end index of the dataset, from 0 to 120

  `T`: the maximum retry time while attacking the model

  `em_t`: the threshold for checking the metric **em**. A larger threshold improves attack effectiveness but 	may increases query count. A smaller threshold reduces queries but may lowers effectiveness.

  `bs`: batch size

  `save`: result save path, in json format

### Analysis

Will be released soon!

## Clarification

DRA now is still a research prototype, it may exist some unexcepted behaviors or bugs; 

Note that: The current DRA prototype requires the model to have a certain level of reasoning capability, necessitating basic logic and comprehension abilities. The current template is not well-suited for 7b models.

Meanwhile, DRA is under active development.

Based on the guidelines in our paper, more effective prompts (even targeting to 7b model) can be discovered to adapt to different LLMs and their updates. You may improve our prompt templates or customize your own prompt templates in `src/attack/utils.py`. Note that the prompt template can be improved and in `src/attack/utils.py`, `xxx_DEV` are the templates under improvement.

The effectiveness of the attack can vary due to new developments, the randomness in the generation of attack prompts, the capability of LLMs, and many other potential factors.

## Ethics

DRA has been responsibly disclosed to LLM providers via emails, Github issuses, and risky content feedback forms in a timely manner. This code is for educational and research purposes only and should not be used for illegal or unethical activities. The files in the `results` directory may contain unfiltered content generated by LLM, which could potentially be offensive to readers.

## Cite

```
@article{liu2024making,
  title={Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction},
  author={Liu, Tong and Zhang, Yingjie and Zhao, Zhe and Dong, Yinpeng and Meng, Guozhu and Chen, Kai},
  journal={arXiv preprint arXiv:2402.18104},
  year={2024}
}
```

