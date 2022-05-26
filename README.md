# Inspecting the Factuality of Hallucinations in Abstractive Summarization

This directory contains code necessary to replicate the training and evaluation for the ACL 2022 paper ["Hallucinated but Factual! Inspecting the Factuality of Hallucinations in Abstractive Summarization"](https://arxiv.org/pdf/2109.09784.pdf) by [Meng Cao](https://mcao516.github.io/), [Yue Dong](https://www.cs.mcgill.ca/~ydong26/) and [Jackie Chi Kit Cheung](https://www.cs.mcgill.ca/~jcheung/).

## Dependencies and Setup
The code is based on Huggingface's [Transformers](https://github.com/huggingface/transformers) library. 
  ```
  git clone https://github.com/mcao516/EntFA.git
  cd ./EntFA
  pip install -r requirements.txt
  python setup.py install
  ```

## How to Run
Conditional masked language model (CMLM) checkpoint can be found [here](https://drive.google.com/drive/folders/10ibVc5R7q4Gc0TH1AIRo7IaLCV83SkpF?usp=sharing). For masked language model (MLM), check Fairseq's [BART](https://github.com/pytorch/fairseq/tree/main/examples/bart) repository.

### Train

### Evaluation
