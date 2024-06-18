# Link-Context Learning for Multimodal LLMs
### requirement.txtでtorchなどをインストールする際はcudatoolkitのバージョンと対応したものを自分で設定してください

## 進捗

1. [x] 11/8 一部コードに関する説明を追加，またImagenetによる学習コードを完備．現状はtextエンコーダにはLLaVA v2が，それ以外はLLaVA v1が使用されているらしい．今後のリリース次第ではあるが少しコード変えるだけでいいかも．また学習はFSDPらしい．
2. [x] 11/9 学習したデータについてGradioでデモを実行，一部のISEKAIデータに関してはしっかりとLCLできている．しかしできないものも存在した．また，LCLの性能はまだよいがVQAタスクに関する性能が落ちたように感じる．シングルタスクになっとる？？より詳細に，存在するタスクについて調査，またconfig周りもしっかりと把握する．
3. [x] 11/15 自作データのjsonを修正，プロンプトは要修正ではあるが正しく学習されていることを確認．Qformerに関しての記述はないが，設定すると性能がどのように変わるのか．LLM前に一旦特徴量混ぜておこうという考え？？
4. [x] 11/20 学習コードを修正．自作データでも正しく動作することを確認．検証コードを作成，検証時のプロンプトはまだちょい修正が必要
5. [x] 11/22 ACではMVtec用の検証コード終了
6. [x] 11/27 マルチタスク用に学習コードを変更
7. [x] 12/3 製品あてタスクを学習コードに追加．ノーマル：普通に製品について教えるだけ　LCLバージョン：コンテキストで製品を教える．クエリではコンテキストにある製品かどうかを推論

<p align="center" width="100%">
<img src="ISEKAI_overview.png"  width="80%" height="80%">
</p>

<div>
<div align="center">
    <a href='https://macavityt.github.io/' target='_blank'>Yan Tai<sup>*,1,2</sup></a>&emsp;
    <a href='https://weichenfan.github.io/Weichen/' target='_blank'>Weichen Fan<sup>*,†,1</sup></a>&emsp;
    <a href='https://zhaozhang.net/' target='_blank'>Zhao Zhang<sup>1</sup></a>&emsp;
    <a href='https://zhufengx.github.io/' target='_blank'>Feng Zhu<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=1c9oQNMAAAAJ&hl=zh-CN' target='_blank'>Rui Zhao<sup>1</sup></a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu<sup>&#x2709,3</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>SenseTime Research&emsp;
    <sup>2</sup>Institute of Automation, CAS&emsp;
    <sup>3</sup>S-Lab, Nanyang Technological University&emsp;
    </br>
    <sup>*</sup> Equal Contribution&emsp;
    <sup>†</sup> Project Lead&emsp;
    <sup>&#x2709</sup> Corresponding Author
    
</div>
 
 -----------------

![](https://img.shields.io/badge/ISEKAI-v0.1-darkcyan)
![](https://img.shields.io/github/stars/isekai-portal/Link-Context-Learning)
![](https://black.readthedocs.io/en/stable/_static/license.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fisekai-portal%2FLink-Context-Learning&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Dataset](https://img.shields.io/badge/Dataset-Download-blue)](https://huggingface.co/ISEKAI-Portal) 
[![Generic badge](https://img.shields.io/badge/DEMO-LCL_Demo-<COLOR>.svg)](http://117.144.81.99:20488/)


## Updates
- **05 Sep, 2023**: :boom::boom: We release the code, data, and [LCL-2WAY-WEIGHT](https://huggingface.co/ISEKAI-Portal/LCL_2WAY_WEIGHT) checkpoint.
- **24 Aug, 2023**: :boom::boom: We release the online demo at [🔗LCL-Demo🔗](http://117.144.81.99:20488/).
- **17 Aug, 2023**: :boom::boom: We release the two subsets of ISEKAI (ISEKAI-10 and ISEKAI-pair) at [[Hugging Face 🤗]](https://huggingface.co/ISEKAI-Portal).

---
This repository contains the **official implementation** and **dataset** of the following paper:

> **Link-Context Learning for Multimodal LLMs**<br>
> https://arxiv.org/abs/2308.07891
>
> **Abstract:** *The ability to learn from context with novel concepts, and deliver appropriate responses are essential in human conversations. Despite current Multimodal Large Language Models (MLLMs) and Large Language Models (LLMs) being trained on mega-scale datasets, recognizing unseen images or understanding novel concepts in a training-free manner remains a challenge. In-Context Learning (ICL) explores training-free few-shot learning, where models are encouraged to "learn to learn" from limited tasks and generalize to unseen tasks. In this work, we propose link-context learning (LCL), which emphasizes "reasoning from cause and effect" to augment the learning capabilities of MLLMs. LCL goes beyond traditional ICL by explicitly strengthening the causal relationship between the support set and the query set. By providing demonstrations with causal links, LCL guides the model to discern not only the analogy but also the underlying causal associations between data points, which empowers MLLMs to recognize unseen images and understand novel concepts more effectively. To facilitate the evaluation of this novel approach, we introduce the ISEKAI dataset, comprising exclusively of unseen generated image-label pairs designed for link-context learning. Extensive experiments show that our LCL-MLLM exhibits strong link-context learning capabilities to novel concepts over vanilla MLLMs.*

  
## Todo

1. [x] Release the [ISEKAI-10](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-10) and [ISEKAI-pair](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-pair).
2. [x] Release the dataset usage.
3. [x] Release the demo.
4. [x] Release the codes and checkpoints.
5. [ ] Release the full ISEKAI dataset.
6. [ ] Release checkpoints supporting few-shot detection and vqa tasks.


## Get Start

- [Install](#install)
- [Checkpoint](#checkpoint)
- [Dataset](#dataset)
- [Demo](#demo)

## Install

```shell
conda create -n lcl python=3.10
conda activate lcl
pip install -r requirements.txt
```

### configure accelerate

```shell
accelerate config
```
## Dataset

### ImageNet

We train the LCL setting on our rebuild ImageNet-900 set, and evaluate model on ImageNet-100 set. You can get the dataset json [here](https://github.com/isekai-portal/Link-Context-Learning/tree/main/docs).

### ISEKAI
We evaluate model on ISEKAI-10 and ISEKAI-Pair, you can download ISEKAI Dataset in [ISEKAI-10](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-10) and [ISEKAI-pair](https://huggingface.co/datasets/ISEKAI-Portal/ISEKAI-pair).


## Checkpoint
Download our [LCL-2WAY-WEIGHT](https://huggingface.co/ISEKAI-Portal/LCL_2WAY_WEIGHT/tree/main) and [LCL-MIX](https://huggingface.co/ISEKAI-Portal/LCL-Mix) checkpoints in huggingface. 



## Demo

To launch a Gradio web demo, use the following command. Please note that the model evaluates in the torch.float16 format, which requires a GPU with at least 16GB of memory.

```shell
python ./mllm/demo/demo.py --model_path /path/to/lcl/ckpt
```

It is also possible to use it in 8-bit quantization, albeit at the expense of sacrificing some performance.

```shell
python ./mllm/demo/demo.py --model_path /path/to/lcl/ckpt --load_in_8bit
```

## Train

After preparing [data](https://github.com/shikras/shikra/blob/main/docs/data.md), you can train the model using the command:

### LCL-2Way-Weight
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_train_2way_weight.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint
```

### LCL-2Way-Mix
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_train_mix1.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/init/checkpoint
```
## Inference

After preparing [data](#dataset), you can inference the model using the command:

### ImageNet-100
```shell
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/lcl_eval_ISEKAI_10.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint
```

mmengine style args and huggingface:Trainer args are supported. for example, you can change eval batchsize like this:

### ISEKAI
```shell
# ISEKAI10
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_multi_pope.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint \
        --per_device_eval_batch_size 1

# ISEKAI-PAIR
accelerate launch --num_processes 4 \
        --main_process_port 23786 \
        mllm/pipeline/finetune.py \
        config/shikra_eval_multi_pope.py \
        --cfg-options data_args.use_icl=True \
        --cfg-options model_args.model_name_or_path=/path/to/checkpoint \
        --per_device_eval_batch_size 1
```

where `--cfg-options a=balabala b=balabala` is mmengine style argument. They will overwrite the argument predefined in config file. And `--per_device_eval_batch_size` is huggingface:Trainer argument.

the prediction result will be saved in `output_dir/multitest_xxxx_extra_prediction.jsonl`, which hold the same order as the input dataset. 

## Cite

```bibtex
@article{tai2023link,
  title={Link-Context Learning for Multimodal LLMs},
  author={Tai, Yan and Fan, Weichen and Zhang, Zhao and Zhu, Feng and Zhao, Rui and Liu, Ziwei},
  journal={arXiv preprint arXiv:2308.07891},
  year={2023}
}
```
