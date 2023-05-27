# DMASTE
Measuring Your ASTE Models in The Wild: A Diversified Multi-domain Dataset For Aspect Sentiment Triplet Extraction. Ting Xu, Huiyun Yang, Zhen Wu, Jiaze Chen, Fei Zhao, Xinyu Dai. In Findings of ACL, 2023.

## Data
The original datasets are released by the paper [1]. [Download](abc)

Data format descriptions are [here](abc).

## Usage
This repository contains the re-implementation of four baseline models for ASTE. The four models included in this implementation are:
* Span-ASTE[2]: a tagging-based method. It explicitly considers the span interaction between the aspect and opinion terms.
* GAS[3]: a generation-based method. It transforms the ASTE tasks
into a text generation problem.
* BART-ABSA[4]: a generation-based method. It employs a pointer network and generates indices of the aspect term, opinion
term, and sentiment polarity sequentially.
* BMRC[5]: a MRC-based method. It extracts aspect-oriented triplets and opinion-oriented triplets. Then it obtains
the final results by merging the two directions.

The source codes for these models are obtained from the original papers and their publicly released codes, with subtle modifications made to adapt them to triplets containing both implicit and explicit aspect terms. 

Detailed running scripts and instructions for each model can be found in their respective subdirectories within this repository. Please refer to the README file provided in each model's directory for specific usage guidelines, examples, and information on how to run the models.

## Citation
If you used the datasets or code, please cite our paper:
```
```
## References
[1] Measuring Your ASTE Models in The Wild: A Diversified Multi-domain Dataset For Aspect Sentiment Triplet Extraction. Ting Xu, Huiyun Yang, Zhen Wu, Jiaze Chen, Fei Zhao, Xinyu Dai. In Findings of ACL, 2023.

[2] Learning span-level interactions for aspect sentiment triplet extraction. Lu Xu, Yew Ken Chia, and Lidong Bing. In ACL, 2021. 

[3] Towards generative aspect-based sentiment analysis. Wenxuan Zhang, Xin Li, Yang Deng, Lidong Bing, and
Wai Lam. In ACL, 2021.

[4] A unified generative framework for aspect-based sentiment analysis. Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, and Zheng
Zhang. In ACL, 2021.

[5] Bidirectional machine reading comprehension
for aspect sentiment triplet extraction. Shaowei Chen, Yu Wang, Jie Liu, and Yuelin Wang. In AAAI, 2021. 

# 文件说明

## 数据集相关
* amazon: 来自ASTE-Data-V2: Position-Aware Tagging for Aspect Sentiment Triplet Extraction的数据集和我们数据集的无监督版本
* dataset: 我们的数据集DMASTE和ASTE-Data-V2，其中 all表示DMASTE中四个源领域数据的总和 
    * 数据格式： sentence####[(aspect, opinion, sentiment), ....]####category
    * 读取方式： 
```
lines = []
with open(file_name) as f:
    lines = f.readlines()
    lines = [x.split('####') for x in lines]
for line in lines:
            sentence, triples = line[:2]
            triples = eval(triples)
```
* eq-dataset: 控制每个领域的训练集大小相同，用于分析迁移性能和领域相似性的关系
* multi-dataset: 4个源领域数据分别组合，用于multi-source cross-domain的研究 

* ia-dataset: 本文的方法对隐式aspect的处理是在数据层面的，在句子前拼接[ia]表示implicit aspect，并移动标签中aspect,opinion的下标
* ia-eq-dataset
* ia-multi-dataset 

## 模型相关
* analyse: 用于分析数据集和统计实验结果的代码
* BMARTABSA: A Unified Generative Framework for Aspect-Based Sentiment Analysis 论文的实现
* BMRC: Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction
    * 包含DANN对抗的实现
* Generative-ABSA: Towards Generative Aspect-Based Sentiment Analysis
* GTS: Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction
* Span-ASTE: Learning Span-Level Interactions for Aspect Sentiment Triplet Extraction
* mySpan-ASTE: 我自己对Span-ASTE代码的重构
    * 包含DANN对抗的实现

