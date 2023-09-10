# DMASTE
[Measuring Your ASTE Models in The Wild: A Diversified Multi-domain Dataset For Aspect Sentiment Triplet Extraction](https://aclanthology.org/2023.findings-acl.178) (Xu et al., Findings 2023)
## Data
The original datasets are released by the paper [1]. [Download](https://github.com/NJUNLP/DMASTE/tree/main/dataset)


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
Ting Xu, Huiyun Yang, Zhen Wu, Jiaze Chen, Fei Zhao, and Xinyu Dai. 2023. Measuring Your ASTE Models in The Wild: A Diversified Multi-domain Dataset For Aspect Sentiment Triplet Extraction. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2837–2853, Toronto, Canada. Association for Computational Linguistics.
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
