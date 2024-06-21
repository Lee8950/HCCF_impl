# Machine Learning - Pytorch Reproduction of HCCF

## Introduction
This is an unofficial implementation for Hypergraph Contrastive Collaborative Filtering - SIGIR 2022. It's a group work for machine learning course@School of Artificial Intelligence@Sun Yat-sen University.

## Authors
```text
709064332@qq.com - Li
2461499240@qq.com - Zhou
```
## Usage
```bash
./run.py --data [yelp|amazon|ml10m]
```

## Notice
Since the official implementation is quite different from the description in their paper, for example in the loss part, loss is define as $max(0, 1 - Pr)$, however they used $-log(sigmoid(Pr))$ to replace the original one. We stick to the paper-description and try to reproduct the real performance.

## Citations
```latex
@inproceedings{hccf2022,
  author    = {Xia, Lianghao and
               Huang, Chao and
	       Xu, Yong and
	       Zhao, Jiashu and
	       Yin, Dawei and
	       Huang, Jimmy Xiangji},
  title     = {Hypergraph Contrastive Collaborative Filtering},
  booktitle = {Proceedings of the 45th International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2022, Madrid,
               Spain, July 11-15, 2022.},
  year      = {2022},
}
```