# Activity Grammars for Temporal Action Segmentation (NeurIPS 2023)
### [Project Page](https://jsleeo424.github.io/KARI/) | [Paper](https://openreview.net/forum?id=oOXZ5JEjPb)
This repository contains the official source code and data for our paper:

> [Activity Grammars for Temporal Action Segmentation](https://openreview.net/forum?id=oOXZ5JEjPb)  
> [Dayoung Gong*](https://gongda0e.github.io/),
> [Joonseok Lee*](https://scholar.google.com/citations?user=ZXcSl7cAAAAJ&hl=ko),
> [Deunsol Jung](https://hesedjds.github.io/),
> [Suha Kwak](https://suhakwak.github.io/), and
> [Minsu Cho](http://cvlab.postech.ac.kr/~mcho/)
> , POSTECH,
> NeurIPS, New Orleans, 2023.

## Key-Action-based Recursive Induction (KARI)
The file 'kari.py' includes a code for grammar induction of KARI for both dataset Breakfast and 50 Salads.

### Environment setup
You can create the environments using 'torch110.yaml' file.
> ```bash
> conda env create --file torch110.yaml
> ```

### How to run code for grammar induction
You can choose {DATASET} = {breakfast, 50salads} with the activity source directory {SOURCE_DIR}={source_breakfast, source_50salads} to generated activity grammars in {RESULT_DIR}.
> ```bash
> python kari.py --dataset {DATASET} --source_dir {SOURCE_DIR} --result_dir {RESULT_DIR} 
> ```
#### Grammar induction for Breakfast
1. Induction for each activity
> ```bash
> python kari.py --dataset breakfast --source_dir source_breakfast --result_dir induced_grammars
> ```
2. Induction for all activity
> ```bash
> python kari.py --dataset breakfast --source_dir source_breakfast --result_dir induced_grammars --merge
> ```

#### Grammar induction for 50 Salads
> ```bash
> python kari.py --dataset 50salads --source_dir source_50salads --result_dir induced_grammars --top_K --K 3
> ```

### Grammar results
The KARI-induced grammars for two benchmark datset is saved in the 'induced_grammars/' folder.
### Folder structure
```bash
    ./
    ├── kari.py
    ├── bep.py
    ├── torch110.yaml
    ├── source_breakfast/
    │   ├── mapping.txt
    │   ├── activity_category.txt
    │   ├── split1/
    │   │   ├── coffee.txt
    │   │   ├── milk.txt
    │   │   ├── ...
    │   │   └── pancake.txt
    │   ├── split2/
    │   ├── split3/
    │   └── split4/
    ├── source_50salads/ 
    └── induced_grammars/
        ├── breakfast/
        │   ├── split1/
        │   │   ├── all.pcfg
        │   │   ├── coffee.pcfg
        │   │   ├── milk.pcfg
        │   │   ├── ...
        │   │   └── pancake.pcfg
        │   │        ├── split2/
        │   │        ├── split3/
        │   │        └── split4/
        └── 50salads/
        │   ├── split1/
        │   ├── ...
        │   └── split5/
```



## Breadth-first Earley Parser

The file 'bep.py' includes a code for Breadth-first Earley Parser(BEP), 

### Components

This code includes **State** class and **BreadthFirstEarley** class.


#### State

This class is an implementation of the state in the parser. It contains the rule, current position, parent state, and parsing/prefix probability. It can be initialized through its constructor.

#### BreadthFirstEarley

This class is an implementation of the Breadth-first Earley Parser. It includes a queue to manage the state, a dictionary to store parsed sequences and probabilities according to frames, and a grammar for parsing. Through the constructor, you can decide whether pruning should be conducted in the queue, whether to incorporate transition probability, and specifying the criteria for sorting the queue. `parse` method gets the result of the segmentation model output $Y \in \mathbb{R}^{T \times C}$ as input and generates the refined action sequence and its probability. We use PCFG class from the NLTK library for grammar.


## Using KARI-induced grammars and BEP in your project
Given the output from the segmentation model 'segmentation_model_output', you can use BEP as follows:
>```bash
>    import bep as BEP
>    from utils import read_grammar, read_mapping_dict
> 
>       ...
>       mapping_file = 'source_breakfast/mapping.txt'
>       actions_dict = read_mapping_dict(mapping_file)
>
>       # Model outputs
>       output = model(input) # output: [T, C]
>
>       # Load Grammar
>       grammar_path = 'induced_grammars/breakfast/split1/all.pcfg'
>       grammar = read_grammar(grammar_path, index=True, mapping=actions_dict)
>
>       # Load Parser
>       '''
>       Arguments of BEP:
>        - sample_stride: temporal sampling stride of input
>        - qsize: maximum size of the queue of states
>        - str_len: maximum length of the activity sequence during parsing
>       '''
>       parser = BEP.BreadthFirstEarley(grammar, prior_flag=True, priority='d', prune_prob=True)
>
>       result_sequence, prob = parser.parse(segmentation_model_output[::sample_stride, ], prune=qsize, str_len=str_len)
>       refined_sequence, _, _ = parser.compute_labels_segment(segmentation_model_output)
>       ...
>```

### Configurations
|Dataset|sample_stride|qsize|str_len|top K (N^key)|
|-------|-------------|-----|-------|-------------|
|50 Salads|100|20|25|3|
|Breakfast|50|20|25|-|


## Citation
If you find our code or paper useful, please consider citing our paper:
```BibTeX
@inproceedings{gong2023activity,
  title={Activity Grammars for Temporal Action Segmentation},
  author={Gong, Dayoung and Lee, Joonseok and Jung, Deunsol and Kwak, Suha and Cho, Minsu},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## Reference

This code is based on Generalized Earley Parser, available at https://github.com/Buzz-Beater/GEP_PAMI/tree/master.   
**Paper**: A Generalized Earley Parser for Human Activity Parsing and Prediction,  TPAMI 2020  
**Author**: Siyuan Qi, Baoxiong Jia, Siyuan Huang, Ping Wei, and Song-Chun Zhu

This code uses NLTK library to utilize grammar.  
**Title**: Bird, Steven, Edward Loper and Ewan Klein (2009). Natural Language Processing with Python.  O'Reilly Media Inc.
