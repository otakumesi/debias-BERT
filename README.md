debias-BERT
---

This repository contains codes for our papers "文表現の摂動正規化: 事前学習済みモデルのDebias手法" in ANLP2021 (言語処理学会第27回年次大会).  
The codes is to fine-tune BERT for debias and to visualize bias metrics on pre-trained language models.  

## Overview
This repository has some experiments.
* Sentence Pertubation Normalizer (will be published at ANLP2021)
* Bias Subspace Transformation of the word embedding in BERT Embeddings
* etc...

#### about `*.ipynb`
Related with ANLP
* Analyze_GLUE.ipynb
* Analyze_crows_pairs.ipynb

Others
* Model_Analyze.ipynb



## notes
The environment of this repository is builded by poetry.  
If you execute my experiments, you put appropriate data into `/data`, and you should run tasks in `/tasks`.  
Please ask me at Issue if you have anything you don't know.  
