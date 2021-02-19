debias-BERT
---

This repository contains codes for our papers "文表現の摂動正規化: 事前学習済みモデルのDebias手法" in ANLP2021 (言語処理学会第27回年次大会).  
The codes is to fine-tune BERT for debias and to visualize bias metrics on pre-trained language models.  

## Overview
...

## Setup
We use poetry for building environments.  
If you can use poetry, you execute commands below.  

```
poetry install

wget https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv -o data/crows_pairs_anonymized.csv
```
