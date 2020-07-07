# EA-WGCN
A Weighted GCN with Logical Adjacency Matrix for Relation Extraction (ECAI 2020)
PyTorch implementation of Deep Learning approach for relation extraction task(SemEval 2010 Task 8 and TACRED) via a Weighted Graph Convolutional Neural Networks with Entity-Attention (EA-WGCN).

This paper proposes a novel weighted graph convolutional network by constructing a logical adjacency matrix which effectively solves the feature fusion of multi-hop relation without additional layers and parameters for relation extraction task. Experimental results show that our model can take better advantage of the structural information in the dependency tree and produce better results than previous models.

You can find the paper [here](http://ecai2020.eu/papers/957_paper.pdf).

# Model Architecture 
![EA-WGCN model](https://github.com/balabala1/LeetCode/blob/master/Imgs/model.png)
 
# Requirements
- Python 3 (tested on 3.7.3)
- PyTorch (tested on 1.1.0)
- tqdm, pickle

# Usage
## Preparation
We evaluate the performance of our model on TACRED and SemEval 2010 Task 8 datasets. This code needs to use the TACRED dataset(LDC license required). If you get the TACRED dataset, please put the JSON files under the directory `dataset/tacred`. In this repository, we provide only sample data from TACRED dataset. If you want to change the data set, you can go [here](http://semeval2.fbk.eu/semeval2.php) to get the SemEval 2010 Task 8 datasets. But you need to do your own data preprocessing.

First, since the pre-training [GLoVe](https://nlp.stanford.edu/projects/glove/) vectors is applied, you need to download from the Stanford NLP group website and unzip it to the directory `dataset/glove`.

Then prepare vocabulary and initial word vectors with:
```
python3 prepare_vocab.py dataset/tacred dataset/vocab dataset/glove
```
The vocabulary and word vectors can be saved under this directory `dataset/vocab`.

## Training
To train a EA-WGCN model, run:
```
python3 train.py
```
Model checkpoints and logs will be saved to `./saved_models`. For details on the use of other parameters, please refer to `train.py`.
## Evaluation
To run evaluation on the test set, run:
```
python3 evaluate.py --model_dir saved_models/best_model
```
This will use the `best_model.pt` file to evaluate the model by default.
## Ensemble
To run an ensemble model, run:
```
python3 ensemble.py
```
You need to specify the parameter `model_file_list` in the `ensemble.py` as shown in the following example:
```
model_file_list = ['saved_models/01',
                   'saved_models/02',
                   'saved_models/03',
                   'saved_models/04',
                   'saved_models/05']                 
```
Store the trained model files under the all `saved_models/XX`.
# Citation
```
@inproceedings{zhou2020eawgcn,
 author = {Zhou, Li and Wang, Tingyu and Qu, Hong and Huang, Li and Liu, Yuguo},
 booktitle = {European Conference on Artificial Intelligence(ECAI)},
 title = {A Weighted GCN with Logical Adjacency Matrix for Relation Extraction},
 url = {http://ecai2020.eu/papers/957_paper.pdf},
 year = {2020}
}
```
