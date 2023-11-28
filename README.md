# Complexity-Guided Curriculum Learning for Text Graphs

TGCL is an advanced spaced repetition framework designed to enhance the training efficacy of GNNs and understand their learning dynamics. It schedules examples for training, optimize the timing and sequence (order) based on the evolving complexity of training data. To achieve this, it uses a combination of multiview graph and text complexity formalisms. TGCL can effectively tailor the curriculum to the unique learning dynamics of each model and can learn curricula that are transferable across different GNN models and datasets.

<p align="center">
<img src="https://github.com/CLU-UML/TGCL/blob/main/tgcl.png" width="900" height="450">
</p>


The architecture of the proposed model, TGCL. It takes subgraphs and text(s) of their target node(s)
as input. The radar chart shows graph complexity indices which quantify the difficulty of each subgraphs from
different perspectives (text complexity indices are not shown for simplicity). Subgraphs are ranked according to
each complexity index and these rankings are provided to TGCL scheduler to space samples over time for training.
# Data 

### Node Classification
There are two datasets for node classification: Arxiv and Cora. 

* **Arxiv:** Arxiv is downloaded from ogbn (https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) using following code:

```
from ogb.nodeproppred import PygNodePropPredDataset

dataset = PygNodePropPredDataset(name = d_name) 

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
graph = dataset[0] # pyg graph object

```
* **Cora:** Cora is downloaded from Pytorch Geometric library (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid). We used following code:

```
dataset = Planetoid(root='/tmp/Cora', name='Cora')
```


* **Citeseer:** Citeseer is downloaded from Pytorch Geometric library (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid). We used following code:

```
dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
```

### Link Prediction
There are two datasets for link prediction: PGR and GDPR. 

* **Phenotype Gene Relation (PGR):**  PGR is created by Sousa et al., NAACL 2019 (https://aclanthology.org/N19-1152/) from PubMed articles and contains sentences describing relations between given genes and phenotypes. In our experiments, we only include data samples in PGR with available text descriptions for their genes and phenotypes. This amounts to ~71% of the original dataset. 

* **Gene, Disease, Phenotype Relation (GDPR):** This dataset is obtained by combining and linking entities across two freely-available datasets: Online Mendelian Inheritance in Man (OMIM, https://omim.org/) and Human Phenotype Ontology (HPO, https://hpo.jax.org/). The dataset contains relations between genes, diseases and phenotypes.

To download datasets with embeddings and Train/Test/Val splits, go to data directory and run download.sh as follows

```
sh ./download.sh
```

# To run the code 
Use the following command with appropriate arguments:
### Node Classification
```
cd node_classification
python3 node_classification.py

```
### Link Prediction
```
cd link_prediction
python3 link_prediction.py
```
# Citation

```
@inproceedings{nidhi-etal-2023-tgcl,
    title = "Complexity-Guided Curriculum Learning for Text Graphs",
    author = "Vakil, Nidhi and  Amiri, Hadi",
    booktitle = "Proceedings of the 2023 Empirical Methods in Natural Language Processing",
    publisher = "Association for Computational Linguistics",
    year = "2023"
    
}
```
