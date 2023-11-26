# Complexity-Guided Curriculum Learning for Text Graphs

TGCL is a complexity guided curriculum learning approach for text graph which employs multiview complexity formalisms to space training samples over time for iterative training. By leveraging text and graph complexity formalisms, TGCL determines optimized timing to revist a training example and also determines an order of training samples. TGCL establishes curricula that are both data driven and model- or learner-dependent.

<p align="center">
<img src="https://github.com/CLU-UML/TGCL/blob/main/tgcl.png" width="900" height="450">
</p>


The architecture of the proposed model, TGCL. It takes subgraphs and text(s) of their target node(s)
as input. The radar chart shows graph complexity indices which quantify the difficulty of each subgraphs from
different perspectives (text complexity indices are not shown for simplicity). Subgraphs are ranked according to
each complexity index and these rankings are provided to TGCL scheduler to space samples over time for training.
