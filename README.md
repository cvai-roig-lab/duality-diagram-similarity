<a href="https://arxiv.org/abs/2008.02107"> Duality Diagram Similarity: a generic framework for initialization selection in task transfer learning</a><br/>
Kshitij Dwivedi, Jiahui Huang, Radoslaw Martin Cichy, Gemma Roig <br/>
ECCV 2020
<div align="left">
  * To assess similarity between two tasks, we extract the features of the Deep Neural Networks(DNNs) trained on these tasks
</div>

<div align="center">
  <img width="80%" alt="Feature Extraction" src="https://github.com/kshitijd20/dummy/blob/master/gifs/features.gif">
</div>


<br/><br/>
<div align="left">
  * We then create the Duality Diagram of a task from extracted feature matrix.
</div>

<div align="center">
  <img width="80%" alt="Duality Diagram from extracted Features" src="https://github.com/kshitijd20/dummy/blob/master/gifs/DD.gif">
</div>


<br/><br/>
<div align="left">
  * We finally compare the Duality Diagrams of two tasks to assess their similarity.
</div>
<div align="center">
  <img width="80%" alt="Duality Diagram Similarity" src="https://github.com/kshitijd20/dummy/blob/master/gifs/DDS.gif">
</div>


<br/><br/>


Here we provide the code to replicate our results on Taskonomy and Pascal VOC transfer benchmark.

## Setup
* Code uses standard python libraries numpy, scipy, scikit-learn so it should run without installing additional libraries
* Download saved features of Taskonomy and Pascal VOC models from this <a href="https://www.dropbox.com/sh/iqg7p97vxmqhkcz/AABwcbMYSZKb2euEIqFMWaLma?dl=0">link </a> , and save the features in ./features directory.

## Taskonomy
* Run ```python computeDDS_taskonomy.py``` to compute DDS between Taskonomy models
* Compare the DDS with transfer learning performance by running the jupyter notebook : DDS_vs_transferlearning(Taskonomy).ipynb

## Pascal VOC
* Run ```python computeDDS_pascal.py``` to compute DDS between Taskonomy models and Pascal VOC model
* Compare the DDS with transfer learning performance by running the jupyter notebook : DDS_vs_transferlearning(Pascal).ipynb


## Cite

If you use our code please consider citing the paper below

```
@article{DBLP:journals/corr/abs-2008-02107,
  author    = {Kshitij Dwivedi and
               Jiahui Huang and
               Radoslaw Martin Cichy and
               Gemma Roig},
  title     = {Duality Diagram Similarity: a generic framework for initialization
               selection in task transfer learning},
  journal   = {CoRR},
  volume    = {abs/2008.02107},
  year      = {2020},
  url       = {https://arxiv.org/abs/2008.02107},
  archivePrefix = {arXiv},
  eprint    = {2008.02107},
  timestamp = {Fri, 07 Aug 2020 15:07:21 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2008-02107.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
