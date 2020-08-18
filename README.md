<a href="https://arxiv.org/abs/2008.02107"> Duality Diagram Similarity: a generic framework for initialization selection in task transfer learning</a><br/>
Kshitij Dwivedi, Jiahui Huang, Radoslaw Martin Cichy, Gemma Roig <br/>
ECCV 2020<br/><br/>
Here we provide the code to replicate our results on Taskonomy and Pascal VOC transfer benchmark.

* To assess similarity between two tasks, we extract the features of the Deep Neural Networks(DNNs) trained on these tasks


<div align="center">
  <img width="80%" alt="Feature Extraction" src="https://github.com/kshitijd20/dummy/blob/master/gifs/features.gif">
</div>


<br/><br/>
* We then create the Duality Diagram of a task from extracted feature matrix.


<div align="center">
  <img width="80%" alt="Duality Diagram from extracted Features" src="https://github.com/kshitijd20/dummy/blob/master/gifs/DD.gif">
</div>


<br/><br/>

* We finally compare the Duality Diagrams of two tasks to assess their similarity.

<div align="center">
  <img width="80%" alt="Duality Diagram Similarity" src="https://github.com/kshitijd20/dummy/blob/master/gifs/DDS.gif">
</div>


<br/><br/>



## Setup
* Code uses standard python libraries numpy, scipy, scikit-learn so it should run without installing additional libraries
* Download saved features of Taskonomy and Pascal VOC models from this <a href="https://www.dropbox.com/sh/iqg7p97vxmqhkcz/AABwcbMYSZKb2euEIqFMWaLma?dl=0">link </a> , and save the features in ./features directory.
* Download taskonomy groundtruth transfer learning results for <a href="https://github.com/StanfordVL/taskonomy/blob/master/results/affinities/all_affinities.pkl">affinities</a> and <a href="https://github.com/StanfordVL/taskonomy/blob/master/results/winrates/wins_vs_pixels_16k.pkl">winrate</a> and save them in ./affinities folder

## Taskonomy
* Run ```python computeDDS_taskonomy.py``` to compute DDS between Taskonomy models
* Compare the DDS with transfer learning performance by running the jupyter notebook : DDS_vs_transferlearning(Taskonomy).ipynb
* The results for affinities using Taskonomy images should be identical to Table below
<div align="center">
  <img width="80%" alt="Duality Diagram Similarity" src="https://github.com/kshitijd20/dummy/blob/master/gifs/Taskonomy_result.png">
</div>

## Pascal VOC
* Run ```python computeDDS_pascal.py``` to compute DDS between Taskonomy models and Pascal VOC model
* Compare the DDS with transfer learning performance by running the jupyter notebook : DDS_vs_transferlearning(Pascal).ipynb
* The results for affinities using Pascal images should be identical to Table below
<div align="center">
  <img width="80%" alt="Duality Diagram Similarity" src="https://github.com/kshitijd20/dummy/blob/master/gifs/pascal_result.png">
</div>

## Cite

If you use our code please consider citing the paper below

```
@inproceedings{dwivedi2020DDS,
  title={Duality Diagram Similarity: a generic framework for initialization
               selection in task transfer learning},
  author={Kshitij Dwivedi and
               Jiahui Huang and
               Radoslaw Martin Cichy and
               Gemma Roig},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2020}
}

```
