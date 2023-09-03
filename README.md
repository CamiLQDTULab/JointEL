====
JointEL
====

Emerging single-cell technologies profile different modalities of data in the same cell,  providing opportunities to study cellular population
and cell development at a resolution that was previously inaccessible. The first and most fundamental step in analyzing single-cell multimodal data
is the identification of the cell types in the data using clustering analysis and classification. However, combining different data modalities for
the classification task in multimodal data remains a computational challenge. We propose an approach for identifying cell types in multimodal omics 
data via joint dimensionality reduction. We first introduce a general framework that extends loss based dimensionality reduction methods such as 
nonnegative matrix factorization and UMAP to multimodal omics data. Our approach can learn the relative contribution of each modality to a concise 
representation of cellular identity that enhances discriminative features and decreases the effect of noisy features. The precise representation of
the multimodal data in a low dimensional space improves the predictivity of classification methods. 
In our experiments using both synthetic and real data, we show that our framework produces unified embeddings that agree with known cell types and allows
the predictive algorithms to annotate the cell types more  accurately than state-of-the-art classification methods. 

----------
Installing
----------

Requirements:


* Python 3.6 or greater
* numpy
* scipy
* scikit-learn >= 0.23.0
* numba


**Install Options**

PyPI install, presuming you have numba and sklearn and all its requirements
(numpy and scipy) installed:


    pip install Jvis-learn

If you have a problem with pip installation then we'd suggest installing
the dependencies manually using anaconda followed by pulling umap from pip:

    conda install numpy 
    conda install scipy==1.5.3
    conda install scikit-learn==0.24.1
    conda install numba==1.19.2
    pip install Jvis-learn

----------
Reproducibility
----------

For more realistic examples and Python scripts to reproduce the results
in our paper for each section are also included in this directory (sim_splat, A. , B.,  C.).

The datasets used in the paper can be found here: https://drive.google.com/file/d/1rP1wTgKlxDi0WMhjkuImffodHBnlIWaU/view?usp=sharing

-------
License
-------

The JointEL package is 3-clause BSD licensed.

This code was tested on 
Python 3.6, 3.7; scikit-learn version 0.24.1; numpy version 1.19.2; scipy version 1.5.3; numba version 0.52.0 
