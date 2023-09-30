JointEL
----------
Installing
----------

Requirements:


* Python 3.6 or greater
* numpy
* scipy
* scikit-learn >= 0.23.0
* numba


**Install**

JointEL uses Jvis-learn package, which we have developed previously. PyPI install, presuming you have numba and sklearn and all its requirements
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
