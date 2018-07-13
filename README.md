# Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection


This implementation is based on the paper
**Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection**
[[Bo Zong et al (2018)]](https://openreview.net/pdf?id=BJJLHbb0-)

It's an **UNOFFICIAL** implementation.

# My Environment
- python 3
- Pandas
- Numpy
- sklearn
- Tensorflow (run on 1.7)

# Dataset
This implement only test on [KDDCup99](http://kdd.ics.uci.edu/databases/kddcup99/)

[KDD99Data.ipynb](KDD99Data.ipynb) shows preprocessing of KDD99

# Usage
```python
python kdd99_train_valid.py 
```

All hyperparameters are in [config.py](model/config.py)

# Result
Result (only one run) : Precision : 0.948, Recall : 0.958, F-score : 0.953

# References
The following implementations are referenced:

Pytorch : [danieltan07/dagmm](https://github.com/danieltan07/dagmm)

Tensorflow: [tnakae/DAGMM](https://github.com/tnakae/DAGMM)