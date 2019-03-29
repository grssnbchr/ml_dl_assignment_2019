# Land cover classification from airborne imagery using shallow learning

## Important links

* Original dataset: https://csc.lsu.edu/~saikat/deepsat/
* Kaggle: https://www.kaggle.com/crawford/deepsat-sat6#X_test_sat6.csv

### Resources

* Scikit-Image: scikit-image.org/

### Tutorials

* https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
* https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
* NVDI: https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index

### Results 

10% of the data, 0.8:0.2 train:validation

#### No preprocessing, 3136 features

* RF with 100 trees: 96% 

#### No preprocessing, but NVDI added, 3920 features

* RF with 100 trees: 97% (training took ca. 75sec)

#### Only grayscale features, 784

* RF with 100 trees: there's still a bug!  (training took ca. 45sec)

#### Standardization, without NVDI, 3136 features

* RF with 100 trees: 96%

#### Standardization, NVDI, stats extracted, 10 features

* RF with 100 trees: 99% (5 secs training)