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

#### No preprocessing, 3136 features

10% of the data, 0.8:0.2 train:validation

* RF with 100 trees: 34%

#### No preprocessing, but NVDI added, 3920 features

10% of the data, 0.8:0.2 train:validation

* RF with 100 trees: 33%

#### Only grayscale features, 784

10% of the data, 0.8:0.2 train:validation

* RF with 100 trees: 38%
