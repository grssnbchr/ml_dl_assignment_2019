# Land cover classification from airborne imagery

## Important links

* Original dataset: https://csc.lsu.edu/~saikat/deepsat/
* Kaggle: https://www.kaggle.com/crawford/deepsat-sat6#X_test_sat6.csv

### Resources

* Scikit-Image: scikit-image.org/
* Implementing scoring function: https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
* Visualizing hyperparameter performance: https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#demonstration-of-multi-metric-evaluation-on-cross-val-score-and-gridsearchcv
* And: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
* Randomized search through hyper parameters: https://www.pyimagesearch.com/2016/08/15/how-to-tune-hyperparameters-with-python-and-scikit-learn/

### Tutorials

* https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
* https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
* NVDI: https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index
* Using Sklearn with Keras: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
* Grid Search for Deep Learning: https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

## Shallow learning

### Results 

10% of the data, 0.8:0.2 train:validation

#### No preprocessing, 3136 features

* RF with 100 trees: 96% 

#### No preprocessing, but NVDI added, 3920 features

* RF with 100 trees: 97% (training took ca. 75sec)

#### Only grayscale features, 784

* RF with 100 trees: 83.9%

#### Standardization, without NVDI, 3136 features

* RF with 100 trees: 96%

#### Standardization, stats extracted, 8 features

* RF with 100 trees: 98.8%

```
Percentage correct:  98.79629629629629
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       311
           1       0.99      0.98      0.99      1430
           2       0.99      1.00      0.99      1111
           3       0.98      0.97      0.98      1034
           4       0.94      0.91      0.92       149
           5       1.00      1.00      1.00      2445

   micro avg       0.99      0.99      0.99      6480
   macro avg       0.97      0.97      0.97      6480
weighted avg       0.99      0.99      0.99      6480
```

#### Standardization, NVDI, stats extracted, 10 features

* RF with 100 trees: 99.0% (5 secs training)
* The same with only 1% of training data: 97.5%

```
Percentage correct:  98.99691358024691
              precision    recall  f1-score   support

           0       0.97      0.97      0.97       311
           1       0.99      0.99      0.99      1430
           2       0.99      1.00      0.99      1111
           3       0.98      0.98      0.98      1034
           4       0.94      0.95      0.94       149
           5       1.00      1.00      1.00      2445

   micro avg       0.99      0.99      0.99      6480
   macro avg       0.98      0.98      0.98      6480
weighted avg       0.99      0.99      0.99      6480
```

* Adding NVDI made the f1-score for the 4th class a bit better by 2 percentage points

#### Standardization, NDVI, stats extracted, optimized model after randomized grid search

* Validation accuracy: 
* Test set accuracy: