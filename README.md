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
* AutoKeras Tutorial: https://www.pyimagesearch.com/2019/01/07/auto-keras-and-automl-a-getting-started-guide/
* Another very exhaustive AutoKeras Tutorial: https://www.simonwenkel.com/2018/08/29/introduction-to-autokeras.html
* Setting up Kaggle in Google Colab: https://towardsdatascience.com/setting-up-kaggle-in-google-colab-ebb281b61463

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

* Validation accuracy: 99.0%
* Test set accuracy: 99.0%
* winning params: {'statsextractor': StatisticsExtractor(), 'standardizer': None, 'rf__n_estimators': 733, 'rf__min_samples_leaf': 1, 'rf__max_features': 'sqrt', 'rf__max_depth': 110, 'rf__bootstrap': False, 'nvdiadder': AddNVDI()}

## Deep learning

### Results 

All examples without preprocessing (except normalization) and all 4 base channels.

* Simple CNN with ~700k params, after 30 epochs (40mins training):

```
test set accuracy according to sklearn.accuracy_score: 0.9734320987654321
              precision    recall  f1-score   support

           0       0.94      0.93      0.93      3714
           1       0.97      0.97      0.97     18367
           2       0.99      0.97      0.98     14185
           3       0.92      0.95      0.94     12596
           4       0.86      0.89      0.87      2070
           5       1.00      1.00      1.00     30068

   micro avg       0.97      0.97      0.97     81000
   macro avg       0.95      0.95      0.95     81000
weighted avg       0.97      0.97      0.97     81000
```

* Best model after 2 hours of neural architecture search (NAS): 

```
test set accuracy according to sklearn.accuracy_score: 0.9922592592592593

              precision    recall  f1-score   support

           0       0.99      1.00      0.99      3714
           1       0.99      0.99      0.99     18367
           2       1.00      0.99      0.99     14185
           3       0.97      0.98      0.98     12596
           4       1.00      0.97      0.98      2070
           5       1.00      1.00      1.00     30068

   micro avg       0.99      0.99      0.99     81000
   macro avg       0.99      0.99      0.99     81000
weighted avg       0.99      0.99      0.99     81000
```
