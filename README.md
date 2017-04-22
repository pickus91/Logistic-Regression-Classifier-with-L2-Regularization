# Logistic-Regression-Classifier-with-L2-Regularization
Logistic regression with L2 regularization for binary classification from scratch

## Dependencies
* [NumPy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)

## Code Example

```
from logisticRegressionClassifier import LogisticRegression
LR = LogisticRegression(learningRate = 0.01, numIterations = 20, penalty = 'L2', C = 0.01)  
```

## Licence
This code is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Synopsis

Logistic regression is a linear classification model that predicts binary outcomes based on a set of explanatory variables (i.e. features). In logistic regression, we are interested in determining the probability that an observation belongs to a given class. We can map a linear combination of weights and sample features and transform them to a probability value between 0 and 1 through the logistic function:
<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/equation1.PNG"  height="245" width="445">
</div>
<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/logisticFunction.png"  height="350" width="425">
</div>

If φ(z) falls above a probabilistic threshold (say, 50%) for a given sample, we categorize the sample as class 1, otherwise class 0. The weights _**w**_ of the logistic function can be learned by minimizing the log-likelihood function *J* (the logistic regression cost function) through gradient descent.

<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/equation2.PNG"  height="350" width="425">
</div>

If the logistic regression model suffers from high variance (over-fitting the training data), it may be a good idea to perform regularization to penalize large weight coefficients. In L2 regularization, we introduce the following bias term to the logistic regression cost function:

<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/equation3.PNG"  height="80" width="160">
</div>

Defining the regularization parameter C=1/λ, the new logistic regression cost function becomes: 
<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/equation4.PNG"  height="275" width="475">
</div>
As seen below, as we increase regularization strength, the weight coefficients of the logistic regression model shrink.
<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/weightsL2Regularization.png"  height="350" width="425">
</div>

## Example - Classification of Breast Cancer Wisconsin Dataset

To test the logistic regression classifier, we’ll be using data from the [Wisconsin Breast Cancer (Diagnostic) Data set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) from the UCI Machine Learning Repository. The data set consists of nine real-valued features computed from a digitized image of a final needle aspirate (FNA) of a breast mass with 699 observations. The features computed describe various characteristics of the cell nuclei present in biopsy images in both benign and malignant breast tumor.

<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/WisconsinBreastCancer_FeatureTable.PNG"  height="350" width="425">
</div>					

After imputing missing features values with their mean feature values, we will divide the dataset into separate training and testing sets (70% training, 30% testing), conduct a z-score normalization on the training data (a requirement of L2 regularization to work), and then perform principle component analysis (PCA) to reduce the feature subspace to avoid excessive dimensionality and improve the computational efficiency of the logistic regression model.
<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/pcaExplainedVariance.png"  height="350" width="425">
`</div>

For the sake of visualizations in this example, we will project the training data onto the feature subspace defined by the first two principle components (2-dimensions), which have a cumulative explained variance of 75%. As seen below, we see that the data is almost linearly separable by class. 

<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/pcaProjection.png"  height="350" width="425">
</div>

If the testing data follows this same pattern, a logistic regression classifier would be an advantageous model choice for classification. We now turn to training our logistic regression classifier with L2 regularization using 20 iterations of gradient descent, a tolerance threshold of 0.001, and a regularization parameter of 0.01.

```
LR = LogisticRegression(learningRate = 0.01, numIterations = 20, penalty = 'L2', C = 0.01)  
LR.train(X_train_pca, y_train, tol = 10 ** -3)
```
Looking at the log-liklihood function, we see that the tolerance of 0.001 is reached in 14 iterations.
```
LR.plotCost()
```

<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/logLiklihoodCost.png"  height="350" width="425">
</div>

Now that we have fit the logistic regression model to our training data, we can evaluate its performance on the testing data using a cutoff probability of 50% or above to classify the sample as Class 1, otherwise Class 0.
```
predictions, probs = LR.predict(X_test_pca, 0.5)
performance = LR.performanceEval(predictions, y_test)
```
<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/performanceMetrics.PNG"  height="350" width="425">
</div>

We can visually see how the model performs using the ```predictionPlot``` and ```plotDecisionRegions``` methods. The prediction plot shows how each test sample maps onto the logistic function, while the decision region plot shows how our logistic regression model divides the feature subspace by predicted class.
```
LR.plotDecisionRegions(X_test_pca, y_test)
LR.predictionPlot(X_test_pca, y_test)
```
<div>
<ul>        
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/logisticCurvePredictionPlot.png"  height="350" width="425">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/decisionBounds.png"  height="350" width="425">
 </ul>
</div>

Note, it is often best practice to use k-fold cross-validation to obtain a reliable estimate of this models generalization error. Increases in performance can also be obtained by tuning the regularization parameter, in which case holdout cross-validation should be used.
