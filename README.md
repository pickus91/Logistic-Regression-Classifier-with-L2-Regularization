
# Logistic-Regression-Classifier-with-L2-Regularization
Logistic regression with L2 regularization for binary classification

## Synopsis

<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/weightsL2Regularization.png"  height="350" width="425">
</div>

## Dependencies
* [NumPy](http://www.numpy.org/)
* [Matplotlib](http://matplotlib.org/)

## Code Example
```
from logisticRegressionClassifier import LogisticRegression
LR = LogisticRegression(learningRate = 0.01, numIterations = 20, penalty = 'L2', C = 0.01)  

```
## Example - Classification of Breast Cancer Wisconsin Dataset

<div align = "center">
<img style="float: left;" src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/pcaExplainedVariance.png"  height="350" width="425">
</div>

<div>
        <ul>
            <img src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/logisticCurvePredictionPlot.png" alt="" class="first" height = "350" width = "425">
            
             <img src="https://github.com/pickus91/Logistic-Regression-Classifier-with-L2-Regularization/blob/master/figures/decisionBounds.png" alt="" class="first" height = "350" width = "425">                                
        </ul>
</div>


## Licence
This code is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
