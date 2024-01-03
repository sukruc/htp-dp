In this study, we use the following dataset obtained from two separate experiments with different operating conditions. The first dataset is obtained from a boiling experiment with a single tube in a vertical orientation. The second dataset is obtained from a boiling experiment with a bundle of 19 tubes in a horizontal orientation. The details of the input and output variables are given in Tables 1-3.

# Imports

## Definitions

Raw inputs:

|Input|Unit|Symbol|Description|
|:---|:---|:---|:---|
|`Input 1`|$kg/m^2s$|$G$|Mass flux|
|`Input 2`|$Pa$|$P_{sat}$|Saturation pressure|
|`Input 3`|$W/m^2K$|$q$|Heat flux|
|`Input 4`|-|$x$|Quality|
|`Input 5`|$Pa$|$\Delta P_t$|Pressure drop|

*Table 1: Raw inputs for the single tube dataset*


Calculated inputs:

|Input|Unit|Symbol|Description|
|:---|:---|:---|:---|
|`Input 1`|-|$Re_l$|Reynolds number|
|`Input 2`|-|$X_{tt}$|Two-phase multiplier|
|`Input 3`|-|$Fr_l$|Froude number|
|`Input 4`|-|${We}_{L}$|Weber number|

*Table 2: Calculated inputs for the single tube dataset*

Outputs:

|Output|Unit|Symbol|Description|
|:---|:---|:---|:---|
|`Output 1`|$W/m^2K$|$h_{TP}$|Heat transfer coefficient|

*Table 3: Outputs for the single tube dataset*

\* *Pressure drop ($\Delta P_t$) from input table and heat transfer coefficient ($h_{TP}$) from the output table are interchanged and machine learning algorithms below are trained to predict the pressure drop and the heat transfer coefficient separately.*

# Data

|Mass flux|Saturation pressure|Heat flux|Quality|Pressure drop|Heat transfer coefficient|Reynolds number|Two-phase multiplier|Froude number|Weber number|Bond number|Tube type|Heat transfer coefficient|
|----|----|----|----|----|----|----|----|----|----|----|----|----|
|190.393921|589276.392344|10183.667411|0.286821|1908.497264|5624.26494|0.463739|0.287211|29.665103|0.317264|3744.893193|Plain tube h|3744.893193|
|190.393921|591130.402063|10306.572617|0.306536|2238.123367|5475.898804|0.426594|0.287399|29.722737|0.294672|3906.912259|Plain tube h|3906.912259|
|190.393921|591645.979654|10291.105025|0.360759|2184.774106|5049.426582|0.342545|0.287446|29.738299|0.242476|3658.382794|Plain tube h|3658.382794|
|190.393921|590962.351114|10336.957572|0.456461|2998.877317|4291.402048|0.239386|0.287352|29.715816|0.176314|4117.791176|Plain tube h|4117.791176|
|190.393921|591167.124112|10260.169839|0.583378|2961.531281|3289.825912|0.151131|0.287399|29.723428|0.117153|4183.575561|Plain tube h|4183.575561|

*Table 4: Sample dataset*

# Segmentation

We first apply an unsupervised PCA decomposition to the dataset in order to identify different patterns in the data. These segments are considered when sampling data for train/test and cross-validation operations.

![png](../img/pca.png)

*Figure 1: PCA decomposition of the dataset*

## Performing segmentation

Given below in Figure 1 is a 3d visualization of the dataset using the first three principal components. The data is colored according to the value of heat transfer coefficient.

![](../img/pca.png)

*Figure 1*

By observing the PCA plot, we can see that the data is clustered into three distinct groups, mainly separated across first two principal components.

|Input type|Input|PCA 1|PCA 2|PCA 3|
|:---|:---|:---|:---|:---|
|`Raw`|`Input 1`|0.371403|0.336128|-0.052853|
|`Raw`|`Input 2`|0.035643|-0.027975|0.788325|
|`Raw`|`Input 3`|-0.031835|0.044657|0.605993|
|`Raw`|`Input 4`|-0.338355|0.365280|0.057596|
|`Raw`|`Input 5`|0.072366|0.519421|0.003173|
|`Calculated`|`Input 1`|0.456948|-0.059758|-0.012223|
|`Calculated`|`Input 2`|0.348064|-0.364637|0.024985|
|`Calculated`|`Input 3`|0.379246|0.325789|-0.015902|
|`Calculated`|`Input 4`|0.380145|0.321854|0.053131|
|`Calculated`|`Input 5`|0.348953|-0.367088|0.036692|

*Table 4: Projection axes by inputs*

In the next step, we will use agglomerative clustering to segment the data into three groups with regard to the first two principal components. The results are given below in Figure 2.

# Principal component analysis

Apart from three distinct clusters, principal component analysis on the dataset also reveals a sub-seperation within each cluster by tube type. This is given below in Figure 2.

![](../img/pc-by-tube.png)

*Figure 2: Samples in principal component space. A sub-separation exists by tube type in each cluster.*

## Agglomerative clustering

We use agglomerative clustering to automatically segment the data into three groups. Results are given below in Figure 3 again in principal component space. On top of cluster configuration obtained by agglomerative clustering, we also define a sub-cluster category by tube type.

![](../img/pc2.png)

*Figure 3: Hierarchical clustering results visualized in principal component space*

# Split

Dataset is split into two parts by stratified sampling across the cluster labels. The first part is used for training/validation and tuning of the model hyperparameters. The second part is used for testing the final model.


## Scoring definitions

### $R^2$

$R^2$ is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. In this study, we use the following definition of $R^2$:

$$ R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} $$

where $y_i$ is the true value of the $i$th sample, $\hat{y}_i$ is the predicted value of the $i$th sample, and $\bar{y}$ is the mean value of the true values.

It is important to note that $R^2$ can be biased upwards for models with more parameters, even if they are meaningless. This is called the [overfitting] phenomenon. To avoid this, we use the adjusted $R^2$ score, which is defined as:

$$ R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1} $$

where $n$ is the number of samples and $p$ is the number of model parameters.

### MAE

Mean absolute error (MAE) is a measure of difference between two continuous variables. For two vectors $y$ and $\hat{y}$, MAE is defined as:

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

### RMSE

Root mean squared error (RMSE) is a quadratic scoring rule that also measures the average magnitude of the error. It is defined as:

$$ RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} $$

$RMSE$ can be driven up by outliers and does not indicate under- or over-estimation. To address this issue, we use the mean absolute percentage error (MAPE) and weighted absolute percentage error (WAPE) scores whose details are given below that penalize errors relative to the true value.

### MAPE

Mean absolute percentage error (MAPE) is a measure of prediction accuracy of a forecasting method in statistics. For two vectors $y$ and $\hat{y}$, MAPE is defined as:

$$ MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| $$

### WAPE

Weighted absolute percentage error (WAPE) is a measure of prediction accuracy of a forecasting method in statistics. For two vectors $y$ and $\hat{y}$, WAPE is defined as:

$$ WAPE = \frac{100}{\sum_{i=1}^{n} y_i} \sum_{i=1}^{n} \left| y_i - \hat{y}_i \right| $$


### Weighted Bias %

Weighted percentage bias is a measure of systematic error in regression models. For two vectors $y$ and $\hat{y}$, weighted percentage bias is defined as:

$$ Weighted Bias \% = \frac{100}{\sum_{i=1}^{n} y_i} \sum_{i=1}^{n} \left( -y_i + \hat{y}_i \right) $$


Checking bias of the model is essential especially when modeling physical phenomena and built models are possibly used for extrapolation. In such cases, a model with a non-zero bias is not reliable.

### Hocanin score

Hocanin score is a measure of prediction accuracy of a forecasting method in statistics. For two vectors $y$ and $\hat{y}$, Hocanin score is defined as:

$$ Hocanin = \frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i - \hat{y}_i}{y_i} \right)^2 $$


# Models

Different models are trained for each cluster. The models are trained using the training/validation dataset and tuned using the validation dataset. The final model is evaluated using the test dataset. The models are trained using the following algorithms:

* [ANN]
* [AdaBoost regressor of decision trees]
* [Locally weighted linear regression]

## MLP definition

ANN configuration we use in this study is as follows:

* 2 hidden layers with 40 and 10 neurons
* `relu` activation function



### ANN search

We use grid search to find the best hyperparameters for the ANN model. The hyperparameters we search are:

* `hidden_layer_sizes`: The number of neurons in the hidden layers
* `activation`: The activation function for the hidden layers
* `tolerance`: The tolerance for the optimization algorithm

Hyperparameter search is executed through a 3-fold cross validation scheme and train/test folds are stratified by 6 cluster labels described in Section x.

Training data is split into 10-folds by following the exact same stratification approach described above, and hyperparameter selection process is repeated on each fold. The best hyperparameters are selected based on the average performance across the folds. Results for each fold are given below in Table x.

The cross-validation results:

|Score type|Mean|Standard deviation|Minimum|25%|50%|75%|Maximum|
|---|---|---|---|---|---|---|---|
|$R^2$|0.8385|0.0983|0.6107|0.8205|0.8490|0.8890|0.9516|
|MAE|-324.4948|98.8830|-475.7393|-364.9229|-324.8835|-284.3145|-169.3304|
|MAPE|6.46%|1.79%|9.93%|6.88%|6.52%|5.88%|3.65%|
|WAPE|6.51%|2.08%|10.38%|7.59%|6.27%|5.74%|3.51%|
|Bias|0.20%|2.82%|-2.67%|-1.94%|-0.92%|1.53%|5.01%|
|Hocanin score|0.8520|0.0960|0.6980|0.7873|0.8890|0.9193|0.9831|

*Table 5. 10-fold cross-validation results for the ANN model.*





# 3.1. Machine Learning Methods

>A small section about how machine learning models can be useful in modeling physical processes.

Machine learning models can be used to model physical processes. These models can be used to predict the output of a physical process given the input. They can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process. Machine learning models can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process. Machine learning models can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process. Machine learning models can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process. Machine learning models can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process. Machine learning models can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process. Machine learning models can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process. Machine learning models can also be used to identify the inputs that have the greatest effect on the output. This can be useful for optimizing the design of a physical process.

## 3.1.2. Artificial Neural Networks (ANN)

>Give a brief description of ANN and its applications in the field of mechanical engineering for modeling.

Artificial Neural Networks (ANN) are a class of machine learning algorithms that are loosely inspired by the brain. They are typically organized in layers, with the first layer being the input layer, the last layer being the output layer, and any layers in between being hidden layers. Each layer is made up of one or more neurons. Each neuron takes a weighted sum of its inputs, applies an activation function, and passes the result to the next layer. The weights and biases of each neuron are adjusted during training. The network is trained by passing training data through the network, calculating the error at the output layer, and then propagating the error back through the network to adjust the weights and biases. This process is repeated until the error reaches a minimum. Once trained, the network can be used to make predictions on new data.

ANNs are useful for modeling complex, non-linear relationships between inputs and outputs. They are also useful for modeling relationships where the inputs are not fully understood or cannot be easily modeled. ANNs can be used for regression or classification problems. They are often used for image recognition, speech recognition, and natural language processing. They are also used for time-series forecasting and financial modeling.

>Describe the structure of ANN and its main components.

The basic structure of an ANN is shown below.

![ANN structure](https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/400px-Colored_neural_network.svg.png)

The input layer is shown in blue. The hidden layers are shown in green. The output layer is shown in red. The connections between the neurons are shown in black. The weights and biases are shown in gray.

>Give a brief overview of the mathematical formulation of ANN.

The output of a neuron is given by:

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

where $x_i$ is the $i^{th}$ input, $w_i$ is the $i^{th}$ weight, $b$ is the bias, and $f$ is the activation function. The activation function is typically a non-linear function such as the sigmoid function or the hyperbolic tangent function. The weights and biases are adjusted during training to minimize the error at the output layer.

>Now to how we used ANN in this study. We used an architecture of (128, 64, 32, 16, 8) with activation function selection of ReLU. Also give ReLU formulation.

In this study, we used a feed-forward neural network with 5 hidden layers. The number of neurons in each layer was 128, 64, 32, 16, and 8. The activation function was the rectified linear unit (ReLU). The ReLU function is given by:

$$
f(x) = \max(0, x)
$$

>Describe the training process of ANN. Briefly talk about different options, but we used lbfgs for optimization. Also talk about why we opted for lbfgs.

The network was trained using the `MLPRegressor` class from the `sklearn.neural_network` module. The `MLPRegressor` class uses the backpropagation algorithm to train the network. The backpropagation algorithm is a gradient descent algorithm that uses the chain rule to calculate the gradient of the error with respect to the weights and biases. The gradient is then used to update the weights and biases. The `MLPRegressor` class uses the `lbfgs` solver for optimization. The `lbfgs` solver is a quasi-Newton method that approximates the Hessian matrix using the previous gradient vectors. The `lbfgs` solver is a fast solver that works well for relatively small datasets.

>Describe how cross validation was used in training. Summary: we used 5-fold group cross validation




## 3.1.2. Locally Weighted Linear Regression (LWR)

LWR is a non-parametric method that fits a linear model to a subset of data points that are close to a given query point. The idea is to give more weight to the data points that are near the query point, and less weight to the data points that are far away. This way, the model can capture the local structure of the data and adapt to non-linear patterns.

In this study, we use restrict locality to the nearest k data points of query sample and define error function as follows:

$$
E(x_q) = \frac{1}{2}\sum_{x \in k}{(f(x)-\hat{f}(x))^2 K(d(x_q, x)) }
$$

where $x_q$ is the query sample, $f(x)$ is the true function, $\hat{f}(x)$ is the predicted function, $d(x_q, x)$ is the distance between the query sample and the data point, and $K(d(x_q, x))$ is the kernel function. The kernel function is used to give more weight to the data points that are close to the query point. Although several choices are available for the kernel function with some esoteric options, we use the Gaussian kernel function for this study. The Gaussian kernel function is given by:

$$
K(d(x_q, x)) = exp(-\gamma d(x_q, x)^2)
$$

where $\gamma$ is a hyperparameter that controls the width of the kernel. The hyperparameter $\gamma$ is tuned using cross-validation.

Distance between two points is calculated using the Euclidean distance:

$$
d(x_q, x) = \sqrt{\sum_{i=1}^{n}{(x_{qi}-x_i)^2}}
$$

where $x_{qi}$ is the $i^{th}$ component of the query sample, and $x_i$ is the $i^{th}$ component of the data point.



Approximation of the function is given by:

$$
\hat{f}(x_q) = \theta^T x_q
$$

where $\theta$ is the weight vector. The weight vector is calculated using the following equation:

$$
\theta = (X^T W X)^{-1} X^T W y
$$

where $X$ is the design matrix, $W$ is the diagonal matrix of weights, and $y$ is the vector of outputs.

## 3.1.3. Other Model Considerations
This study is constrained to parametric methods due to limited extrapolation capabilities of non-parametric methods. Therefore, the following methods, although being proven to be effective in other studies, are not considered in this study:
- Support Vector Machines (SVM)
- Decision Trees
- Random Forests
- K-Nearest Neighbors (KNN)
- Gaussian Processes (GP)


# 3.2. Validation Setting

>Describe the validation setting. We performed a pca analysis, identified clusters visually, and then used a gaussia mixture model to identify clusters. We then used the clusters to perform a 5-fold group cross validation by performing a stratified split on the clusters.

The validation setting is described in detail in the following sections.

## 3.2.1. Principal Component Analysis (PCA)

>Describe the PCA analysis. We used PCA to reduce the dimensionality of the data. We used the first 3 principal components to visualize the data. W

Principal component analysis (PCA) is a dimensionality reduction technique that is used to reduce the dimensionality of a dataset while preserving as much of the variance as possible. It is often used to visualize high-dimensional datasets. It is also used to reduce the dimensionality of a dataset before applying a machine learning algorithm. PCA is a linear transformation that transforms the data into a new coordinate system. The new coordinate system is chosen such that the first coordinate has the largest possible variance, the second coordinate has the second largest possible variance, and so on. The first coordinate is called the first principal component, the second coordinate is called the second principal component, and so on. The first principal component is the direction in which the data has the largest variance. The second principal component is the direction in which the data has the second largest variance, and so on. The principal components are orthogonal to each other. The principal components are also uncorrelated with each other. The principal components and the original variables are related by the following equation:

$$
y = W^T x
$$

where $y$ is the vector of principal components, $W$ is the matrix of principal components, and $x$ is the vector of original variables. The matrix $W$ is called the loading matrix. The loading matrix is calculated using the following equation:

$$
W = (X^T X)^{-1} X^T y
$$

where $X$ is the design matrix, $y$ is the vector of principal components, and $W$ is the loading matrix. The loading matrix is calculated using the following equation:

$$
W = (X^T X)^{-1} X^T y
$$

where $X$ is the design matrix, $y$ is the vector of principal components, and $W$ is the loading matrix. The loading matrix is calculated using the following equation:

$$
W = (X^T X)^{-1} X^T y




## 3.4 Sampling for extrapolation analysis

In this section, we demonstrate the selected models' strength for making accurate estimations outside observed ranges during the experiment.
Extrapolation performance analysis is conducted by following approach: clusters are identified in principal component space through fitting a Gaussian Mixture Model (GMM), and a Ledoit-Wolf (LW) covariance estimator is fitted separately to each identified cluster again in principal component space.

>Ledoit-Wolf (LW) covariance estimator is fitted separately to each identified cluster again in principal component space.

The LW estimator is a shrinkage estimator that shrinks the sample covariance matrix towards a structured estimator. The structured estimator is a matrix with a constant diagonal and constant off-diagonal elements. The LW estimator is given by:

$$
\hat{\Sigma} = \alpha S + (1 - \alpha) \hat{\Sigma}_{\text{diag}}
$$

where $\hat{\Sigma}$ is the LW estimator, $S$ is the sample covariance matrix, $\hat{\Sigma}_{\text{diag}}$ is the structured estimator, and $\alpha$ is the shrinkage parameter. The shrinkage parameter is given by:

$$
\alpha = \frac{1}{n} \sum_{i=1}^{n} \frac{(\hat{\sigma}_{ii} - \hat{\sigma}_{\text{diag}})^2}{(\hat{\sigma}_{ii} - \hat{\sigma})^2}
$$

where $\hat{\sigma}_{ii}$ is the $i^{th}$ diagonal element of the sample covariance matrix, $\hat{\sigma}_{\text{diag}}$ is the diagonal element of the structured estimator, and $\hat{\sigma}$ is the average of the diagonal elements of the sample covariance matrix.


Samples to be used for extrapolation are then determined by Mahalanobis distance estimations coming from LW estimators fitted on each cluster. 10 samples with the highest Mahalanobis distance are then held out for test and the rest of the samples are used for training the models.

Mahalanobis distance is a measure of the distance between a point and a distribution. It is a multi-dimensional generalization of the one-dimensional Euclidean distance. It is defined as the square root of the sum of the squared differences between the point and the mean of the distribution, divided by the covariance matrix of the distribution. It is given by:

$$
d(x, \mu, \Sigma) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}
$$

where $x$ is the point, $\mu$ is the mean of the distribution, and $\Sigma$ is the covariance matrix of the distribution.

