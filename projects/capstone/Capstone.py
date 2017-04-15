
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Model Evaluation & Validation
# ## Project: Predicting Boston Housing Prices
# 
# Welcome to the first project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a *good fit* could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Housing). The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. For the purposes of this project, the following preprocessing steps have been made to the dataset:
# - 16 data points have an `'MEDV'` value of 50.0. These data points likely contain **missing or censored values** and have been removed.
# - 1 data point has an `'RM'` value of 8.78. This data point can be considered an **outlier** and has been removed.
# - The features `'RM'`, `'LSTAT'`, `'PTRATIO'`, and `'MEDV'` are essential. The remaining **non-relevant features** have been excluded.
# - The feature `'MEDV'` has been **multiplicatively scaled** to account for 35 years of market inflation.
# 
# Run the code cell below to load the Boston housing dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[4]:

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)


# ## Data Exploration
# In this first section of this project, you will make a cursory investigation about the Boston housing data and provide your observations. Familiarizing yourself with the data through an explorative process is a fundamental practice to help you better understand and justify your results.
# 
# Since the main goal of this project is to construct a working model which has the capability of predicting the value of houses, we will need to separate the dataset into **features** and the **target variable**. The **features**, `'RM'`, `'LSTAT'`, and `'PTRATIO'`, give us quantitative information about each data point. The **target variable**, `'MEDV'`, will be the variable we seek to predict. These are stored in `features` and `prices`, respectively.

# ### Implementation: Calculate Statistics
# For your very first coding implementation, you will calculate descriptive statistics about the Boston housing prices. Since `numpy` has already been imported for you, use this library to perform the necessary calculations. These statistics will be extremely important later on to analyze various prediction results from the constructed model.
# 
# In the code cell below, you will need to implement the following:
# - Calculate the minimum, maximum, mean, median, and standard deviation of `'MEDV'`, which is stored in `prices`.
#   - Store each calculation in their respective variable.

# In[5]:

# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)


# ### Question 1 - Feature Observation
# As a reminder, we are using three features from the Boston housing dataset: `'RM'`, `'LSTAT'`, and `'PTRATIO'`. For each data point (neighborhood):
# - `'RM'` is the average number of rooms among homes in the neighborhood.
# - `'LSTAT'` is the percentage of homeowners in the neighborhood considered "lower class" (working poor).
# - `'PTRATIO'` is the ratio of students to teachers in primary and secondary schools in the neighborhood.
# 
# _Using your intuition, for each of the three features above, do you think that an increase in the value of that feature would lead to an **increase** in the value of `'MEDV'` or a **decrease** in the value of `'MEDV'`? Justify your answer for each._  
# **Hint:** Would you expect a home that has an `'RM'` value of 6 be worth more or less than a home that has an `'RM'` value of 7?

# **Answer: **
# 
# RM is the Average Number of rooms among homes in the neighborhood. This is an indirect indicator of prosperous neighborhood. Neighborhoods where there are more average number of rooms they should positively affect the MEDV or the prices as this is likely to be in a higher strata of neighborhood in the society. Slums have very low average number of rooms versus rich neighborhoods which have higher average number of rooms. although we will need to hypothesise this and prove it but based on intution it should have a positive affect on prices and lead to an **increase** in prices of the market.
# 
# LSTAT is the representation of Lower class or Working poor. As the percentage of lower class people goes up in a neighborhood we are likely to see a decrease in prices. It really matters who your neighboor is if they are rich and famous you would want to pay a premium to buy that house near them versus avoid one where low calss people are living. This is how the buyers of real estate think when making a purchase. Premium is given to rich neighborhoods so the more the lower class in the locality the more likely it is to command a lower price in the market and **decrease** the real estate prices. 
# 
# PTRATIO concerns the pupils or studets to the number of teachers in the neighborhood schools. A high PTRATIO means there are more pupils to the number of students for the schools in the neighborhood. Based on intution and observing the trend in the society I can say that the schools which have a low PTRATIO command a premium fees and therefore attract high society individuals to enrol their kids. This is also an indirect indicator of the fact that the neighborhood is rich or poor. A high ratio should bring the prices down and **decrease** for the real estate property as parents try to settle down nearest to the schools their children are enrolled in to avoid commute times for small children and reduce stress.
# 

# ----
# 
# ## Developing a Model
# In this second section of the project, you will develop the tools and techniques necessary for a model to make a prediction. Being able to make accurate evaluations of each model's performance through the use of these tools and techniques helps to greatly reinforce the confidence in your predictions.

# ### Implementation: Define a Performance Metric
# It is difficult to measure the quality of a given model without quantifying its performance over training and testing. This is typically done using some type of performance metric, whether it is through calculating some type of error, the goodness of fit, or some other useful measurement. For this project, you will be calculating the [*coefficient of determination*](http://stattrek.com/statistics/dictionary.aspx?definition=coefficient_of_determination), R<sup>2</sup>, to quantify your model's performance. The coefficient of determination for a model is a useful statistic in regression analysis, as it often describes how "good" that model is at making predictions. 
# 
# The values for R<sup>2</sup> range from 0 to 1, which captures the percentage of squared correlation between the predicted and actual values of the **target variable**. A model with an R<sup>2</sup> of 0 is no better than a model that always predicts the *mean* of the target variable, whereas a model with an R<sup>2</sup> of 1 perfectly predicts the target variable. Any value between 0 and 1 indicates what percentage of the target variable, using this model, can be explained by the **features**. _A model can be given a negative R<sup>2</sup> as well, which indicates that the model is **arbitrarily worse** than one that always predicts the mean of the target variable._
# 
# For the `performance_metric` function in the code cell below, you will need to implement the following:
# - Use `r2_score` from `sklearn.metrics` to perform a performance calculation between `y_true` and `y_predict`.
# - Assign the performance score to the `score` variable.

# In[6]:

# TODO: Import 'r2_score'
from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
   
    score = r2_score(y_true, y_predict)
    
    # Return the score
    return score


# ### Question 2 - Goodness of Fit
# Assume that a dataset contains five data points and a model made the following predictions for the target variable:
# 
# | True Value | Prediction |
# | :-------------: | :--------: |
# | 3.0 | 2.5 |
# | -0.5 | 0.0 |
# | 2.0 | 2.1 |
# | 7.0 | 7.8 |
# | 4.2 | 5.3 |
# *Would you consider this model to have successfully captured the variation of the target variable? Why or why not?* 
# 
# Run the code cell below to use the `performance_metric` function and calculate this model's coefficient of determination.

# In[7]:

# Calculate the performance of this model
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)


# **Answer:**  
# 
# The Model has a Coefficient of Determination is R^2, of 0.923
#  This represents the proportion of the variance in the dependent variable that is predicted from the independent variable. A value of R2 closer to 1 since 0.923 is closer to 1 means that the dependent variable can be predicted with very little error from the independent variable. 
#  This also means **92.3** percent of the variance in Dependant variable is predictable from Independant variable. So this is a good percentage of capturing the variation.

# ### Implementation: Shuffle and Split Data
# Your next implementation requires that you take the Boston housing dataset and split the data into training and testing subsets. Typically, the data is also shuffled into a random order when creating the training and testing subsets to remove any bias in the ordering of the dataset.
# 
# For the code cell below, you will need to implement the following:
# - Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the `features` and `prices` data into training and testing sets.
#   - Split the data into 80% training and 20% testing.
#   - Set the `random_state` for `train_test_split` to a value of your choice. This ensures results are consistent.
# - Assign the train and testing splits to `X_train`, `X_test`, `y_train`, and `y_test`.

# In[8]:

# TODO: Import 'train_test_split'
from sklearn import cross_validation
from sklearn.utils import shuffle

# TODO: Shuffle and split the data into training and testing subsets
features=shuffle(features,  random_state=0)
prices=shuffle(prices,  random_state=0)
#Shuffle
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, prices, test_size=0.2, random_state=0)

# Success
print "Training and testing split was successful."


# ### Question 3 - Training and Testing
# *What is the benefit to splitting a dataset into some ratio of training and testing subsets for a learning algorithm?*  
# **Hint:** What could go wrong with not having a way to test your model?

# **Answer: **
# 
# Splitting of dataset into some ratio of training and testing subsets serves as a check against overfitting. Making predictions by learning from a subset of data namely the training data and then to make reliable predictions on the untrained data shows if the model will actually work when deployed. This does not mean that any model that shows promise in testing will perform great on deployment. But testing the model on test data gives us some degree of confidence that the model if it performs well during testing it is likely to succeed in deployment.
# 
# If we did not do this training and testing split then we will not get an estimate of how well the model is going to perform. We will simply be in the dark and waste a lot of time deploying models which fail in production. The chances of failure and cost to deployment of model will go up significantly as a result of this.
# 
# In conclusion in Supervized learning we should split our data between train and test and estimate the performance to look if the model is underfitting and overfitting the test data or it does a good job of prediction.

# ----
# 
# ## Analyzing Model Performance
# In this third section of the project, you'll take a look at several models' learning and testing performances on various subsets of training data. Additionally, you'll investigate one particular algorithm with an increasing `'max_depth'` parameter on the full training set to observe how model complexity affects performance. Graphing your model's performance based on varying criteria can be beneficial in the analysis process, such as visualizing behavior that may not have been apparent from the results alone.

# ### Learning Curves
# The following code cell produces four graphs for a decision tree model with different maximum depths. Each graph visualizes the learning curves of the model for both training and testing as the size of the training set is increased. Note that the shaded region of a learning curve denotes the uncertainty of that curve (measured as the standard deviation). The model is scored on both the training and testing sets using R<sup>2</sup>, the coefficient of determination.  
# 
# Run the code cell below and use these graphs to answer the following question.

# In[9]:

# Produce learning curves for varying training set sizes and maximum depths
vs.ModelLearning(features, prices)


# ### Question 4 - Learning the Data
# *Choose one of the graphs above and state the maximum depth for the model. What happens to the score of the training curve as more training points are added? What about the testing curve? Would having more training points benefit the model?*  
# **Hint:** Are the learning curves converging to particular scores?

# **Answer: **
# 
# Selecting Graph number 2 max_depth is 3 for the graph.
# In this graph we can see it has the visually closest lines between train score and test score points in all the four graphs.
# The testing score starts at 25 and then quickly goes on to learn at points 50. the training score starts higher at 1.0 and then comes down to 0.9 and then near and very close to 0.8 at points 100. It almost remains steady even if more points are added and never goes down below 0.8 level. However the training score of 350 is where the testing score converges or comes close to the training score. Immediately after the point 350 and 400 we see a sudden nudge towards south and the training score seems to be going down. This indicates adding any more data points beyond 350 is futile beyond this point.

# ### Complexity Curves
# The following code cell produces a graph for a decision tree model that has been trained and validated on the training data using different maximum depths. The graph produces two complexity curves — one for training and one for validation. Similar to the **learning curves**, the shaded regions of both the complexity curves denote the uncertainty in those curves, and the model is scored on both the training and validation sets using the `performance_metric` function.  
# 
# Run the code cell below and use this graph to answer the following two questions.

# In[10]:

vs.ModelComplexity(X_train, y_train)


# ### Question 5 - Bias-Variance Tradeoff
# *When the model is trained with a maximum depth of 1, does the model suffer from high bias or from high variance? How about when the model is trained with a maximum depth of 10? What visual cues in the graph justify your conclusions?*  
# **Hint:** How do you know when a model is suffering from high bias or high variance?

# **Answer: **
# 
# In max_depth= 1, this model has high bias. The training and validation scores are low compared to other areas of the graph. This means that the model is not able to explain the variance in the data.
# 
# In the max_depth of 10 we can clearly see that the model is indicating high variance. The graph has a training score that is much higher than the validation score in comparision to max_depth=1. 
# This is seen throughout the chart in max_depth=10 as the distance between training and test scores never comes closer. 
# 
# In a nutshell the rules to identify the High bias and High variance charts are the following:
# High variance models have a gap between the training and validation scores- We see this in graph max_depth=10
# High bias models have have a small or no gap between the training and validations scores- We see this in graph max_depth=1

# ### Question 6 - Best-Guess Optimal Model
# *Which maximum depth do you think results in a model that best generalizes to unseen data? What intuition lead you to this answer?*

# **Answer: **
# 
# From the max_depths graphs we see that at 1 it is having a high bias and max_depth 10 is having a high variance. At max_depth 3 we see some convergence. However there is no graph for max_depth=4 or 5 which we can check to see if it generalizes to the unseen data. Max_depth=4 or Max_depth=5 is the answer.

# -----
# 
# ## Evaluating Model Performance
# In this final section of the project, you will construct a model and make a prediction on the client's feature set using an optimized model from `fit_model`.

# ### Question 7 - Grid Search
# *What is the grid search technique and how it can be applied to optimize a learning algorithm?*

# **Answer: **
# 
# Grid Search algorithm is an exhaustive search algorithm as compared to other algorithms. In comparision to other algorithm in scikit-learn RandomSearchCV it is computationally expensive[1].
# Grid search means you have a set of models and we train each of the models and evaluate it using cross-validation. We then select the one that performed best through the performance metric through cross validating on the training set.
# Grid-search is a way to select the best of a family of models, parametrized by a grid of parameters. Basically the algo uses cross validation to vote for the best grid of the parameter and select that performed the best.
# 
# One drawback with Grid Search algorithm which is overcome by Random Search algorithm is when there are a lot of Unimportant parameters in the data then it is not Robust. Random search is more robust in such a case. Also when the unimportant parameters are high then the runtime for grid search increases substantially[2].
# 
# Refenreces:
# [1]How to find the best model parameters in scikit-learn, https://www.youtube.com/watch?v=Gol_qOgRqfA
# [2]Scikit Learn Workshop with Andreas Mueller 3-30-20, https://www.youtube.com/watch?v=0wUF_Ov8b0A&feature=youtu.be&t=17m38s

# ### Question 8 - Cross-Validation
# *What is the k-fold cross-validation training technique? What benefit does this technique provide for grid search when optimizing a model?*  
# **Hint:** Much like the reasoning behind having a testing set, what could go wrong with using grid search without a cross-validated set?

# **Answer: **
# In this k-fold cross-validation training technique, we use original sample  and we randomly partition into k equal sized subsamples or subsets. Of the k subsamples or subsets, a single subsample is taken to be the  validation data for testing the model, and the remaining k − 1 subsamples or subsets are used as individual training data. The cross-validation process is then repeated k fold times with each of the k subsets used exactly once as the validation data. The k results from the folds are then be averaged to produce a single estimation.
# 
# When we use K-fold cross-validation we have more reliable estimate of out-of-sample performance than single train/test split.
# We are also able to reduce the variance of a single trial of a train/test split.
# The average testing accuracy is used in k-fold cross-validation as a benchmark to decide which is the most optimal set of parameters for the learning algorithm.
# 
# Without using cross-validation when we set and we run grid-search our estimate of out-of-sample performance would lead to a high variance.
# In summary, without k-fold cross-validation the risk is higher that grid search will select hyper-parameter value combinations that perform very well on a specific train-test split but poorly otherwise. K-fold gives us that extra confidence in building our model which is more likely to perform better on an independant dataset or the doployment model.

# ### Implementation: Fitting a Model
# Your final implementation requires that you bring everything together and train a model using the **decision tree algorithm**. To ensure that you are producing an optimized model, you will train the model using the grid search technique to optimize the `'max_depth'` parameter for the decision tree. The `'max_depth'` parameter can be thought of as how many questions the decision tree algorithm is allowed to ask about the data before making a prediction. Decision trees are part of a class of algorithms called *supervised learning algorithms*.
# 
# In addition, you will find your implementation is using `ShuffleSplit()` for an alternative form of cross-validation (see the `'cv_sets'` variable). While it is not the K-Fold cross-validation technique you describe in **Question 8**, this type of cross-validation technique is just as useful!. The `ShuffleSplit()` implementation below will create 10 (`'n_splits'`) shuffled sets, and for each shuffle, 20% (`'test_size'`) of the data will be used as the *validation set*. While you're working on your implementation, think about the contrasts and similarities it has to the K-fold cross-validation technique.
# 
# For the `fit_model` function in the code cell below, you will need to implement the following:
# - Use [`DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) from `sklearn.tree` to create a decision tree regressor object.
#   - Assign this object to the `'regressor'` variable.
# - Create a dictionary for `'max_depth'` with the values from 1 to 10, and assign this to the `'params'` variable.
# - Use [`make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html) from `sklearn.metrics` to create a scoring function object.
#   - Pass the `performance_metric` function as a parameter to the object.
#   - Assign this scoring function to the `'scoring_fnc'` variable.
# - Use [`GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) from `sklearn.grid_search` to create a grid search object.
#   - Pass the variables `'regressor'`, `'params'`, `'scoring_fnc'`, and `'cv_sets'` as parameters to the object. 
#   - Assign the `GridSearchCV` object to the `'grid'` variable.

# In[37]:

# TODO: Import 'make_scorer', 'DecisionTreeRegressor', and 'GridSearchCV'
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a 
        decision tree regressor trained on the input data [X, y]. """
    
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    
    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor(random_state=0)

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': list(range(1,11))}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer' 
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, scoring = scoring_fnc, cv = cv_sets)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_


# ### Making Predictions
# Once a model has been trained on a given set of data, it can now be used to make predictions on new sets of input data. In the case of a *decision tree regressor*, the model has learned *what the best questions to ask about the input data are*, and can respond with a prediction for the **target variable**. You can use these predictions to gain information about data where the value of the target variable is unknown — such as data the model was not trained on.

# ### Question 9 - Optimal Model
# _What maximum depth does the optimal model have? How does this result compare to your guess in **Question 6**?_  
# 
# Run the code block below to fit the decision tree regressor to the training data and produce an optimal model.

# In[40]:

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])


# **Answer: **
# Optimal model is 4 for the model. So I am happy because that is what I was suspecting earlier that 4 or 5 could prove to be better than max_depth=3.

# ### Question 10 - Predicting Selling Prices
# Imagine that you were a real estate agent in the Boston area looking to use this model to help price homes owned by your clients that they wish to sell. You have collected the following information from three of your clients:
# 
# | Feature | Client 1 | Client 2 | Client 3 |
# | :---: | :---: | :---: | :---: |
# | Total number of rooms in home | 5 rooms | 4 rooms | 8 rooms |
# | Neighborhood poverty level (as %) | 17% | 32% | 3% |
# | Student-teacher ratio of nearby schools | 15-to-1 | 22-to-1 | 12-to-1 |
# *What price would you recommend each client sell his/her home at? Do these prices seem reasonable given the values for the respective features?*  
# **Hint:** Use the statistics you calculated in the **Data Exploration** section to help justify your response.  
# 
# Run the code block below to have your optimized model make predictions for each client's home.

# In[41]:

# Produce a matrix for client data
client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)


# **Answer: **
# Model Predictions below:
# Predicted selling price for Client 1's home: $410,803.45
# Predicted selling price for Client 2's home: $231,639.13
# Predicted selling price for Client 3's home: $893,100.00
# 
# In Data exploration I had said for each variable **RM** or Total number of rooms in home or dwelling would bring the prices for the property up if number of rooms go up. We see this with Client 1 and Client 3 where the property prices are substantially up.
# 
# For the feature LSTAT or Neibhorhood poverty level the highest percentage is that of Client 2 at 32\% and this is the main reason for bringing the prices down for a 4 room dwelling to $231639.13. Even for Client 1 with a 5 room house the LSTAT of 17% has brough down the price of the property. Whereas for Client 3 the LSTAT is low at 3% and it looks like a posh locality so the house prices are significantly higher at $893,100.00
# 
# The PTRATIO is the Pupil to Teachers ratio which is also a factor looked at by parents before shifting to a new locality for the respective schools. The highest PTRATIO being for Client 2 which has a higher LSTAT ratio signifying that there are more poor people in the neighborhood is at 22:1 this has brought the property price for client 2 significantly down. For Client 1 although comparatively better PTRATIO than Client 2 but still being higher than Client 3 does not command premium price relative to Client 3's property. The lowest PTRATIO is that of Client 3 which has jacked up the price of the property.
# 
# In conclusion I would say the biggest impacting features are the RM and LSTAT which sway the prices in either direction based on their inputs. PTRATIO also has some impact but I would guess it would follow the pattern with LSTAT at leas in this Client sample.
# 
# 
# 

# ### Sensitivity
# An optimal model is not necessarily a robust model. Sometimes, a model is either too complex or too simple to sufficiently generalize to new data. Sometimes, a model could use a learning algorithm that is not appropriate for the structure of the data given. Other times, the data itself could be too noisy or contain too few samples to allow a model to adequately capture the target variable — i.e., the model is underfitted. Run the code cell below to run the `fit_model` function ten times with different training and testing sets to see how the prediction for a specific client changes with the data it's trained on.

# In[43]:

vs.PredictTrials(features, prices, fit_model, client_data)


# ### Question 11 - Applicability
# *In a few sentences, discuss whether the constructed model should or should not be used in a real-world setting.*  
# **Hint:** Some questions to answering:
# - *How relevant today is data that was collected from 1978?*
# - *Are the features present in the data sufficient to describe a home?*
# - *Is the model robust enough to make consistent predictions?*
# - *Would data collected in an urban city like Boston be applicable in a rural city?*

# **Answer: **
# 
# Although the data was collected in 1978 but it was scaled to current market prices. However to deploy this model in a real world setting if we are to deploy this model we need real current data maybe last 2 or three years should help gauge as to how the market has changed since then.
# 
# From 1978 onwards a lot of things change people's preferences the buying patterns etc. New financial mortgage instruments etc. which may come in way of making simple property buying decisions. I can tell you from my case that different people do different trade-offs between locality, availability of amount of funds, Previous owner's goodwill value(Rich famous owner like an author will command higher price although in a relatively poor locality), location of property(Property near the main road is likely to fetch higher prices than one in the interior). Or a property very near a school is bound to command a premium than one away from it if children's education is the main buying criteria. Some people want privacy and like to live in a quite neighborhood and are willing to commute for school or office. This model does not take care of all these preferences of the buyers.
# 
# When we did predictTrials above we say a very high variation of Range in prices: $105,700.00 which shows that the model is not very robust and consistent in its performance.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
