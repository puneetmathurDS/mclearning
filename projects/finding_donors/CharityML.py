# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))

# TODO: Total number of records
n_records = len(data.index)

# TODO: Number of records where individual's income is more than $50,000

l=data[data['income'] == ">50K"].index
n_greater_50k = len(l)

# TODO: Number of records where individual's income is at most $50,000
l=data[data['income'] == "<=50K"].index
n_at_most_50k = len(l)

# TODO: Percentage of individuals whose income is more than $50,000
p=float(n_greater_50k)/n_records*100.0
greater_percent =p

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)




# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
vs.distribution(features_raw, transformed = True)

# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# Show an example of a record with scaling applied
display(features_raw.head(n = 1))

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

# TODO: Encode the 'income_raw' data to numerical values
d={"<=50K": 0, ">50K": 1}
income = income_raw.map(d)
#Checking output
display(income_raw.head(10),income.head(10))

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
print encoded


# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# TODO: Calculate accuracy
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import fbeta_score
#from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

gnb = GaussianNB()
y_pred = gnb.fit(X_test, y_test).predict(X_test)

accuracy=accuracy_score(y_test, y_pred)

# TODO: Calculate F-score using the formula above for beta = 0.5
fscore = fbeta_score(y_test, y_pred, average='binary', beta=0.5)

# Print the results 
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train.head(sample_size), y_train.head(sample_size))
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] =  end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    beta=0.5
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test, predictions_test, beta)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results


# Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(criterion="entropy",random_state=0)
clf_C = SVC(random_state=0)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
#Defining function since percent is required 3 times
def getsample(percent):
    return int((float(percent)/100)*X_train.shape[0])

samples_1 = getsample(1.0)
samples_10 = getsample(10.0)
samples_100 = getsample(100.0)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

#--TEST CODE------------------
# Import the three supervised learning models from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Initialize the three models
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(criterion="entropy",random_state=0)
clf_C = SVC(random_state=0)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
def get_sample_size(percentage):
    return int((float(percentage)/100)*X_train.shape[0])

samples_1 = get_sample_size(1.0)
samples_10 = get_sample_size(10.0)
samples_100 = get_sample_size(100.0)

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


import numpy as np
from scipy.spatial import distance
print(features.columns)
print distance.correlation(features.age, income)
print distance.correlation(features['capital-gain'], income)
print distance.correlation(features['capital-loss'], income)
print distance.correlation(features['education-num'], income)
print distance.correlation(features['hours-per-week'], income)
print distance.correlation(features.age, features['capital-gain'])
print distance.correlation(features.age, features['capital-loss'])
print distance.correlation(features.age, features['education-num'])
print distance.correlation(features.age, features['hours-per-week'])

from scipy.stats.stats import pearsonr
features=pd.DataFrame(features)
pearsonr(features.age, income)
pearsonr(features['capital-gain'], income)
pearsonr(features['education-num'], income)
pearsonr(features['hours-per-week'], income)

############DOING THE GRID SEARCH CROSS VALIDATION############################
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from IPython.display import display
import pickle, os.path

# TODO: Initialize the classifier
clf = SVC(random_state=0)

# TODO: Create the parameters list you wish to tune
parameters = {'C':range(1,5),'kernel':['linear','poly','rbf','sigmoid'],'gamma':range(1,10)}

# TODO: Make an fbeta_score scoring object
beta=0.5
score=fbeta_score(y_test, y_pred, beta)
scorer = make_scorer(score)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))


#################trying
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from IPython.display import display
import pickle, os.path
from sklearn.svm import SVC

def getscore(y_true, y_predict):
    return fbeta_score(y_true, y_predict, beta)

best_clf = None


#Note to self: Do not proceed if the below Grid Search Parameter is not enabled, it fails
#Enabling Grid Search
GRID_SEARCH_ENABLED = True

#Initialize the classifier
#Note to Reviewer: If you want to run the code optimally then use max_iter=-1
#My code ran Ran Overnight so limited to max_iter=1 to limit calculation time
clf = SVC(random_state=0, max_iter=1)

# Create the parameters list you wish to tune
parameters = {'C':range(1,100),'kernel':['linear','poly','rbf','sigmoid'],'degree':range(1,6)}

# Make an fbeta_score scoring object
scorer = make_scorer(getscore)

# Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

    # Fit the grid search object to the training data and find the optimal parameters   
grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
best_clf = grid_fit.best_estimator_

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))

# Print the final parameters
df = pd.DataFrame(grid_fit.grid_scores_).sort_values('mean_validation_score').tail()
display(df)
print "Parameters for the optimal model: {}".format(clf.get_params())


print(data.drop(['income'], axis=1).dtypes)



##########-----------------------TESTING#####################################
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

eclf = ExtraTreesClassifier()
eclf = eclf.fit(X_train, y_train)
fi=eclf.feature_importances_

print(len(fi))
x=0
together=pd.DataFrame()
together['features']=0
ft=np.array(features.columns)
for x in range(0,103):
    print(ft[x],fi[x])
    together['features'][x]=ft[x]
    together['importance'][x]=fi[x]

print(together.shape)
print(together)
    
import matplotlib.pyplot as plt
plt.plot(fi) #Better
plt.hist(fi)
# plot
plt.bar(range(len(eclf.feature_importances_)), eclf.feature_importances_)
plt.show()
print(features)
features=pd.DataFrame(features)
importances = eclf.feature_importances_
indices= np.argsort(importances)[::-1]
eclf.n_features_

for i in range(0,103):
    print(features[i],importances[indices[i]])

print(features,fi)
print(features)
features.columns



