1. SGDClassifier can be used to switch between the logistic (log loss ) or SVC(hinge loss) and it works great for large datasets.
It stands for sorchatic gradient descent clasifier and uses alpha which is 1/c where c is inverse of regularisation
so larger alpha means more regularisation

linear_classifier = SGDClassifier(random_state=0)
parameters = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
             'loss':['log', 'hinge'], 'penalty':['l1', 'l2']}
searcher = GridSearchCV(linear_classifier, parameters, cv=10)
searcher.fit(X_train, y_train)

print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))


2. For using SVM you can either use LinearSVC or SVC . SVC allows to use a kernal to fit non linear decision boundary by creating additional
features. Gamma controls the smoothness of the decision boundary , more the value more complex the decision boundary hence overfitting
C is inverse of regularisation larger the value lower the regularisation.

svm = SVC()
parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
searcher = GridSearchCV(svm, parameters)
searcher.fit(X_train, y_train)

3. Support vectors are defined as training examples that influence the decision boundary,the algo learns the decision fn only using
these support vectors

4. Logistic regression can be used to perform multi class classsfication by either suing one vs rest or using softmax(multinomial)

lr_ovr = LogisticRegression() # does one vs rest hence creates n model where n is the no of target classes
lr_ovr.fit(X_train, y_train)
# Fit softmax classifier
lr_mn = LogisticRegression(multi_class="multinomial", solver="lbfgs") # creates only one model 
lr_mn.fit(X_train, y_train)

5. Both logistic and SVM are linear classifiers , and can be used with kernels but the logistic with kernel is slow as t uses 
all data , logistic output meaningful prob whereas svm does not. Both of them can do multi class clasification. L1 or l2 can be applied
to logistic where as l2 is used in svm.

Logistic hyper- C, penalty , multiclass 
SVC- C, kernel, gamma


6. Voting Classifier uses ensemble technique to combine multiple predictors(Same data multiple estimators)

from sklearn.ensemble import VotingClassifier
lr = LogisticRegression(random_state=SEED)
knn = KNN(n_neighbors=27)
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)
# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
vc = VotingClassifier(estimators=classifiers)  
vc.fit(X_train, y_train)   
y_pred = vc.predict(X_test)# pass the list of tuples that have the indivisual estimators


7. BaggingClassifier (same estimator , multiple dataset created using bootstrap aggregation)

dt = DecisionTreeClassifier(random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
bc.fit(X_train, y_train)
y_pred = bc.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test)) 


8. You can estimate the performance of the enseble model using Out of Bag instances as on an average 63% of 
training samples are sampled at any time and 37% constitute the OOB instances when using bootstraping . 
The model gets trained on the bootstraped samples and evaluated on the OOB and then you average out the OOB 
scores for all the estimators you have in your ensemble.

bc = BaggingClassifier(base_estimator=dt,n_estimators=50,oob_score=True,random_state=1)
bc.fit(X_train, y_train)
bc.oob_score_

It is generally seen that the OOB score is very close to the score on the test set this 
OOB serves as a great way of providing the generalisation capability of the model.

9. Random forest is an ensemble technique that uses decision tree as base estimator , each estimator trained on a 
diff bootstrap sample having the same size as the training set, RF further introduces randomisation in training of 
individual trees by picking up d features at each node without replacement.


10. IN boosting you can do Adaboost which weights the indivisual observation , Gradient boosting which uses the residuals 
to train succesive models or you can use Sorchastic Gradient boosting which submsamples the data without replacement for 
each estimatior and also the features at each split to create more 
diversity among the learners

from sklearn.ensemble import GradientBoostingRegressor
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9, # use 90% of training data for each predictor
            max_features=0.75, # use 75% of features at each split
            n_estimators=200,                                
            random_state=2)  # this will create a Sorchastic gradient boosting algo , if you remove submsaple and max_feature you end up Gradient boosting

11. Decision tree will have categorical or cotninous value in the leaf depending upon the task (reg or classification) 
but in CART each leaf always contain a real valued score which can be later converted/thresholded to categories for a 
classification problem


12. For hyper parameter tuning you can use either Grid search , Random search or Bayesian optimisation . After doing any of those
you can visualise the results of different params to study the effect on your model. Gridsearch is exhaustive while. Use 
these with a cv object which could be a simple Kfold or StratifiedKfold and pass it to GridSearchCV or RandomSearchCV



