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
