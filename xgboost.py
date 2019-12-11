# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)

#-------------------------------------

#Evaluating model with k fold cross validation
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

model = XGBClassifier()
kfold = KFold(n_splits=10, random_state=7) 
results = cross_val_score(model, X, Y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# for class imbalanced use stratified splits
kfold = StratifiedKFold(n_splits=10, random_state=7)

#-------------------------------------------------

# plotting a tree
from xgboost import XGBClassifier
from xgboost import plot_tree
from matplotlib import pyplot

model = XGBClassifier()
model.fit(X, y)
plot_tree(model, num_trees=0, rankdir='LR') # num_trees means the index of tree to be plotted and rankdir means to plot left to right
pyplot.show()

#---------------------------------------------------

#Saving and loading model

import pickle
pickle.dump(model, open("pima.pickle.dat", "wb"))
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))


#---------------------------------------------------

#Printing and plotting feature importance
from xgboost import plot_importance

model = XGBClassifier()
model.fit(X, y)
print(model.feature_importances_)
plot_importance(model)
pyplot.show()


#-----------------------------------------------------

#Feature selection using xgboost and sklearn selectFromModel

from sklearn.feature_selection import SelectFromModel

thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	selection = SelectFromModel(model, threshold=thresh, prefit=True) # select features using threshold
	select_X_train = selection.transform(X_train)
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
		accuracy*100.0))


#-------------------------------------------------------

#perform early stopping

eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss",eval_set=eval_set, verbose=True)

#differnt eval_metric supported are:rmse,mae ,logloss, mlogloss,error, auc

#--------------------------------------------------------

#Enable multithreading in xgboost

model = XGBClassifier(nthread=n) # n is the no of cpu cores on your machine

# you can also enable parallelism while using cross validation 
results = cross_val_score(model, X, label_encoded_y, cv=kfold, scoring='neg_log_loss',n_jobs=1, verbose=1)



#-------------------------------------------------------------
Perform grid search

from sklearn.model_selection import GridSearchCV
n_estimators = range(50, 400, 500) # no of trees
max_depth = range(1, 11, 2) # depth of tree
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] # learning rate
subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] # row subsamples
colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] # col subsamples by tree
colsample_bylevel = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0] # col subsample for each split
param_grid = dict(n_estimators=n_estimators,max_depth=max_depth,learning_rate=learning_rate,subsample=subsample,
	colsample_bytree=colsample_bytree,colsample_bylevel=colsample_bylevel)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=1, cv=kfold)
result = grid_search.fit(X, label_encoded_y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



#----------------------------------------------------------------------------------
Xgboost cross validation using built in 

import xgboost as xgb 
churn_dmatrix = xgb.DMatrix(data=X, label=y) # xgboost own optimised data structure
params = {"objective":"reg:logistic", "max_depth":3}
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=123) # boost_round means no of trees and metric can have any option like error , auc etc...

#-------------------------------------------------------
Common loss functions in Xgboost

reg:linear # for linear regression
reg:logistic # for classification with output as classes
binary:logistic # for get prob as output


- You can use 2 styles for Xgboost :
    - Sklearn api - xgb.XGBRegressor(objective="reg:linear", n_estimator=10,seed=10) # this will always use tree as base learner
    - Learner api with linear base learners :
                            DM_train = Xgb.DMatrix(data=Xtrain, label =ytrain)
                            params ={"booster": "gblinear", "objective": "reg:linear"}
                            xgb.train(DM_train, params , num_boost_round =10)
                            xgb.predict()

