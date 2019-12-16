# 1.) # remove constant features
constant_features = [
    feat for feat in X_train.columns if X_train[feat].std() == 0
]
 
X_train.drop(labels=constant_features, axis=1, inplace=True)
X_test.drop(labels=constant_features, axis=1, inplace=True)


# 2.)remove quasi-constant feature
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
sel.fit(X_train)
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)


# 3.) Cjeck duplicated values
df.T.drop_duplicates()


# 4.)  find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
 
corr_features = correlation(X_train, 0.8)
X_train.drop(labels=corr_features, axis=1, inplace=True)
X_test.drop(labels=corr_features, axis=1, inplace=True)

# 5.) Check for duplicated features

duplicated_feat = []
for i in range(0, len(X_train.columns)):
    if i % 10 == 0:  # this helps me understand how the loop is going
        print(i)
 
    col_1 = X_train.columns[i]
 
    for col_2 in X_train.columns[i + 1:]:
        if X_train[col_1].equals(X_train[col_2]):
            duplicated_feat.append(col_2)
            
# 6.) Remove features using univariate roc_auc
from sklearn.metrics import roc_auc_score
roc_values = []
for feature in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train[feature].fillna(0).to_frame(), y_train)
    y_scored = clf.predict_proba(X_test[feature].fillna(0).to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))
roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns
roc_values.sort_values(ascending=False).plot.bar(figsize=(20, 8))

selected_feat = roc_values[roc_values>0.5]

# 7.) Remove features using Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
sel_.fit(scaler.transform(X_train), y_train)
X_train_lasso = pd.DataFrame(sel_.transform(X_train))
X_test_lasso = pd.DataFrame(sel_.transform(X_test))
X_train_lasso.columns = X_train.columns[(sel_.get_support())]
X_test_lasso.columns = X_train.columns[(sel_.get_support())]


# 8.) Feature importance through trees
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

sel_ = SelectFromModel(RandomForestClassifier(n_estimators=100))
sel_.fit(X_train.fillna(0), y_train)
selected_feat = X_train.columns[(sel_.get_support())]

# Recursive feature elimination
from sklearn.feature_selection import RFE
sel_ = RFE(RandomForestClassifier(n_estimators=100), n_features_to_select=10)
sel_.fit(X_train.fillna(0), y_train)

# by GrADIENT Boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, RFE
sel_ = SelectFromModel(GradientBoostingClassifier())
sel_.fit(X_train.fillna(0), y_train)

# RFE
sel_ = RFE(GradientBoostingClassifier(), n_features_to_select=10)
sel_.fit(X_train.fillna(0), y_train)

# 9.) Quick way to get feature importance withiout andy special cleaning
from sklearn.ensemble import RandomForestClassifier
# Create a copy to work with
X = train.copy()
# Save and drop labels
y = train.y
X = X.drop('y', axis=1)
# fill NANs 
X = X.fillna(-999)
# Label encoder
for c in train.columns[train.dtypes == 'object']:
    X[c] = X[c].factorize()[0]
rf = RandomForestClassifier()
rf.fit(X,y)
plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist(), rotation=90);


# 10.) Create new features from date

column_1 = dates.ix[:,0]

pd.DataFrame({"year": column_1.dt.year,
              "month": column_1.dt.month,
              "day": column_1.dt.day,
              "hour": column_1.dt.hour,
              "dayofyear": column_1.dt.dayofyear,
              "week": column_1.dt.week,
              "weekofyear": column_1.dt.weekofyear,
              "dayofweek": column_1.dt.dayofweek,
              "weekday": column_1.dt.weekday,
              "quarter": column_1.dt.quarter,
             })


###########Data camp

11. To perform feature selection you can simply drop features which have ~0 variance , you can drop features with a single 
class , you can drop features which  are heavliy correlated or you can drop features which are constant, all of these 
information can be derived using EDA on your dataset .

12. Feature selection basically involves picking up features whereas feature extraction involves combining original features 
in meaning full way like what PCA does


13. Before using the VarianceThreshold selector from sklearn you should try to normalise your dataset by diving by mean 
vector othewise you would have features with unequal variance and you wont get the right set of value:

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.001)
sel.fit(head_df / head_df.mean())
mask = sel.get_support()
reduced_df = head_df.loc[:, mask]


14. Above we saw ways to select features based on there indivisual property (there variance , missing property , constants etc), 
we can also check how the features relate to one another for which we can use corelation matrix 

15. To better visualize the heatmap with the corelation matrix you can create a mask:
corr = ansur_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()


To drop features having high corr:

corr_matrix = ansur_df.corr().abs()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
tri_df = corr_matrix.mask(mask)
to_drop = [c for c in tri_df.columns if any(tri_df[c] >  0.95)]
reduced_df = ansur_df.drop(to_drop, axis=1)
print("The reduced dataframe has {} columns.".format(reduced_df.shape[1]))


16. One of the way to takle the problem of feature selection is using the model predictions , for ex we can fit a linear model 
and get the weight and drop the one with the smallest value , we can continue to do the same until we get desired no of 
features , you can use this in sklearn using RFE(Recurrsive feature elimination)


17. You can also use trees who have the inherent nature of selecting the features by calculating the feature importance and 
you can create a mask using the theresholding values , recommended way is to do that reccursively using RFE


18 In RFE you can provide the step parameter which tells that in each iterations how many features to drop setting a 
value to 5 will drop 5 features in each iteration

19. You can also use Lasso() to perform feature selection as it will force most of the insignificant features to end up being 
0 .To pick up best regularisation value of Lasso you can use the LassoCV() estimator that will give the best value of reg coffecitent based upon the cross validated set.



20. A novel way to perform feature selection is to combine the feature selection capability from multiple models and vote 
which features to keep for example if we have 3 voters we can keep feature which got 2 or more than 2 votes 

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

lcv = LassoCV() # model 1 for selecting k best features
lcv.fit(X_train, y_train)
lcv_mask = lcv.coef_!=0 # get all coff that is not 0

rfe_gb = RFE(estimator=GradientBoostingRegressor(), 
             n_features_to_select=10, step=3, verbose=1) # eliminate feature using gbregressor 
gb_mask = rfe_gb.support_ # get the support for feratures selected

rfe_rf = RFE(estimator=RandomForestRegressor(), 
             n_features_to_select=10, step=3, verbose=1)
rf_mask = rfe_rf.support_ # get the support for features selected

votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0) # sum up the votes for each features
meta_mask = votes==3 # select features that have 3 votes
X_reduced = X.iloc[:, meta_mask] # create a subset of data with features that got 3 votes


21. You can use RFECV for automatically select features

svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X, y)
print("Optimal number of features : %d" % rfecv.n_features_)


22. You can extrcat relevant information from PCA like the explained variance ratio and components . Components can be used to 
understand how each features are combined to form a PC , this helps us understand the roles of features in indivisual PC

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()),
        		 ('reducer', PCA(n_components=2))])

# Fit it to the dataset and extract the component vectors
pipe.fit(poke_df)
vectors = pipe.steps[1][1].components_.round(2) # get the components (weights in which original features are combined)

# Print feature effects
print('PC 1 effects = ' + str(dict(zip(poke_df.columns, vectors[0]))))
print('PC 2 effects = ' + str(dict(zip(poke_df.columns, vectors[1]))))

print(pipe.steps[1][1].explained_variance_ratio_) # explained var ratio for each PC
print(pipe.steps[1][1].explained_variance_ratio_.cumsum()) # cumulative sum of explained var ratio


23. n_component param for PCA can either take integer value which means no of features to keep or a float value between 0 and 1 which 
specify how of the variance in the original data should be preserved. Even though there is no single right way of getting the 
no of components that should work for your use case , you can conisder drawing an elbow plot of the explained varaince and 
pick up that many no of components where you see an elbow.

You can also do an inverse_transform of your principal components to get back the original data in original dimension , although 
there would be some loss in the data.

PCA(n_components=0.8) # preserve 80% of variance

