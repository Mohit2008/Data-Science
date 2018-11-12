# 1.) find categorical variables
 categorical = [var for var in df.columns if df[var].dtype=='O']
# 2.) Check cardinality
for var in categorical:
    print(var, ' contains ', len(data[var].unique()), ' labels')
    
# 3.) print variables with missing data
for col in discrete:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())
