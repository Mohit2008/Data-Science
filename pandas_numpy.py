df=pd.read_csv("path") # import file

df.columns.values # get all column names

df[(df.column > value1) & (df.column < value2)] # filter data based onmultiple condition

### Iterate over dataframe
for item, row in df.iterrows():
  print row()
####  

df[column].value_counts() # get the count of all unique values

df[df['A'].isin([3, 6])] # filter column elements in a list

series.apply(lambda x: 0 if x=='a' else 1) # apply a funct to every elemnent in a series

df.columns = ['class', 'x1', 'x2', ...] #set the column of a dataframe

df.dtypes,df.describe(),df.isnull().sum(),df.shape,df.index,df.columns,df.info(),df.count() # general commands

## selecting elements from a dataframe

df_subset = df[['x1', 'x2', ... ]] #
x1_list = df.iloc[0:100, 1].values #get by element
x1_list = df.loc[row_index, col_index] # get by index
df.ix[1,'Capital'] #hybrid
###

df.drop('Country', axis=1) # drop columns
 
df.sort_index() #Sort by labels along an axis
df.sort_values(by='Country') #Sort by the values along an axis

### Apply functions on dataframe
>>> f = lambda x: x*2
>>> df.apply(f) # Apply function on each row
>>> df.applymap(f) # apply func on each element
####

df.rename(columns={'old_name': 'new_ name'}) # rename a column

df.set_index('column_one') # set an index for df

df.groupby(col1)[col2] # group by col1 and get the info from col2

df.pivot_table(index=col1,values=[col2,col3],aggfunc=mean) #Create a pivot table that groups by col1 and calculates the mean of col2 and col3

### Join dataframes
df1.append(df2) #Add the rows in df1 to the end of df2 (columns should be identical)
pd.concat([df1, df2],axis=1) #Add the columns in df1 to the end of df2 (rows should be identical)
df1.join(df2,on=col1,how='inner')
###

df.str_columnName.str.anyStringFunction # apply string functions on a any column elements

df.groupby('name')['activity'].value_counts().unstack().fillna(0) # Unstack switches the rows to columns

df.drop_duplicates( ) # drop duplicates

pd.get_dummies( ) # one hot encoding

income.Index.nunique() # no of unique values in a column

crops['cost'].fillna(value = "UNKNOWN",inplace = True) # replace na by a value

pd.crosstab(df.regiment, df.company, margins=True) # create a crosstab

df.at[dates[0],'A'] # get a scalar value

df.at[dates[0],'A'] = 0 ,df.iat[0,1] = 0 # set a scalar value

