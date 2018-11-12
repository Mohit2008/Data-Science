import numpy as np
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # setting the seaborn style sheet
np.random.seed(3064) # set the random seed

csize = 10;       # number of centers
p = 2;      # no of classes
s = 1;      # sd for generating the centers within each class                    
m1 = np.reshape(np.random.normal(size=csize*p), (csize, p))*s + np.reshape([1, 0]*csize,  (csize,p))
m0 = np.reshape(np.random.normal(size=csize*p), (csize, p))*s + np.reshape([0, 1]*csize,  (csize,p))

#-----Train data
n = 100;  
id1 = np.random.choice(csize,n) # Randomly allocate the n samples for class 1  to the 10 clusters
id0 = np.random.choice(csize,n)# Randomly allocate the n samples for class 1 to the 10 clusters

s = np.sqrt(float(1)/5);                               # sd for generating x. 
traindata = np.reshape(np.random.normal(size= 2*n*p), (2*n, p))*s + np.concatenate(
    (m1[id1,:], m0[id0,:]),axis=0)
Ytrain = np.concatenate(([1]*n, [0]*n))


#--------Test data
N = 5000;  
id1 = np.random.choice(csize, N)
id0 = np.random.choice(csize, N)  
testdata = np.reshape(np.random.normal(size = 2*N*p), (2*N, p))*s + np.concatenate(
    (m1[id1, :], m0[id0, :]), axis = 0)
Ytest = np.concatenate(([1]*N, [0]*N))



def mixnorm(x):
  ## return the density ratio for a point x, where each 
  ## density is a mixture of normal with 10 components
    return np.sum(np.exp(-np.sum((np.sum((m1, -x), axis = 0))**2,
                                 axis=1)*float(5)/float(2)))/np.sum(np.exp(-np.sum((np.sum((m0, -x), axis = 0))**2,
                                 axis=1)*float(5)/float(2)))
    
    

# bayes error
Ytest_pred_Bayes = [mixnorm(x)>1 for x in testdata]
test_err_Bayes = sum(Ytest !=  Ytest_pred_Bayes) / float(2*N)


# linear regression
reg = linear_model.LinearRegression()
reg.fit(traindata, Ytrain)
predction_train=reg.predict(traindata)
predction_test=reg.predict(testdata)

output_train=[1 if pred>0.5 else 0 for pred in (predction_train)]
output_test=[1 if pred>0.5 else 0 for pred in (predction_test)]

correct_train = np.array(output_train) == Ytrain
correct_test = np.array(output_test) == Ytest

train_err_LS = (1-(np.sum(correct_train) / len(Ytrain)))
test_err_LS = (1-(np.sum(correct_test) / len(Ytest)))




# knn
myk = [151, 101, 69, 45, 31, 21, 11, 7, 5, 3, 1]
m = len(myk);

train_err_knn = []
test_err_knn = []

for k in myk:
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(traindata, Ytrain)
    pred_train=neigh.predict(traindata)
    pred_test=neigh.predict(testdata)
    
    correct_train = np.array(pred_train) == Ytrain
    correct_test = np.array(pred_test) == Ytest
    
    error_train = (1-(np.sum(correct_train) / len(Ytrain)))
    error_test = (1-(np.sum(correct_test) / len(Ytest)))
    train_err_knn.append(error_train)
    test_err_knn.append(error_test)
    
    

df = 2*n//np.array(myk)
fig=plt.figure(num=None, figsize=(9, 7), dpi=500, facecolor='w', edgecolor='k')
ax1 = plt.subplot(1,1,1)
ax2 = ax1.twiny()


pknn_train = ax1.plot(range(m), train_err_knn, color = "blue", linestyle=':',marker = "o",label="Train error knn")
pknn_test = ax1.plot(range(m), test_err_knn, color = "red", linestyle=':',marker = "o",label="Test error knn")
ax1.plot(2, train_err_LS, marker = 's', linewidth = '2', color = "orange")
ax1.annotate("Linear regression train error",(2, train_err_LS),xytext=(3, train_err_LS+0.017),arrowprops=dict(facecolor='orange', shrink=0.05),)
ax1.plot(2, test_err_LS, marker = 's', linewidth = '2', color = "green")
ax1.annotate("Linear regression test error",(2, test_err_LS),xytext=(2.7, test_err_LS+0.01),arrowprops=dict(facecolor='green', shrink=0.05),)
pbayes = ax1.axhline(y = test_err_Bayes, color = 'green', linestyle = '-',label="Bayes")
ax1.set_xticks(range(m))
ax1.set_xticklabels(df)
ax1.set_xlabel('Degree of Freedom- N/k')
ax1.set_ylabel('Misclassification Error')
ax1.legend()

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(range(m))
ax2.set_xticklabels(myk)
ax2.set_xlabel('k −  Number of Nearest Neighbors')
#ax1.title("Plot showing the effect on training and testing error for bivariate Gaussian distrubtion for different classifier")

plt.savefig('AssignmentOutput_1_3064_mkhanna2.pdf',dpi=1000,bbox_inches="tight", pad_inches=0.5)
print("Execution Succesfull, and the output is saved to disk")




######################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as nn
import sklearn.linear_model as lm

np.random.seed(1111) # set the random seed

csize = 10;       # number of centers
p = 2;      # no of classes
s = 1;      # sd for generating the centers within each class                    
m1 = np.reshape(np.random.normal(size=csize*p), (csize, p))*s + np.reshape([1, 0]*csize, (csize,p))
m0 = np.reshape(np.random.normal(size=csize*p), (csize, p))*s + np.reshape([0, 1]*csize, (csize,p))


n = 100;  # no of train samples in each class
id1 = np.random.choice(csize,n)# Randomly allocate the n samples for class 1  to the 10 clusters
id0 = np.random.choice(csize,n)  # Randomly allocate the n samples for class 1 to the 10 clusters
s = np.sqrt(float(1)/5); # sd for generating x.



def mixnorm(x):
    return np.sum(np.exp(-np.sum((np.sum((m1, -x), axis = 0))**2,
                                 axis=1)*float(5)/float(2)))/np.sum(np.exp(-np.sum((np.sum((m0, -x), axis = 0))**2,
                                 axis=1)*float(5)/float(2)))

#Train data generation
traindata = np.reshape(np.random.normal(size= 2*n*p), (2*n, p))*s + np.concatenate(
    (m1[id1,:], m0[id0,:]),axis=0)
Ytrain = np.concatenate(([1]*n, [0]*n))


# Test data generation
N = 5000;  # no of test samples in each class
id1 = np.random.choice(csize, N)
id0 = np.random.choice(csize, N)  
testdata = np.reshape(np.random.normal(size = 2*N*p), (2*N, p))*s + np.concatenate(
    (m1[id1, :], m0[id0, :]), axis = 0)
Ytest = np.concatenate(([1]*N, [0]*N))


# get bayes error
Ytest_pred_Bayes = [mixnorm(x)>1 for x in testdata]
test_err_Bayes = sum(Ytest !=  Ytest_pred_Bayes) / float(2*N)



# do classification using linear regression by setting a threshold at 0.5
regr = lm.LinearRegression()
regr.fit(traindata, Ytrain)
Ytrain_pred_LS = (regr.predict(traindata)>0.5)
Ytest_pred_LS = (regr.predict(testdata)>0.5)
train_err_LS = sum(Ytrain !=  Ytrain_pred_LS) / float(2*n);  
test_err_LS = sum(Ytest !=  Ytest_pred_LS) / float(2*N);


# do classification using knn at different k
myk = [151, 101, 69, 45, 31, 21, 11, 7, 5, 3, 1]
m = len(myk);

train_err_knn = [0]*m;
test_err_knn = [0]*m;

for j in range(m):
    nbrs = nn.KNeighborsClassifier(n_neighbors = myk[j]).fit(traindata, Ytrain)
    Ytrain_pred = nbrs.predict(traindata)
    train_err_knn[j] = sum(Ytrain != Ytrain_pred)/float(2*n)
    Ytest_pred = nbrs.predict(testdata)
    test_err_knn[j] = sum(Ytest != Ytest_pred)/float(2*N)
    

# do plotting
df = 2*n//np.array(myk)
fig=plt.figure(figsize=(10,6))
ax1 = plt.subplot(1,1,1)
ax2 = ax1.twiny()
pknn_train = ax1.scatter(range(m), train_err_knn, color = "blue", marker = "o", alpha = 0.5)
pknn_test = ax1.scatter(range(m), test_err_knn, color = "red", marker = "o", alpha = 0.5,linestyle='-')
ax1.scatter(2, train_err_LS, s = 100, marker = '^', edgecolor = 'blue', linewidth = '2')
ax1.scatter(2, test_err_LS, s = 100, marker = '^', edgecolor = 'red', linewidth = '2')
pbayes = ax1.axhline(y = test_err_Bayes, color = 'green', linestyle = '-')

ax1.set_xticks(range(m))
ax1.set_xticklabels(df)
ax1.set_xlabel('Degree of Freedom - N/k')
ax1.set_ylabel('Test Error')
ax1.legend((pknn_train, pknn_test, pbayes), 
           ('train error', 'test error', 'Bayes error'), loc = 'lower left')

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(range(m))
ax2.set_xticklabels(myk)
ax2.set_xlabel('k − Number of Nearest Neighbors')
plt.show()

#plt.savefig('AssignmentOutput_1_UID_NETID.pdf',dpi=1000)
