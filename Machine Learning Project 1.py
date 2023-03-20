#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
data = pd.read_csv("/Users/joenemeczky/Desktop/Star39552_balanced.csv")


# In[49]:


data.head()


# In[50]:


df = data.drop('SpType', axis=1)
df = df.drop('e_Plx', axis=1)


# In[51]:


df.head()


# In[52]:


from sklearn import preprocessing
import numpy as np

min_max_scalar = preprocessing.MinMaxScaler()
df_min_max = min_max_scalar.fit_transform(df)
print(df_min_max)


# In[54]:


from sklearn.model_selection import train_test_split

X, y = df_min_max[:, 0:4], df_min_max[:, 4]


# In[55]:


X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.2,
                     random_state=0, 
                     stratify=y)


# In[99]:


print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)

X_train2 = X_train[:,0:2]
print(X_train2)


# In[57]:


print(X_test)
print(y_test)
print(X_test.shape)
print(y_test.shape)


# In[79]:


import numpy as np
class Perceptron:

    def __init__(self, eta=0.01, n_iter=50, random_state=1): 
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state 
        
    def fit(self, X, y):  
        rgen = np.random.RandomState(self.random_state) 
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.) 
        self.errors_ = [] 
            
        for _ in range(self.n_iter): 
            errors = 0  
            for xi, target in zip(X, y): 
                update = self.eta * (target - self.predict(xi)) 
                self.w_ += update * xi 
                self.b_ += update 
                errors += int(update != 0.0) 
            self.errors_.append(errors) 
        return self  
        
    def net_input(self, X): 
        return np.dot(X, self.w_) + self.b_ 
        
    def predict(self, X): 
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# In[119]:


from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
    
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan') 
    cmap = ListedColormap(colors[:len(np.unique(y))]) 
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
    np.arange(x2_min, x2_max, resolution)) 
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) 
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap) 
    plt.xlim(xx1.min(), xx1.max()) 
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)): 
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1], 
                    alpha=0.8, 
                    c=colors[idx], 
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')


# In[199]:


import matplotlib.pyplot as plt 
ppn5 = Perceptron(eta=0.1, n_iter=10)
ppn5.fit(X_train, y_train)
plt.plot(range(1, len(ppn5.errors_) + 1), 
         ppn5.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
predictions5 = ppn5.predict(X_train)
from sklearn.metrics import precision_score
pre_val5 = precision_score(y_true=y_train, y_pred=predictions5) 
print(f'Precision: {pre_val5:.3f}')


# In[198]:


import matplotlib.pyplot as plt 
ppn = Perceptron(eta=0.1, n_iter=30)
ppn.fit(X_train, y_train)
plt.plot(range(1, len(ppn.errors_) + 1), 
         ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
predictions = ppn.predict(X_train)
from sklearn.metrics import precision_score
pre_val = precision_score(y_true=y_train, y_pred=predictions) 
print(f'Precision: {pre_val:.3f}')


# The perceptron after 10 epochs starts to converge but never actually reaches a point of complete convergence. I incremently increased from 10 to 30 epochs and each perceptron was similar where it sort of started to converge but then continued to bounce around within a certain range. 

# In[216]:


confmat1 = confusion_matrix(y_true = y_train, y_pred = predictions)
print(confmat1)
print(5436+1092)
print(5436+1092+14728+10385)


# There are 6528 misclassifications out of the 31641 total cases. This is based on the perceptron with 30 iterations and a learning rate of 0.1

# In[213]:


import matplotlib.pyplot as plt 
ppn2 = Perceptron(eta=0.1, n_iter=30)


# In[209]:


plot_decision_regions(X_train2, y_train, classifier=ppn2) 
plt.xlabel('Vmag')
plt.ylabel('Plx')
plt.legend(loc='upper left')
plt.show()


# In[200]:


import matplotlib.pyplot as plt 
ppn3 = Perceptron(eta=0.25, n_iter=10)
ppn3.fit(X_train, y_train)
plt.plot(range(1, len(ppn3.errors_) + 1), 
         ppn3.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

predictions3 = ppn3.predict(X_train)
from sklearn.metrics import precision_score
pre_val3 = precision_score(y_true=y_train, y_pred=predictions3) 
print(f'Precision: {pre_val3:.3f}')


# In[217]:


import matplotlib.pyplot as plt 
ppn7 = Perceptron(eta=0.5, n_iter=10)
ppn7.fit(X_train, y_train)
plt.plot(range(1, len(ppn7.errors_) + 1), 
         ppn7.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

predictions7 = ppn7.predict(X_train)
from sklearn.metrics import precision_score
pre_val7 = precision_score(y_true=y_train, y_pred=predictions7) 
print(f'Precision: {pre_val7:.3f}')


# In[227]:


import matplotlib.pyplot as plt 
ppn4 = Perceptron(eta=0.33, n_iter=30)
ppn4.fit(X_train, y_train)
plt.plot(range(1, len(ppn4.errors_) + 1), 
         ppn4.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

predictions4 = ppn4.predict(X_train)
from sklearn.metrics import precision_score
pre_val4 = precision_score(y_true=y_train, y_pred=predictions4) 
print(f'Precision: {pre_val4:.3f}')


# Changing the learning rate didn't change the convergence that much when each one was set to have 10 iterations. When I started to changed both the learning rate and the number of iterations is when I started to see the most change in the convergence. Changing from a learning rate of 0.25 to 0.5 changed the number of incorrect classifications from 7276 to 3958.  

# In[242]:


from sklearn.metrics import confusion_matrix
confmat1 = confusion_matrix(y_true = y_train, y_pred = predictions)

print(confmat1)
print()

confmat3 = confusion_matrix(y_true = y_train, y_pred = predictions3)
print(confmat3)
print()

confmat4 = confusion_matrix(y_true = y_train, y_pred = predictions4)
print(confmat4)
print(1949+2003)
print()

confmat5 = confusion_matrix(y_true = y_train, y_pred = predictions5)
print(confmat5)
print()

confmat7 = confusion_matrix(y_true = y_train, y_pred = predictions7)
print(confmat7)
print(1993+1965)
print(1993+1965+13827+13856)
print((31641-3958)/31641)


# After testing out all of the confusion matrices for all of the models I found that the perceptron 4 with the inputs (eta=0.33, n_iter=30) and the perceptron 7 with inputs (eta=0.5, n_iter=10) performed the best on the training dataset.

# In[231]:


import matplotlib.pyplot as plt 
ppn6 = Perceptron(eta=0.5, n_iter=10)
ppn6.fit(X_train2, y_train)


# In[232]:


plot_decision_regions(X_train2, y_train, classifier=ppn6) 
plt.xlabel('Vmag')
plt.ylabel('Plx')
plt.legend(loc='upper left')
plt.show()


# In[240]:


predictions7test = ppn7.predict(X_test)
confmat7test = confusion_matrix(y_true = y_test, y_pred = predictions7test)
print(confmat7test)
print(482+509)
print(482+509+3447+3473)
print((7911-991)/7911)


# In[237]:


predictions4test = ppn4.predict(X_test)
confmat4test = confusion_matrix(y_true = y_test, y_pred = predictions4test)
print(confmat4test)
print(502+498)


# Since both perceptron 4 and 7 performed similarly on the training dataset I wanted to test both of them out on the test dataset as well. After predicting the values on the test dataset for both I found that perceptron 7 performed better and only had 991 misclassification compared to the 1000 misclassifications for perceptron 4. Overall I thought that the perceptron performed pretty well considering the data was not very linearly separable. It had a successful classification rate of 87.47% on the test dataset. This was almost identical in performance to the training dataset which had a successful classification rate of 87.49%. The only thing that I was confused about was there was not relationship between the precision score that I found and the classification rate. Almost all of the other perceptrons had a higher precision score than the final perceptron that I decided on however it successfully classified the target variable the best. 

# In[221]:


fig, ax = plt.subplots(figsize=(2.5, 2.5)) 
ax.matshow(confmat7test, cmap=plt.cm.Blues, alpha=0.3) 
for i in range(confmat7test.shape[0]): 
    for j in range(confmat7test.shape[1]): 
        ax.text(x=j, y=i, s=confmat7test[i, j], 
                va='center', ha='center') 
ax.xaxis.set_ticks_position('bottom')
plt.xlabel('Predicted label') 
plt.ylabel('True label') 
plt.show()

