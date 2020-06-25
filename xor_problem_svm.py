# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:07:49 2020

@author: Sushilkumar.Yadav
"""

import matplotlib.pyplot as plt
import numpy as np

#%%
#XOR plotting
x_linear = np.array([[0,0],[0,1],[1,0],[1,1]])
t = np.array([0,1,1,0]).T
fig = plt.figure()
ax=fig.add_subplot(111)
plt.title('XOR plotting by Sushilkumar Yadav')
plt.scatter(x_linear[:,0], x_linear[:,1], s=200, c=t[:]*1)
plt.show()

#%%
#for 400 input samples plotting XOR code
import numpy as np
np.random.seed(0)
X = np.random.randn(400, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.where(y, 1, -1)
plt.Figure()
plt.scatter(X[y==1, 0], X[y==1, 1], c='yellow', marker='x', label='1')
plt.scatter(X[y==-1, 0], X[y==-1, 1], c='purple', marker='s', label='-1')
plt.title ('XOR plotting with large data')
plt.ylim(-4.0, 4.0)
plt.xlim(-4.0, 4.0)
plt.legend()
plt.show()

#%%
#XOR plotting using kernel
#refer 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('purple', 'yellow', 'lightgreen', 'gray', 'gray')
   cmap = ListedColormap(colors[:len(np.unique(x))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
   np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   # plot all samples
   X_test, y_test = X[test_idx, :], y[test_idx]
   for idx, cl in enumerate(np.unique(y)):
      plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
               alpha=0.8, c=cmap(idx),
               marker=markers[idx], label=cl)
   # highlight test samples
   if test_idx:
      X_test, y_test = X[test_idx, :], y[test_idx]
      plt.scatter(X_test[:, 0], X_test[:, 1], c='',
               alpha=1.0, linewidth=1, marker='o',
               s=55, label='test set')

# create xor dataset
np.random.seed(0)
X_xor = np.random.randn(400, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

# SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)

# draw decision boundary
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.title('Plotting XOR using SVM RBF Kernel')
plt.show()
