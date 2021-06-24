from collections import Counter
import pandas as pd
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble.forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
#from sklearn.utils import shuffle

class ReverseSMOTE(object):
  def __init__(self, k_neighbors=5, random_state=None):
    self.k = k_neighbors
    self.random_state = random_state
  def sample(self, n_samples):
      """Generate samples.

      Parameters
      ----------
      n_samples : int
          Number of new synthetic samples.

      Returns
      -------
      S : array, shape = [n_samples, n_features]
          Returns synthetic samples.
      """
      np.random.seed(seed=self.random_state)

      S = np.zeros(shape=(n_samples, self.n_features))
      i=0
      while(i!=n_samples):
      # Calculate synthetic samples.
      # for i in range(n_samples):
          j = np.random.randint(0, self.X.shape[0])

          # Find the NN for each sample.
          # Exclude the sample itself.
          rnn = self.reverseNN(pd.DataFrame(self.X), j)
          # print(rnn,j)
          if rnn:
            nn_index = np.random.choice(rnn)
            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]
            i=i+1
          else:
            i=i-1
      return S
  def reverseNN(self, X, i):
    rnn=[]
    for index, row in X.iterrows():
      NN=self.neigh.kneighbors([row],return_distance=False)[:, 1:]
      # print(NN[0])
      if i in NN[0]:
      #   print(j)
        rnn.append(index)
    # print(rnn)
    return rnn

  def fit(self, X):
      """Train model based on input data.

      Parameters
      ----------
      X : array-like, shape = [n_minority_samples, n_features]
          Holds the minority samples.
      """
      self.X = X
      self.n_minority_samples, self.n_features = self.X.shape

      # Learn nearest neighbors.
      self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
      self.neigh.fit(self.X)
      

      return self