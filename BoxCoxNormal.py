#  Box-Cox
aa = np.array(X_train_reduction.skew(axis = 0).abs()>0.5)
k=0
for i in range(len(X_train_reduction.skew(axis = 0))):
  if k<len(X_train_reduction.skew(axis = 0)):
    if aa[i]:
   #   print(i)
      a = np.array(X_train_reduction[X_train_reduction.columns[k]].values)+1
      #print(a)
      #print(a.shape)
      if not np.all(np.array(X_valid_reduction[X_valid_reduction.columns[k]]) == np.array(X_valid_reduction[X_valid_reduction.columns[k]])[0]):
        if not np.all(np.array(X_test_reduction[X_test_reduction.columns[k]]) == np.array(X_test_reduction[X_test_reduction.columns[k]])[0]):
          X_train_reduction[X_train_reduction.columns[k]],fitted_lambda = stats.boxcox(np.array(X_train_reduction[X_train_reduction.columns[k]].values)+1)
          X_valid_reduction[X_valid_reduction.columns[k]] = stats.boxcox(np.array(X_valid_reduction[X_valid_reduction.columns[k]].values)+1, fitted_lambda)
          vv = np.array(X_test_reduction[X_test_reduction.columns[k]])
        #  for i in range(len(vv)):
         #  print(vv[i]+1)
          X_test_reduction[X_test_reduction.columns[k]] = stats.boxcox(np.array(X_test_reduction[X_test_reduction.columns[k]].values)+1, fitted_lambda)
  k = k+1
