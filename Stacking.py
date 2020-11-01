# Stacking method 
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression, BayesianRidge, Lasso	
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

splitN = 4  #4 is the best one! # 5 was also good
regXGB = [XGBRegressor(learning_rate=0.01, n_estimators=7000,
max_depth=3, min_child_weight=0, gamma=0.001, subsample=0.5, colsample_bytree=0.1, colsample_bylevel= 0.9,
nthread=-1, scale_pos_weight=0.2, seed=5, reg_alpha=0.01) for i in range(splitN)]

regKernelRidge = [KernelRidge(alpha = 0.01, coef0= 100, degree = 1, kernel='poly') for i in range(splitN)]

regBayesianRidge = [BayesianRidge(n_iter=3000, tol = 1e-5, alpha_1 = 1e-8, alpha_2 = 1e-8, 
                    lambda_1 = 1e-8, lambda_2 = 1e-8) for i in range(splitN)]

refGBR = [GradientBoostingRegressor(loss = 'huber', learning_rate=0.02, 
                                    n_estimators=1200, max_depth=3, alpha = 0.25) for i in range(splitN)]

regLasso = [Lasso(alpha=0.0005, fit_intercept=True, normalize=False, precompute=False, 
             copy_X=True, max_iter=9000, tol=0.00001, warm_start=False, 
             positive=False, random_state=None, selection='cyclic') for i in range(splitN)]

reg_meta = [XGBRegressor() for i in range(splitN)]
# XGBRegressor() 12.012 with 4 folds


#XGBRegressor(learning_rate= 0.1, n_estimators=100, max_depth = 3, min_child_weight=0, gamma=0.001, subsample=0.5, colsample_bytree=0.1, colsample_bylevel= 0.9,
#nthread=-1, scale_pos_weight=0.2, seed=5, reg_alpha=0.01)



kf = KFold(n_splits=splitN, shuffle=True)
i = 0

y1_test = np.zeros((splitN, len(X_test_reduction.index)))
y2_test = np.zeros((splitN, len(X_test_reduction.index)))
y3_test = np.zeros((splitN, len(X_test_reduction.index)))
y4_test = np.zeros((splitN, len(X_test_reduction.index)))
y5_test = np.zeros((splitN, len(X_test_reduction.index)))


for train_index, valid_index in kf.split(X_train_reduction):
  #print("TRAIN:", train_index, "TEST:", test_index)
  Xftrain = X_train_reduction.loc[train_index].drop(columns=['SalePrice'])
  yftrain = X_train_reduction.loc[train_index]['SalePrice']

  Xfvalid = X_train_reduction.loc[valid_index].drop(columns=['SalePrice'])
  yfvalid = X_train_reduction.loc[valid_index]['SalePrice']


  regXGB[i].fit(Xftrain, yftrain)
  y1_val = regXGB[i].predict(Xfvalid)
  y1_test[i,:] = regXGB[i].predict(X_test_reduction)

  regKernelRidge[i].fit(Xftrain, yftrain)
  y2_val = regKernelRidge[i].predict(Xfvalid)
  y2_test[i,:] = regKernelRidge[i].predict(X_test_reduction)

  regBayesianRidge[i].fit(Xftrain, yftrain)
  y3_val = regBayesianRidge[i].predict(Xfvalid)
  y3_test[i,:] = regBayesianRidge[i].predict(X_test_reduction)

  refGBR[i].fit(Xftrain, yftrain)
  y4_val = refGBR[i].predict(Xfvalid)
  y4_test[i,:] = refGBR[i].predict(X_test_reduction)

  regLasso[i].fit(Xftrain, yftrain)
  y5_val = regLasso[i].predict(Xfvalid)
  y5_test[i,:] = regLasso[i].predict(X_test_reduction)

  data = {'y1':  y1_val, 'y2':  y2_val, 
          'y3':  y3_val, 'y4':  y4_val, 'y5':  y5_val}
  new_train_X_set = pd.DataFrame(data, columns = ['y1', 'y2', 
                                                  'y3', 'y4', 'y5'])

  
  reg_meta[i].fit(new_train_X_set, yfvalid)

  i = i+1
  print(i)



print(y1_test.shape)
y1_test[0,:].shape
np.zeros((splitN, y1_test.shape[1])).shape

