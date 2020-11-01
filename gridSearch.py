# Tuning the hyperparameters using the grid search method
# We consider a set of parameters for each ML algorithm

XX = pd.DataFrame.to_numpy(X_train_reduction)
yy = pd.DataFrame.to_numpy(y_train)

from sklearn.model_selection import GridSearchCV
#parameters = [{#'learning_rate': [0.005, 0.01, 0.02],
              #'n_estimators': [5000, 6000, 7000],
              #'max_depth':[3,4,5,7,9, 13, 17],
              # 'min_child_weight':[0, 3, 5],
               #'gamma':[0.001, 0.01, 0.1, 1],
             # 'subsample':[0.1,  0.5, 0.9],
              # 'colsample_bytree': [0.1,  0.5, 0.9],
              # 'colsample_bylevel':[0.1,  0.5, 0.9],
              # 'colsample_bynode':[0.1,  0.5, 0.9],
             #  'scale_pos_weight': [0.2, 0.6, 1, 2, 4],
              # 'reg_alpha': [0.01, 0.05, 0.25, 1],
               #'reg_lambda': [0.01, 0.05, 0.25, 1]
 #              }]



#parameters = [{'n_iter': [30, 100, 200 ,300, 1000, 3000, 7000],
 #             'tol': [1e-5, 1e-4, 1e-3, 1e-2],
  #            'alpha_1':[1e-8, 1e-6, 1e-4, 1e-2],
   #            'alpha_2':[1e-8, 1e-6, 1e-4, 1e-2],
    #           'lambda_1':[1e-8, 1e-6, 1e-4, 1e-2],
     #         'lambda_2':[1e-8, 1e-6, 1e-4, 1e-2],
      #         }]


#parameters = [{'kernel':['additive_chi2', 'chi2', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid', 'cosine'],
   #           'alpha': [0.01, 0.1, 1, 10],
    #          'degree':[1, 2, 3, 4, 5],
    #           'coef0':[0.01, 0.1, 1, 10, 100]
     #          }]

#parameters = [{'kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
 #              'C': [0.01, 0.1, 1, 10],
  #             'degree':[1, 2, 3, 4, 5],
   #             'coef0':[0.01, 0.1, 1, 10, 100]}]



#parameters = [{'n_estimators': [50, 100, 200, 400, 800, 1600, 3200],
 #             'max_depth':[3,4,5,7,9, 13, 17]}]




parameters = [{#'learning_rate': [0.005, 0.02, 0.5],
              #'n_estimators': [400, 1200, 3000],
              #'max_depth':[3,5],
               #'alpha': [0.01, 0.25, 1],
              # 'min_samples_split':[2,3,4,5, 6],
                'min_samples_leaf': [1,2,3,4,5],
               }]

#parameters = [{'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
   #           'max_iter':[100, 300, 1000, 3000, 9000],
    #           'alpha': [0.0001, 0.0005, 0.003, 0.02, 0.1, 1, 2]
    #           }]




#parameters = [{'l1_ratio': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
  #            'max_iter':[100, 300, 1000, 3000, 9000],
   #            'alpha': [0.0001, 0.0005, 0.003, 0.02, 0.1, 1, 2],
    #           'tol': [0.00001, 0.0001, 0.001, 0.01, 0.1]
     #          }]



grid_search = GridSearchCV(estimator = reg, cv = 5, scoring = 'neg_mean_squared_error', 
  param_grid = parameters)


grid_search.fit(XX, yy)
best_accuracy = grid_search.best_score_
best_param = grid_search.best_params_
print(best_accuracy)
best_param
