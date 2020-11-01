# positive skewness of the price, we make the normal distribution for the sale price.
trainF['SalePrice'] = train['SalePrice'].map(lambda x:np.log(x))

# missing values
trainImpFirst = trainF.fillna(trainF.median())
trainImp = trainImpFirst.apply(lambda x: x.fillna(x.value_counts().index[0]))

# dummy variable
trainImpd = pd.get_dummies(trainImp)

train_copy = trainImpd.copy()
#train_set = train_copy.sample()

X_train =  train_copy.drop(columns=['SalePrice'])
y_train = train_copy['SalePrice']

# feature reduction
from xgboost import XGBRegressor, plot_importance

model = XGBRegressor()
model.fit(X_train, y_train)
# plot feature importance
plot_importance(model)

#print(model.feature_importances_)
#print(len(model.feature_importances_))

cc = X_train.columns
#print(len(cc))
dd = []
for i in range(len(cc)):
  dd.append(cc[i])
#print(len(dd))
#dd

aa = []
bb = model.feature_importances_
for i in range(len(bb)):
    #print(i)
    argmin = np.argmin(bb)
    #print(argmin)
    aa.append(dd[argmin])
    bb = np.delete(bb, [argmin])
    del dd[argmin]
#print(aa)

X_train_sorted = X_train[aa]

#X_train_sorted
#X_train_sorted['SalePrice'] = y_train



#train_copy = X_train_sorted .copy()
#t_set = train_copy.sample(frac=0.9)
#v_set = train_copy.drop(t_set.index)

#XTrain =  t_set.drop(columns=['SalePrice'])
#yTrain = t_set['SalePrice']

#XValid =  v_set.drop(columns=['SalePrice'])
#yValid = v_set['SalePrice']

X_train_reduction = X_train_sorted.drop(X_train_sorted.columns[0:200], axis=1)
X_train_reduction['SalePrice'] = y_train

# We add some features in the test set because of the process of taking care of the dummy variable
X_test = X_test.drop(columns = ['MSSubClass_t12'])
X_test['Exterior1st_Stone'] = 0
X_test['Exterior1st_ImStucc'] = 0
X_test['RoofMatl_Membran'] = 0
X_test['RoofMatl_ClyTile'] = 0
X_test['TotalBsmtSFn_c7'] = 0
X_test['RoofMatl_Roll'] = 0
X_test['RoofMatl_Metal'] = 0
X_test['Electrical_Mix'] = 0

# preparing a test set based on the number of features obtained from feature reduction
X_test_sorted = X_test[aa]
X_test_reduction = X_test_sorted.drop(X_test_sorted.columns[0:200], axis=1)
