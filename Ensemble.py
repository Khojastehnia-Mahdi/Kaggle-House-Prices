# predicting
dataTest = [pd.DataFrame for i in range(splitN)]
predictAll = np.zeros((splitN, y1_test.shape[1]))

for i in range(y1_test.shape[0]):
  data = {'y1':  y1_test[i,:], 'y2':  y2_test[i,:],
          'y3':  y3_test[i,:], 'y4':  y4_test[i,:], 'y5':  y5_test[i,:]}
  new_test_X_set = pd.DataFrame(data, columns = ['y1', 'y2',
                                                 'y3', 'y4', 'y5'])
  dataTest[i] = new_test_X_set

  predictAll[i,:] = reg_meta[i].predict(new_test_X_set)

# Ensemble method (weighted mean)
meanOfXGB = (dataTest[0]['y1']+ dataTest[1]['y1']+ dataTest[2]['y1']+ dataTest[3]['y1'])/4
meanOfregKernelRidge = (dataTest[0]['y2']+ dataTest[1]['y2']+ dataTest[2]['y2']+ dataTest[3]['y2'])/4
meanOfregBayesianRidge = (dataTest[0]['y3']+ dataTest[1]['y3']+ dataTest[2]['y3']+ dataTest[3]['y3'])/4
meanOfrefGBR = (dataTest[0]['y4']+ dataTest[1]['y4']+ dataTest[2]['y4']+ dataTest[3]['y4'])/4
meanOfregLasso = (dataTest[0]['y5']+ dataTest[1]['y5']+ dataTest[2]['y5']+ dataTest[3]['y5'])/4

meanOfXGB

predictions11 = np.mean(predictAll, axis = 0)
print(predictions11.shape)
predictions1 = (3*meanOfXGB + meanOfregKernelRidge + meanOfregBayesianRidge + meanOfrefGBR + meanOfregLasso
                + 6*predictions11)/13
predictions1 

predictions = np.exp(predictions1)
