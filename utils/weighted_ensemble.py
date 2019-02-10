import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import multioutput
from joblib import dump

pred1= pd.read_csv('data/prediction_train.csv')
pred2= pd.read_csv('data/rfcn_resnet101_final.csv')
X=pd.merge(pred1,pred2,left_on='image_name',right_on='image_name',how='left')
print(X.head())
X=X.drop(columns=['image_name'])

expected= pd.read_csv('training.csv')
expected= pd.merge(expected,pred1,left_on='image_name',right_on='image_name',how='right')
expected= expected.drop(columns=['image_name','x1_y','x2_y','y1_y','y2_y'])

X=X.values
Y=expected.values

xgb_model= xgb.XGBRegressor(silent=False)

ensemble_model = multioutput.MultiOutputRegressor(estimator=xgb_model)
ensemble_model.fit(X,Y)

dump(ensemble_model, 'ensemble_model.joblib.dat')

pred1= pd.read_csv('data/prediction.csv')
pred2= pd.read_csv('data/rfcn_resnet101_final_test.csv')
Jah=pred1
XXX=pred2

X=pd.merge(pred1,pred2,left_on='image_name',right_on='image_name',how='left')
names=X['image_name']
X=X.drop(columns=['image_name'])

predictions=ensemble_model.predict(X).astype(dtype=int)
pred= pd.DataFrame(predictions,columns=['x1','x2','y1','y2'])
Onfrey=XXX[~XXX['image_name'].isin(Jah['image_name'])]
#print(predictions.shape)
pred['image_name']=names.values
print(pred.shape)
print(Onfrey.shape)
final=pd.concat([pred,Onfrey],sort=True)
print(final.shape)
final.to_csv('final.csv', index=False)

final2=pd.concat([pred1,Onfrey],sort=True)

final2.to_csv('final2.csv',index=False)
