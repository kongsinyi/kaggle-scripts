import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import sklearn.preprocessing as pp
from sklearn.cross_validation import train_test_split as tts


#convert labels in both train and test sets
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
merged = pd.concat([train,test])
merged.Sex = merged.Sex.map({'male' : 1, 'female' :0})
merged['embarked_num'] = merged.Embarked.map({'S':0, 'C' :1, 'Q':2})


m_Age = merged[merged.Sex == 1]['Age'].median()
f_Age = merged[merged.Sex==0]['Age'].median()
merged['age_fill'] = merged['Age']
merged.loc[merged.Age.isnull(),'age_fill'] = 27.5

#scale and fill NaN with mean
cols_to_scale = ['Fare','Pclass','Sex','age_fill','embarked_num','Parch','SibSp']
merged[cols_to_scale] = merged[cols_to_scale].fillna(merged[cols_to_scale].mean())

for i in range(len(cols_to_scale)):
    merged[[cols_to_scale[i]]] = pp.scale(merged[[cols_to_scale[i]]])

train = merged[:len(train)]
test = merged[len(train):]

#modeling with logit
xtrain,xval, ytrain,yval= tts(np.array(train[cols_to_scale]), np.ravel(train['Survived']))
LR = lm.LogisticRegression()
model = LR.fit(xtrain, ytrain)
score = model.score(xval,yval)
print('validation score: ',score)

xtest = np.array(test[cols_to_scale])
results = pd.DataFrame([test['PassengerId'], model.predict(xtest)], index = None).transpose()
results =results.rename(columns = {'Unnamed 0' : 'Survived'})

with open('./Submission.csv','w') as wfile:
    results.to_csv(wfile, index = False)
    wfile.close()


                             



