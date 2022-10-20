import pandas as pd
from sklearn.model_selection import train_test_split

data_all= pd.read_csv('csv/label.csv')

X=data_all.drop('class', axis=1)
y=data_all['class']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=1010)

y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

data_train=pd.merge(X_train,y_train,left_index=True,right_index=True)
data_test=pd.merge(X_test,y_test,left_index=True,right_index=True)

data_train.to_csv('csv/train.csv')
data_test.to_csv('csv/test.csv')