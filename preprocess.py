import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')



df=pd.read_csv("dataset/SDN_Intrusion.csv")

# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.head())

for i in df.select_dtypes(include='number').columns.values:
    df[i]=df[i].fillna(df[i].mean())

lab=LabelEncoder()
for i in df.select_dtypes(include='object').columns.values:
    df[i]=lab.fit_transform(df[i])

# print(dict(zip(lab.classes, lab.transform(lab.classes))))
# {'BENIGN': 0, 'DDoS': 1, 'Web Attack � Brute Force': 2, 'Web Attack � Sql Injection': 3, 'Web Attack � XSS': 4}

X=[]
for i in df.columns.values:
    df['z-scores']=(df[i]-df[i].mean())/(df[i].std())
    outliers=np.abs((df['z-scores']>3))
    if outliers.sum() >0.2:
        X.append(i)
    
X=[]
for i in df.columns.values:
    df['z-scores']=(df[i]-df[i].mean())/(df[i].std())
    outliers=np.abs((df['z-scores']>3))
    if outliers.sum() >0:
        X.append(i)

thresh=2.5
for i in df[X[:38]].columns.values:
    upper=df[i].mean()+thresh * df[i].std()
    lower=df[i].mean()-thresh * df[i].std()
    df=df[(df[i]<upper)&(df[i]>lower)]

x=[]
corr=df.corr()['Class']
corr=corr.drop(['z-scores','Class'])
for i in corr.index:
    if corr[i]>0.2 and corr[i] <0.65:
        # print(corr[i])
        x.append(i)

# print(x)

# plt.figure(figsize=(17, 6))
# corr = df.corr(method='spearman')
# my_m = np.triu(corr)
# sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
# plt.show()

# correlation_matrix = df.corr()
# sn.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# plt.show()

x=[]
corr=df.corr()['Class']
corr=corr.drop(['z-scores','Class'])
for i in corr.index:
    if corr[i]>0.2 and corr[i] <0.8:
        # print(corr[i])
        x.append(i)

x=df[x]
y=df['Class']
# x_train,x_test,y_train,y_test=train_test_split(x,y)

# lr = LogisticRegression(max_iter=35)
# lr.fit(x_train, y_train)
# print('The logistic regression: ', lr.score(x_test, y_test))

# tree = DecisionTreeClassifier(criterion='gini', max_depth=1)
# tree.fit(x_train, y_train)
# print('Dtree ', tree.score(x_test,y_test))

# rforest = RandomForestClassifier(criterion='gini',max_depth=1)
# rforest.fit(x_train, y_train)
# print('The random forest: ', rforest.score(x_test, y_test))

Y=pd.get_dummies(y)
x_tran,x_tst,y_tran,y_tst=train_test_split(x,Y)