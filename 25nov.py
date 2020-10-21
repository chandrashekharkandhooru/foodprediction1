#!/usr/bin/env python
# coding: utf-8

# In[96]:


import numpy as np
import pandas as pd


# In[97]:


train=pd.read_excel('C:/Users/chandu prakash/Downloads/Data_train_new2.xlsx')
test=pd.read_excel('C:/Users/chandu prakash/Downloads/Data_test_new2.xlsx')


# In[98]:


test.head()


# In[99]:


train["Average_Cost"]= train["Average_Cost"].replace('for','s200')


# In[100]:


import re
import string
def clean_text_round2(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('%s' % re.escape(string.punctuation),'',text)
    text=re.sub(r'\(.*\)', '',text)
    
    text=re.sub(':%s' % re.escape(string.punctuation),'',text)
    text=re.sub("\D", "", text)
    
    return text
round2=lambda x: clean_text_round2(x)
train['Average_Cost']=train.Average_Cost.apply(round2)
train['Minimum_Order']=train.Minimum_Order.apply(round2)
test['Average_Cost']=test.Average_Cost.apply(round2)
test['Minimum_Order']=test.Minimum_Order.apply(round2)


# In[101]:


train['Cuisines'] = train['Cuisines'].str.count(',') + 1
test['Cuisines'] = test['Cuisines'].str.count(',') + 1


# In[102]:


train["Minimum_Order"]=train.Minimum_Order.astype(int)
train["Average_Cost"]=train.Average_Cost.astype(int)
test["Minimum_Order"]=test.Minimum_Order.astype(int)
test["Average_Cost"]=test.Average_Cost.astype(int)


# In[103]:


train["Rating"]=train.Rating.astype(float)
train["Votes"]=train.Votes.astype(int)
train["Reviews"]=train.Reviews.astype(int)
test["Rating"]=test.Rating.astype(float)
test["Votes"]=test.Votes.astype(int)
test["Reviews"]=test.Reviews.astype(int)


# In[104]:


location=pd.get_dummies(train.Location,drop_first=True)


# In[105]:


location1=pd.get_dummies(test.Location,drop_first=True)


# In[106]:


train['vr'] = train.Votes + train.Reviews


# In[107]:


test['vr'] = test.Votes + test.Reviews


# In[108]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Delivery_Time= le.fit_transform(train.Delivery_Time)


# In[109]:


final=pd.DataFrame(train,columns=['Cuisines','Minimum_Order','Average_Cost','Votes','Reviews','Rating'])
#apply scaling
from sklearn.preprocessing import StandardScaler
SS=StandardScaler(with_mean=True,with_std=True)
finall=SS.fit_transform(final)
final2=pd.DataFrame(finall,columns=final.columns[:])
#X_train_cont2[0:10]
final2.head()


# In[110]:


final1=pd.DataFrame(test,columns=['Cuisines','Minimum_Order','Average_Cost','Votes','Reviews','Rating'])
#apply scaling
from sklearn.preprocessing import StandardScaler
SS=StandardScaler(with_mean=True,with_std=True)
fina=SS.fit_transform(final1)
final3=pd.DataFrame(fina,columns=final1.columns[:])
#X_train_cont2[0:10]
final3.head()
testfile=pd.concat([final3,location1],axis=1,join='outer')
testfile1=pd.concat([final1,location1],axis=1,join='outer')


# In[228]:


finaln=pd.concat([final2,location],axis=1,join='outer')
x=finaln
y=pd.DataFrame(train,columns=['Delivery_Time'])


# In[266]:


Notscale=pd.concat([final,location],axis=1,join='outer')
X1=Notscale
Y1=pd.DataFrame(train,columns=['Delivery_Time'])


# In[265]:


import warnings
warnings.filterwarnings("ignore")
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
model_SVM = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.3,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
from sklearn.tree import DecisionTreeClassifier
bag_clf=BaggingClassifier(DecisionTreeClassifier(max_depth=5,class_weight=None),n_estimators=500,max_samples=100,
                         bootstrap=True,n_jobs=-1)
from sklearn import model_selection
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='brute', leaf_size= 1, n_neighbors= 9, weights ='uniform')
model = XGBClassifier()
from sklearn.neural_network import MLPClassifier
ann = MLPClassifier()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=500,algorithm="SAMME.R",learning_rate=0.5)
X = finaln
Y = Y1
num_instances = len(X)
seed = 8
kfold = model_selection.KFold(n_splits=10, random_state=seed)
results = model_selection.cross_val_score(model_SVM, X, Y, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[267]:


model_SVM.fit(X,Y)


# In[268]:


ann1=model_SVM.predict(testfile)
an1=pd.DataFrame(ann1,columns=['Delivery_Time'])
np.unique(an1.Delivery_Time,return_counts=True)


# In[269]:


an1.to_excel("26nov11svm1111.xlsx")


# In[209]:


# splitting the data into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y , test_size = 0.3,random_state=0)


# In[223]:


from sklearn.svm import SVC
model_SVM = SVC()
model_SVM.fit(X_train1,y_train1)


# In[224]:


pred_svm = model_SVM.predict(X_test1)
print(np.unique(pred_svm,return_counts=True))
acc3=accuracy_score(y_test1,pred_svm)
acc3


# In[210]:


params_knn = {'n_neighbors':[5,6,7,8,9,10,12],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute']}


# In[211]:


model_knn_GS = GridSearchCV(KNeighborsClassifier(), param_grid=params_knn)
model_knn_GS.fit(X_train,y_train)


# In[213]:


model_knn_GS.best_params_


# In[198]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)


# In[212]:


y_pred6=model_knn_GS.predict(X_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred6)
acc


# In[200]:


ns=knn.predict(testfile)
ns1=pd.DataFrame(ns,columns=['Delivery_Time'])
np.unique(ns1.Delivery_Time,return_counts=True)


# In[187]:


ns1.to_excel("knnn3.xlsx")


# In[231]:


# splitting the data into Train and Test
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,Y1 , test_size = 0.3,random_state=0)


# In[232]:


from sklearn.svm import SVC
model_SVM = SVC()
model_SVM.fit(X_train1,y_train1)


# In[233]:


pred_svm = model_SVM.predict(X_test1)
print(np.unique(pred_svm,return_counts=True))
acc3=accuracy_score(y_test1,pred_svm)
acc3


# In[236]:


result4=model_SVM.predict(testfile1)
np.unique(result4,return_counts=True)


# In[237]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf=DecisionTreeClassifier(max_depth=5,max_features=8)
ada_clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=500,algorithm="SAMME.R",learning_rate=0.5)


# In[169]:


params = {'max_features': ['auto', 'sqrt', 'log2'],
       'min_samples_split': [2,3,4,5,6,7,8,9,10],        
       'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
         'max_depth':[2,3,4,5,6,7,8,9]}


# In[170]:


from sklearn.model_selection import GridSearchCV
clf1 = GridSearchCV(clf, param_grid=params)
clf1.fit(X_train1,y_train1)


# In[173]:


modelF = clf1.best_estimator_
modelF


# In[174]:


modelF.fit(X_train1, y_train1)


# In[175]:


y_pred6=modelF.predict(X_test1)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test1,y_pred6)
acc


# In[176]:


ns=modelF.predict(testfile1)
ns1=pd.DataFrame(ns,columns=['Delivery_Time'])
np.unique(ns1.Delivery_Time,return_counts=True)


# In[238]:


ada_clf.fit(X_train1, y_train1)


# In[239]:


y_pred6=ada_clf.predict(X_test1)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test1,y_pred6)
acc


# In[240]:


result1=ada_clf.predict(testfile1)
np.unique(result1,return_counts=True)


# In[126]:


result1=pd.DataFrame(result1,columns=["Delivery_Time"])


# In[127]:


result1.to_excel("adaboost2.xlsx")


# In[241]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train1, y_train1)


# In[242]:


y_pred3=model.predict(X_test1)
np.unique(y_pred3,return_counts=True)


# In[243]:


from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test1,y_pred3)
acc


# In[244]:


result1=model.predict(testfile1)
np.unique(result1,return_counts=True)


# In[245]:


result1=pd.DataFrame(result1,columns=["Delivery_Time"])
result1.head()


# In[246]:


result1.to_excel("xgb1.xlsx")


# In[247]:


from sklearn.ensemble import RandomForestClassifier
rnd_clf=RandomForestClassifier(n_estimators=400,max_leaf_nodes=16,n_jobs=-1)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,max_samples=100,
                         bootstrap=True,n_jobs=-1)


# In[248]:


bag_clf.fit(X_train1, y_train1)
rnd_clf.fit(X_train1, y_train1)


# In[249]:


y_pred1= bag_clf.predict(X_test1)
y_pred2= rnd_clf.predict(X_test1)
np.unique(y_pred1,return_counts=True)


# In[250]:


print ("Accuracy : ", accuracy_score(y_test1, y_pred1)) 
print ("Accuracy : ", accuracy_score(y_test1, y_pred2))


# In[251]:


result2=bag_clf.predict(testfile1)
np.unique(result2,return_counts=True)


# In[252]:


result3=rnd_clf.predict(testfile1)
np.unique(result3,return_counts=True)


# In[139]:


np.unique(y_pred2,return_counts=True)


# In[140]:


print ("Accuracy : ", accuracy_score(y_test, y_pred1)) 
print ("Accuracy : ", accuracy_score(y_test, y_pred2))


# In[141]:


from sklearn.svm import SVC


# In[142]:


model_SVM = SVC()
model_SVM.fit(X_train,y_train)


# In[143]:


result4=model_SVM.predict(testfile)
np.unique(result4,return_counts=True)


# In[144]:


result4=pd.DataFrame(result4,columns=["Delivery_Time"])
result4.head()


# In[145]:


result4.to_excel("svm1.xlsx")


# In[146]:


pred_svm = model_SVM.predict(X_test)
print(np.unique(pred_svm,return_counts=True))
acc3=accuracy_score(y_test,pred_svm)
acc3


# In[ ]:




