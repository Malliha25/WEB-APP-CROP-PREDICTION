import imp
import streamlit as st
st.header('Crop prediction System')
import pandas as pd
dataset=pd.read_csv('crop.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
cls=KNeighborsClassifier()
cls.fit(X_train,Y_train)
Y_pred=cls.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_pred,Y_test))
N=st.text_input('quantity of nitrogen')
P=st.text_input('quantity of phosphorous')
K=st.text_input('quantity of potassium')
T=st.text_input('enter temparature')
H=st.text_input('enter Humidity')
Ph=st.text_input('enter ph')
R=st.text_input('enter amount of rainfall')
if st.button('predict crop'):
    crop=cls.predict([[N,P,K,T,H,Ph,R]])[0]
    st.success(crop)
