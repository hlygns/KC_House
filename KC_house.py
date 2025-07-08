#King Country House Price Prediction
#kütüphanleri yükle -import et
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor

df=pd.read_csv('kc_house_data.csv')
print(df.head())
print(df.info())
print("en düşük ev fiyatı: ",df["price"].min())
print("en yüksek ev fiyatı: ",df["price"].max())
print("ortalama ev fiyatı: ",df["price"].mean())
#en yüksek fiyatlı evin özellikleri
print(df[df['price']==df['price'].max()])

print(df[df["price"] == df["price"].max()][["lat", "long", "price"]])
#veri setindeki aykırı değerleri ayıklamak outlier
#outlier hesaplanacak sütunları belirledik
df_outlier_col=df[['price','bedrooms','sqft_living','sqft_lot']]
outliers=df_outlier_col.quantile(q=.99)
print(outliers)

df_clean=df[df['price']<outliers['price']]
print(df_clean['price'].max())

df_clean=df_clean[df_clean['bedrooms']<outliers['bedrooms']]
df_clean=df_clean[df_clean['sqft_living']<outliers['sqft_living']]
df_clean=df_clean[df_clean['sqft_lot']<outliers['sqft_lot']]

df_clean.describe()
df_corr=df_clean.corr(numeric_only=True).sort_values('price',ascending=False)['price'].head(10)
print(df_corr)


df_clean['Age']=2015-df_clean['yr_built']
df_clean['restore']=np.where(df_clean['yr_renovated']==0,0,1)
df_clean['sqft_total_size']=df_clean['sqft_living']+df_clean['sqft_lot']
df_clean['sqft_basement']=np.where(df_clean['sqft_basement']==0,0,1)

print(df_clean.shape)

X=df_clean.drop(['price','date','id','yr_built','yr_renovated','lat','long','sqft_living','sqft_lot'],axis=1)

y=df_clean['price']
print(X.shape)
print(y.shape)
print(X.info())

#eksik veri yok sadece zipcode onehot encoding ile sayısal değerlere çevirildi
X=pd.get_dummies(X,columns=['zipcode'],drop_first=True)
print(X.info())

scaler=StandardScaler()
X=scaler.fit_transform(X)#hem uyduruyo hem öğreniyo hem de dönüştürüyo
print(X)

#veri setini bölelim
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#modeli oluştur

print('Model 1')
model=LinearRegression()

#model eğitim

model.fit(X_train,y_train)
#model test
y_pred=model.predict(X_test)

#model değerlendirme
print(" r2 score: ",r2_score(y_test,y_pred))
print(" mse: ",mean_squared_error(y_test,y_pred)**.5)

#-----------------------------------------------
print('Model 2')
model=RandomForestRegressor()

model.fit(X_train,y_train)
#model test
y_pred=model.predict(X_test)

#model değerlendirme
print(" r2 score: ",r2_score(y_test,y_pred))
print(" mse: ",mean_squared_error(y_test,y_pred)**.5)
#--------------------------------------------------
#model 3
print('Model 3')
model=

model.fit(X_train,y_train) #??
#model test
y_pred=model.predict(X_test)

#model değerlendirme
print(" r2 score: ",r2_score(y_test,y_pred))
print(" mse: ",mean_squared_error(y_test,y_pred)**.5)