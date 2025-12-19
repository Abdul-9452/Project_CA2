
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")
df = pd.read_csv("INT234Dataset.csv")
df.head()
df.describe()
df.shape
df.drop_duplicates()
missing_values = df.isnull().sum().sort_values()
print(missing_values)
sing_values.plot(kind='bar')
plt.show()
df= df.dropna()
issming_values = df.isnull().sum().sort_values()
print(missing_values)
cols = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
for i in cols:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR

    df = df[(df[i]>= LB) & (df[i]<=UB)]
from sklearn.preprocessing import MinMaxScaler, StandardScaler
num_cols = ['pollutant_min', 'pollutant_max', 'pollutant_avg']
# MinMiax Scaler
df_MinMax= df.copy()
MM = MinMaxScaler()
df_MinMax[num_cols] = MM.fit_transform(df[num_cols])
print(df_MinMax)
# Standard Scaler
SC = StandardScaler()
df_SC = df.copy()
df_SC[num_cols] = SC.fit_transform(df[num_cols])
print(df_SC)

from sklearn.preprocessing import LabelEncoder
label_cols = ['country', 'state', 'city', 'station', 'pollutant_id']

le = LabelEncoder()
for col in label_cols:
    df[col + "_label"] = le.fit_transform(df[col].astype(str))
df_ohe = pd.get_dummies(df, 
                        columns=['country', 'pollutant_id'],                         drop_first=True)
df_ohe.head()
df_reg = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']]
df_reg.head()

df_reg = df_reg.dropna()
X = df_reg[['pollutant_min','pollutant_max']]
y = df_reg['pollutant_avg']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:", model.intercept_)
print("Coefficiants:", model.coef_)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE: ",mse)
print("R2 Score: ",r2)
plt.figure(figsize = (6,4))
sns.scatterplot(x = y_test, y = y_pred)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], '-',color = 'red', linewidth=2)
plt.xlabel = ("Actual pollutant_avg")
plt.ylabel("Predicted pollutant_avg")
plt.title("Actual vs Predicted Values")
plt.show()
