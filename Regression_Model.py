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
