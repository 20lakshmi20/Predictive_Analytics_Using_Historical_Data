import pandas as pd

data = pd.read_excel("Historical_Sales_Data_120_Months.xlsx")
print(data.head())

X = data[['Month']]
y = data['Sales']

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

future_months = pd.DataFrame({
'Month':[121,122,123,124,125,126]
})

predictions = model.predict(future_months)

print("Future Sales Forecast")

for month,sale in zip(future_months['Month'],predictions):
    print(f"Month {month}: ₹{sale:.2f}")

from sklearn.metrics import r2_score

predicted = model.predict(X)

accuracy = r2_score(y,predicted)

print("Accuracy:",accuracy)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

plt.scatter(X,y,label="Historical Data")

plt.plot(X,
model.predict(X),
color='red',
label='Regression Line')

plt.scatter(future_months,
predictions,
color='green',
label='Forecast')

plt.xlabel("Month")

plt.ylabel("Sales")

plt.title("Sales Forecast Using Historical Data")

plt.legend()

plt.show()