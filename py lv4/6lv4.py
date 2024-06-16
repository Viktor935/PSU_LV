import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

# ucitavanje podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())

# Definiranje kategoričkih varijabli
categorical_vars = ['fuel', 'seller_type', 'transmission', 'owner']

# One-hot kodiranje kategoričkih varijabli
df_encoded = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Odabir ulaznih varijabli
X = df_encoded.drop('selling_price', axis=1)
y = df_encoded['selling_price']

# Odabir numeričkih ulaznih varijabli
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
X_numeric = X[numeric_features]

# Podjela na train i test
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=300)

# Skaliranje numeričkih ulaznih varijabli
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Izrada modela
linear_model = LinearRegression()
linear_model.fit(X_train_s, y_train)

# Evaluacija modela
y_pred_train = linear_model.predict(X_train_s)
y_pred_test = linear_model.predict(X_test_s)

print("R2 test", r2_score(y_pred_test, y_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_pred_test, y_test)))
print("Max error test:", max_error(y_pred_test, y_test))
print("MAE test:", mean_absolute_error(y_pred_test, y_test))

# Plot rezultata na testnim podacima
fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_test, line_kws={'color': 'green'})
ax.set(xlabel='Predikcija', ylabel='Stvarna vrijednost', title='Rezultati na testnim podacima')
plt.show()

#rezultati se nisu značajno poboljšali