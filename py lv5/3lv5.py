import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# ucitaj podatke za ucenje
df = pd.read_csv('occupancy_processed.csv')

feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].values
y = df[target_name].values

# Podjela na setove za učenje i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Kreiraj i istreniraj DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Predikcija modela
y_pred = dt.predict(X_test)

# Evaluiraj klasifikaciju
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average=None))
print("Recall:", recall_score(y_test, y_pred, average=None))

# Vizualizacija
plt.figure(figsize=(12, 8))
plot_tree(dt, filled=True, feature_names=feature_names, class_names=class_names)
plt.show()

#Promijenom parametar max-depth stabla odlučivanja mijenjamo kompleksnost stabla. Povećanjem možemo doći do kompleksnijeg stabla. Smanjenjem možemo doći do jednostavnijeg.
#Ako ne koristimo skaliranje ulaza, stablo može imati problem u pronalaženju optimalnih granica.