import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

#print(os.getcwd())

file_path = "C:/Users/fabio.classo/Downloads/fabio/estudos/PUC/Disciplinas/" \
    "10_Machine_Learning/scripts_aula/Datasets/"
os.chdir(file_path)

sonar = pd.read_excel("./sonar.xlsx", sheet_name=0)
print("\nDimensões: {}".format(sonar.shape))
print("="*160)
print("\nCampos: {}".format(sonar.keys()))
print("="*160)
print("Sonar Head():")
print(sonar.head())
print("="*160)
print("\nSonar describe:")
print(sonar.describe(), sep='\n')
print("="*160)

X = sonar.iloc[:,0:(sonar.shape[1] - 1)]
le = LabelEncoder()
y = le.fit_transform(sonar.iloc[:,(sonar.shape[1] - 1)])

class_names = ['Rocha', 'Mina']

X_train, y_train, X_test, y_test = train_test_split(X, y, random_state=0)

sonar_tree = DecisionTreeClassifier(random_state=0, criterion='entropy')

params = {
    'min_inpurity_decrease':[0.01, 0.05, 0.1], 
    'min_samples_leaf':[1, 2,3]
    }
sonar_grid = GridSearchCV(sonar_tree, param_grid=params)

y_pred = sonar_grid.fit(X_train, y_train).predict(X_test)
print("\nClassificando usando arvore de decisao com GrdSearch")
print("\nOs melhores parametros econtrados pelo GrdSearch")
print(sonar_grid.best_params_, '\n')

print(classification_report(y_test, y_pred, target_names=class_names))

# Calculando a matriz de confusão
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)
