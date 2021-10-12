import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import os

print(os.getcwd())

file_path = "C:/Users/fabio.classo/Downloads/fabio/estudos/PUC/Disciplinas/10_Machine_Learning/scripts_aula"\
    "/Datasets/"
os.chdir(file_path)
sonar = pd.read_excel("./sonar.xlsx")
print(sonar.head())
print()
print("Dimensões: {}".format(sonar.shape))
print("\nCampos: {}".format(sonar.keys()))
print('='*160)
print(sonar.describe(), sep='\n')
print()

X_train = sonar.iloc[:,0:(sonar.shape[1] - 1)]
#print(X_train)
le = LabelEncoder()
Y_train = le.fit_transform(sonar.iloc[:,(sonar.shape[1] - 1 )])
#print(Y_train) 0 e 1



# Calculando a Acuracia
# sonar_tree = DecisionTreeClassifier(random_state=0)
# sonar_tree = sonar_tree.fit(X_train, Y_train)
# print("Acuracia (score): {}".format(sonar_tree.score(X_train, Y_train)))
# print()

# Acurácia de previsão
# print(classification_report(Y_train, Train_predict))
# Train_predict = sonar_tree.predict(X_train)
# print("Acurácia de previsão: {}".format(accuracy_score(Y_train, Train_predict)))

# Salvando em um arquivo .dot (grafo)
def save_file():
    """Salvando em um arquivo .dot (grafo)"""
    file_path = "C:/Users/fabio.classo/Downloads/fabio/estudos/PUC/Disciplinas/10_Machine_Learning/Scripts/und_3"
    os.chdir(file_path)
    with open("sonar.dot", "w") as file:
        file= tree.export_graphviz(sonar_tree, out_file=file)

#save_file()

# convertentdo .dot em pdf
#dot -Tpdf 'sonar.dot' -o sonar.pdf

print()
# Particionando a base
[Xtreinamento, Xteste, Ytreinamento, Yteste] = train_test_split(X_train, Y_train, random_state=0)

# Calculando a Acuracia
sonar_tree = DecisionTreeClassifier(random_state=0)
sonar_tree = sonar_tree.fit(Xtreinamento, Ytreinamento)
print("Acuracia (score): {}".format(sonar_tree.score(Xtreinamento, Ytreinamento)))
print()

# Acurácia de previsão
Train_predict = sonar_tree.predict(Xteste)
print("Acurácia de previsão: {}".format(accuracy_score(Yteste, Train_predict)))
print(classification_report(Yteste, Train_predict))
