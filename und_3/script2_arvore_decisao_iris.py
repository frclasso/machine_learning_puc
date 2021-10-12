""""
Arvore de decisão
Ilustra o funcionamento dol algoritmo de árvore de decisão com atributos numéricos. 
Prof Hugo de Paula - PUC-MG
"""

"""
Base de dados : iris dataset (espécies de lírios)

https://archive.ics.uci.edu/ml/datasets/Iris/ 
3 classes (setosa, virgínica, versicolor) 
50 amostras 
4 atributos reais positivos (comp. pétala, comp. sépala, larg. pétala, larg. sépala)
"""

# pacotes extras para visualização
# pip3 install pydotplus
# pip3 install dtreeviz

import pandas as pd
from sklearn import datasets, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dtreeviz.trees import *
import pydotplus
from IPython.display import Image

np.set_printoptions(threshold=None, precision=2)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('precision', 2)


"""
Carga dos dados e particionamento das bases de treinamento e teste
trains_tes_split(x,y) -- particiona a base de dados original em bases de treinamento e teste. 
Por padrão, 75% da base é utilizada para treinamento e 25% para testes. 
No código a seguir , são utilizados 85% para treinamento e 15% para testes.
"""


print("=" * 40 + "importando a base de dados iris" + "=" * 40)
# importando a base de dados iris
iris = datasets.load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

# Particionando a base de dados
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.15)


iris_dataframe = pd.DataFrame(np.c_[iris['data'], iris['target']],
                              columns=np.append(iris['feature_names'], 'target'))

ax2 = pd.plotting.scatter_matrix(iris_dataframe.iloc[:,:4], figsize=(11,11), c=y, marker='o',

                                hist_kwds={'bins':20}, s=60, alpha=.8)
# Gerando graficos 
plt.figure()
ax3 = pd.plotting.parallel_coordinates(iris_dataframe, "target")
# plt.show()

print()

"""
Indução do modelo
-------------------
Os 3 passsos para indução de um modelo são:

    Instanciar o modelo: DecisionTreeClassifier()
    Treinar o modelo fit()
    Testar o modelo: predict()
"""

print("=" * 40 + "Indução do modelo" + "=" * 40)
tree_iris = DecisionTreeClassifier(random_state=0, criterion='entropy', class_weight={0:1, 1:1})
tree_iris = tree_iris.fit(X_train, y_train)
print("Acurácia (base de treinamento): {}".format(tree_iris.score(X_train, y_train)))

y_pred = tree_iris.predict(X_test)
print("Acurácia de previsão: {}".format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_table = pd.DataFrame(data=cnf_matrix, index=iris.target_names, columns=[x + "(prev)" for x in iris.target_names])
print(cnf_table)
print()

print("=" * 40 + "Exibição da árvore de decisão" + "=" * 40)


viz = dtreeviz(tree_iris,
               X_train,
               y_train,
               target_name="especie",
               feature_names=iris.feature_names,
               class_names=['setosa', 'versicolor', 'virginica'])
viz.view()


dot_data = tree.export_graphviz(tree_iris, out_file=None, 
                                rounded=True,
                                filled=True,
                                feature_names=iris.feature_names,
                                class_names=['setosa', 'versicolor','virginica'])

# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)

#Show graph
Image(graph.create_png())





"""
UCI Datasets
https://archive-beta.ics.uci.edu/
"""

