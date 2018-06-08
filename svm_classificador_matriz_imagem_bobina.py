__author__ = "Luiz Gustavo Silveira Rossa"
__version__ = "1.5"

'''
Script responsavel por comparar uma nova matriz inserida no sistema, com a base de dados treinada anteriormente.
'''

#Exemplo:
#No cmd do windows
#C:\Users\administrator.AUTOMATIONPR\PycharmProjects\script_classificador_teste_3  python script_classificador_teste_3.py Vetor1_normal.csv


#Resultado da classificacao
#1 - SEM VINCO
#2 - COM VINCO

import pickle
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support

from sklearn import svm
from sklearn import datasets
from matplotlib import style
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score


# Parametros de configuracao do script para classificar a matriz
labels = None
pos_label = 1
average = 'binary'
sample_weight = None
target_names = None
target_values = None
sample_weight = None


def acuracia(clf,X,y):
   resultados = cross_val_predict(clf, X, y, cv=5)
   return accuracy_score(y,resultados)

# le o nome do arquivo informado pelo usuario
# for aux in sys.argv:
#    file = aux

# Recebe o arquivo cadastrado pelo usuario
# data = open(file, 'rt')
# vector = np.loadtxt(data, delimiter=",")

file1 = 'imagens_classificacao/sobelX10639.txt'
data1 = open(file1, 'rt')
vector = np.loadtxt(data1, delimiter=",")


# carrega o script treinado
classifier = pickle.load(open('svm_treinado_matriz.sav', 'rb'))

#Compara vetor por vetor do classificador
# PASSA PARA O SCRIPT OS DADOS DO ARQUIVO ENVIADO PELO USUARIO PARA SER UTILIZADO COMO VETOR DE CLASSIFICACAO
dataTest = [vector]
#VETORES UTILIZADOS NO TREINAMENTO
target = [1,2,3]
# 1 - SEM VINCO
# 2 - COM VINCO
# 3 - MESA

test = classifier.predict(dataTest)


#analise geral
samples = [vector, vector, vector]
analyse = classifier.predict(samples)


print("----------------------")
print("VETORES DE TREINAMENTO")
print("vetor 1 - Conjunto de dados sem vinco")
print("vetor 2 - Conjunto de dados com vinco")
print("vetor 3 - Conjunto de dados da mesa da linha")
print("----------------------")

print("Classification report for classifier %s:\n%s\n"
    % (classifier, metrics.classification_report(target, analyse)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(target, analyse))
# Percorre cada um dos elementos do target e compara com o script treinado.
# Com isso e possivel determinar a precissao da classificacao para cada um dos targets treinados.
try:
    precision_list = []
    id_target = []
    max_value = []

    for i in target:
        aux = [i]
        print("Target {}".format(aux))
        print("\n")
        print("Classification report for classifier %s:\n%s\n"
              % (classifier, metrics.classification_report(aux, test)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(aux, test))

        p, r, f1, s = precision_recall_fscore_support(aux, test,
                                                      labels=labels,
                                                      average=None,
                                                      sample_weight=sample_weight)

        #print(acuracia(classifier,dataTest,aux))

        # Captura a precissao de acerto para cada item classificado
        precision = np.average(p, weights=s)
        p = "{0:0.{1}f}".format(precision, 2)
        print("Precision: {}".format(p))
        # id_target.append(i)
        precision_list.append(p)
        # precision_list.insert(i, p)
        print("===============")

    for id, value in enumerate(precision_list):
        # print("List: [%d] = %d" %(id, value))
        print(id, value)
        # Percorre a lista de precissao e captura o maior valor
        max_value = max(precision_list)
        # Captura o endereco onde o maior valor da lista esta armazenado
        id_value = precision_list.index(max_value)
        # print(max_value, id_value)
    print("***")

    # CALCULA A MEDIA DA PRECISSAO PARA O VETOR CLASSIFICADO
    semVinco = 0.00
    comVinco = 0.00
    mesa = 0.00
    #meanCritical = 0.00

    # CONVERTE A LISTA DE STRING PARA UMA LISTA FLOAT
    new_list = map(float, precision_list)
    #list_len = len(precision_list)

    count = 5

    # GRUPOS DE CLASSIFICACAO
    group1 = [0, 1, 2]

    # SEPARA OS ELEMENTOS DA LISTA EM TRES GRUPOS COM CINCO ELEMENTOS EM CADA
    # groups = [new_list[i:i + count] for i in range(0, len(new_list), count)]
    # print(groups)

    for aux4 in group1:
        semVinco = new_list.__getitem__(0)
        comVinco = new_list.__getitem__(1)
        mesa = new_list.__getitem__(2)
        #meanCritical = new_list.__getitem__(2)
    # print(meanNormal)
    # print(meanAlarm)
    # print(meanCritical)

    # Classificacao da amostra analisada.
    if semVinco >= 0.95:
        print("Matriz sem vinco")
    elif comVinco >= 0.95:
        print("Matriz com vinco")
    elif mesa >= 0.95:
        print("Imagem da mesa da linha")
    else:
        print("Nao consta na base de treino")
    #elif meanCritical >= 0.95:
    #    print("Critico")
except Exception:
    print("Problems to run classification !!!")
