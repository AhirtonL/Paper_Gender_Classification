# -*- coding: utf-8 -*-'''
'''
Created on 09/12/2016

@author: Rodrigo
'''


'''
Created on 05/09/2016

@author: Rodrigo
'''

import os
PATH_THIS = "home/ahirton/Python/gender_classification/"
#PATH_THIS = "home/rpasti/workspace/gender_classification/"
#os.chdir("..//..") #CHANGEDIR
PATH_SRC = "home/ahirton/Python/gender_classification/"
#PATH_SRC = "home/rpasti/workspace/gender_classification/"

import time
from file_utils import FileUtils

'''
IMPORT DOS COMPONENTES
'''

from meta_attributes import MetaAttributes
from text_processing import TextProcessing
import nltk
import csv
from file_search import FileSearch
import math
import numpy
import time

def main():
   
    root_path_config = PATH_THIS
    
    #file_utils = FileUtils(root_path_config)
    
    #dbConfig = file_utilsr.load_json('db_config')
   
    #connectionInfo = {'user_input':dbConfig['connection1']['user_input'], 'password_input':dbConfig['connection1']['password_input'],
     #                    'host_input':dbConfig['connection1']['host_input'], 'database_input':dbConfig['connection1']['database_input']}
   
    
    
    start = time.clock()
    metaAttTest = GenderClassification()
    
    metaAttTest.run()
    print("Tempo de execucao: " + str(time.clock() - start))
    
class GenderClassification():
    '''
    
    '''
    def __init__(self):
        '''
        Inicialização dos parametros
        '''
        self.loadCSV = False
        self.singleDataSet = True
        self.filterEntropy = True
        self.pSamplesEntropy = 0.8 #Porcentagem de amostras com maior entropia, para cada dataset e cada classe
        self.nSamplesTraining = 500 #OBS: Para cada classe
        #self.nSamplesTest = 10000 #Para cada classe
        self.method = "svm" #svm #naive_bayes #mlp #dt
        self.nExp = 1
        self.reverseDataSet = False
        self.categoriesMA = {"C":False, "W":False, "TS":False, "TM":True, "PC":False, "LIWC":False}   

    
    def run(self):
        '''
        
        '''
        if self.reverseDataSet:
            pathFilesF_set2 = "/home/ahirton/Python/gender_classification/tweets/set1/feminino"
            pathFilesM_set2 = "/home/ahirton/Python/gender_classification/tweets/set1/masculino"
            pathFilesF_set1 = "/home/ahirton/Python/gender_classification/tweets/set2/feminino"
            pathFilesM_set1 = "/home/ahirton/Python/gender_classification/tweets/set2/masculino"
        else:        
            pathFilesF_set1 = "/home/ahirton/Python/gender_classification/tweets/set1/feminino"
            pathFilesM_set1 = "/home/ahirton/Python/gender_classification/tweets/set1/masculino"
            pathFilesF_set2 = "/home/ahirton/Python/gender_classification/tweets/set2/feminino"
            pathFilesM_set2 = "/home/ahirton/Python/gender_classification/tweets/set2/masculino"
            
            #pathFilesF_set1 = "/home/rpasti/workspace/gender_classification/tweets/set1/feminino"
            #pathFilesM_set1 = "/home/rpasti/workspace/gender_classification/tweets/set1/masculino"
            #pathFilesF_set2 = "/home/rpasti/workspace/gender_classification/tweets/set2/feminino"
            #pathFilesM_set2 = "/home/rpasti/workspace/gender_classification/tweets/set2/masculino"
        
        pathSave = "/home/ahirton/Python/gender_classification/outputfiles/teste1"
        #pathSave = "/home/rpasti/workspace/gender_classification/outputfiles/"
        
        print("Processando arquivos e extraindo meta-atributos")
        dataSet1, dataSet1MA = self.load_and_process_samples(pathFilesF_set1, pathFilesM_set1, pathSave, setId=1)
        dataSet2, dataSet2MA = self.load_and_process_samples(pathFilesF_set2, pathFilesM_set2, pathSave, setId=2)
        
        print("Agregando bases...")
        totalSamples = 0
        totalSet1 = 0
        totalSet2 = 0
        for key in dataSet1.keys():
            totalSamples = totalSamples + len(dataSet1MA[key])
            totalSet1 = totalSet1 + len(dataSet1MA[key])
            totalSamples = totalSamples + len(dataSet2MA[key])
            totalSet2 = totalSet2 + len(dataSet2MA[key])
        print("Número total de textos: " + str(totalSamples) + " / Conjunto 1: " + str(totalSet1) + " / Conjunto 2: " + str(totalSet2))
        
        
        '''Unir ou não as bases'''
        if self.singleDataSet:
            #print("unindo bases de dados")
            dataSetUMA = self.data_set_union(dataSet1MA, dataSet2MA)
            dataSetU = self.data_set_union(dataSet1, dataSet2)
            '''Filtra por entropia'''            
            if self.filterEntropy:                 
                dataSetU, dataSetUMA = self._filter_entropy(dataSetU, dataSetUMA)
                       
        '''Filtra por entropia'''            
        if self.filterEntropy:                 
            dataSet1, dataSet1MA = self._filter_entropy(dataSet1, dataSet1MA)
            dataSet2, dataSet2MA = self._filter_entropy(dataSet2, dataSet2MA)
       
        allResults = []
        numpy.random.seed(10)
        for iExp in range(0,self.nExp):
            print("Executando experimento " + str(iExp))
        
            '''Aplicar bagging para reamostrar e escolher bases de treinamento e teste'''
            
            if self.singleDataSet:
                self.trainingSet, dataSetUMARed = self.bagging(dataSetUMA, self.nSamplesTraining)
                self.testSet = self.select_all(dataSetUMARed)
                self.nSamplesTestMale = len(dataSetUMARed["male"])
                self.nSamplesTestFemale = len(dataSetUMARed["female"])
            else:            
                self.trainingSet, dataSet1Red = self.bagging(dataSet1MA, self.nSamplesTraining)
                self.testSet = self.select_all(dataSet2MA)
                self.nSamplesTestMale = len(dataSet2MA["male"])
                self.nSamplesTestFemale = len(dataSet2MA["female"])
          
                
            nTrainingSamples = len(self.trainingSet)
            nTestSamples = len(self.testSet)
            
            print("Conjunto de treinamento: " + str(nTrainingSamples) + " / Conjunto de teste: " + str(nTestSamples))
            
            
            nDim = self.trainingSet[0][0].shape[0]
            '''convertendo para o formato do scikit-learn'''
            self.samplesTraining = numpy.zeros([nTrainingSamples,nDim])
            self.samplesTrainingClasses = numpy.zeros(nTrainingSamples)
            self.samplesTest = numpy.zeros([nTestSamples,nDim])
            self.samplesTestClasses = numpy.zeros(nTestSamples)
            
            for iSample in range(0,nTrainingSamples):
                self.samplesTraining[iSample] = self.trainingSet[iSample][0]
                self.samplesTrainingClasses[iSample] = self.trainingSet[iSample][1]
            for iSample in range(0,nTestSamples):
                self.samplesTest[iSample] = self.testSet[iSample][0]
                self.samplesTestClasses[iSample] = self.testSet[iSample][1]            
           
            #print(samplesTraining[0])
            #print(samplesTest[0])
            '''Fazendo classificação (treinamento + avaliação por meio de base de teste)'''
            self.learning()
            result = self.evaluate()
            allResults.append(result)
        
        self.evaluate_total(allResults)    
    
    
    def data_set_union(self, dataSet1, dataSet2):    
        newDataSet = {}
        for key in dataSet1.keys():
            newDataSet[key] = []
            
        for key in dataSet1.keys():
            newDataSet[key] = newDataSet[key] + dataSet1[key]
            newDataSet[key] = newDataSet[key] + dataSet2[key]
        return newDataSet
        
    def evaluate_total(self, allResults):
        sumResults = {}
        nRes = len(allResults)
        vAcc = numpy.zeros(nRes)
        for iRes in range(0,nRes):
            vAcc[iRes] = allResults[iRes]["acc"]
        
        sumResults["std"] = numpy.std(vAcc)
        sumResults["mean"] = numpy.mean(vAcc)
        sumResults["min"] = numpy.min(vAcc)
        sumResults["max"] = numpy.max(vAcc)
        sumResults["recall"] = numpy.recall(vAcc)
        
        print(vAcc)
        print(sumResults)    
            
    def evaluate(self):
    
        predictedClasses = self.clf.predict(self.samplesTest)
        '''Avaliar classes preditas'''
        results = {}
        nSamplesTest = len(predictedClasses)
        results["acc"] = 0
        results["acc_male"] = 0
        results["acc_female"] = 0
        for iSampleTest in range(0,nSamplesTest):
            if predictedClasses[iSampleTest] == self.samplesTestClasses[iSampleTest]:
                results["acc"] = results["acc"] + 1
                if self.samplesTestClasses[iSampleTest] == 1:
                    results["acc_male"] = results["acc_male"] + 1
                else:
                    results["acc_female"] = results["acc_female"] + 1
       
        results["acc"] = results["acc"]/nSamplesTest
        results["acc_male"] = results["acc_male"]/(self.nSamplesTestMale)
        results["acc_female"] = results["acc_female"]/(self.nSamplesTestFemale)
        #print(predictedClasses)
        #print(samplesTestClasses)
        return results   
        
        
    def learning(self):
        '''
        Aprendizagem de Máquina usando métodos do Scikit-learn
        ''' 
      
        from sklearn import preprocessing
        #samplesTraining = preprocessing.normalize(samplesTraining, norm='max', axis=0, copy=True, return_norm=False)
        #samplesTest = preprocessing.normalize(samplesTest, norm='max', axis=0, copy=True, return_norm=False)
            
        if self.method == "svm":        
            from sklearn import svm
            self.samplesTraining = preprocessing.scale(self.samplesTraining)
            self.samplesTest = preprocessing.scale(self.samplesTest)
            self.clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                              decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                              max_iter=-1, probability=False, random_state=None, shrinking=True,
                              tol=0.001, verbose=False)
            self.clf.fit(self.samplesTraining, self.samplesTrainingClasses)
           
        if self.method == "naive_bayes":
            self.samplesTraining = preprocessing.minmax_scale(self.samplesTraining, feature_range=(-0.01, 0.01), axis=0, copy=True)
            self.samplesTest = preprocessing.minmax_scale(self.samplesTest, feature_range=(-0.01, 0.01), axis=0, copy=True)
            from sklearn.naive_bayes import GaussianNB
            self.clf = GaussianNB()    
            self.clf.fit(self.samplesTraining, self.samplesTrainingClasses)
            
        if self.method == "mlp":
            self.samplesTraining = preprocessing.minmax_scale(self.samplesTraining, feature_range=(-0.01, 0.01), axis=0, copy=True)
            self.samplesTest = preprocessing.minmax_scale(self.samplesTest, feature_range=(-0.01, 0.01), axis=0, copy=True)
            #self.samplesTraining = preprocessing.scale(self.samplesTraining)
            #self.samplesTest = preprocessing.scale(self.samplesTest)
            from sklearn.neural_network import MLPClassifier              
            self.clf = MLPClassifier(hidden_layer_sizes=(10, 2), activation='tanh', solver='lbfgs', alpha=1e-5, random_state=1)
            self.clf.fit(self.samplesTraining, self.samplesTrainingClasses)
            
#        if self.method== "dt":
#            from sklearn import tree
#            self.samplesTraining = preprocessing.scale(self.samplesTraining)
#            self.samplesTest = preprocessing.scale(self.samplesTest)
#            self.clf = tree(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
#                                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
#                                                      random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
#            self.clf.fit(self.samplesTraining, self.samplesTrainingClasses)
            
    def _extract_meta_attributes(self, samplesInClasses):
        '''
        Extrai meta-atributos das amostras
        '''
        print("extraindo meta-atributos...")
        countProcess = 0
        namesOfClasses = list(samplesInClasses.keys())
        nClasses = len(namesOfClasses)
        samplesInClassesT = {}
        for iClass in range(0,nClasses):
            samplesInClassesT[namesOfClasses[iClass]] = []
        
        for iClass in range(0,nClasses):
            documentsList = list(samplesInClasses[namesOfClasses[iClass]])
            nDocs = len(documentsList)
            for iDoc in range(0,nDocs):
                #start = time.clock()
                print("extraindo meta-atributos do texto " + str(countProcess))
                metaAttributes = MetaAttributes(documentsList[iDoc])
                countProcess = countProcess + 1
                #ma = metaAttributes.all_meta_attributes(categories=self.categoriesMA)
                samplesInClassesT[namesOfClasses[iClass]].append(metaAttributes.all_meta_attributes(self.categoriesMA))
                #print("Tempo de extração: " + str(time.clock() - start))
        return samplesInClassesT        
        
    
    def _filter_entropy(self, samplesInClasses, samplesInClassesMA):
        '''
        Filtra amostras pela entropia
        '''
        namesOfClasses = list(samplesInClasses.keys())
        nClasses = len(namesOfClasses)
        samplesInClassesT = {}
        samplesInClassesTMA = {}
        
        for iClass in range(0,nClasses):
            samplesInClassesT[namesOfClasses[iClass]] = []
            samplesInClassesTMA[namesOfClasses[iClass]] = []
                        
        for iClass in range(0,nClasses):
            documentsList = list(samplesInClasses[namesOfClasses[iClass]])
            documentsListMA = list(samplesInClassesMA[namesOfClasses[iClass]])
            nDocs = len(documentsList)
            vEntropy = numpy.zeros(nDocs)
            '''calcula entropia de cada texto'''
            for iDoc in range(0,nDocs):
                vEntropy[iDoc] = self._entropy(documentsList[iDoc])
               
            
            '''seleciona as N com maior entropia'''
            #if namesOfClasses[iClass] == "female":
            nMaxSamplesEntropy = int(numpy.ceil(nDocs*self.pSamplesEntropy))
            #else:
            #    nMaxSamplesEntropy = nDocs
            for iDoc in range(0,nMaxSamplesEntropy):
                iMax = numpy.argmax(vEntropy)                
                samplesInClassesT[namesOfClasses[iClass]].append(documentsList[iMax])
                samplesInClassesTMA[namesOfClasses[iClass]].append(documentsListMA[iMax])                            
                vEntropy[iMax] = 0
        
           
        
        return samplesInClassesT, samplesInClassesTMA
                
    def _entropy(self, words):
        FREQ = dict(nltk.FreqDist(words))
        V = dict(nltk.FreqDist(FREQ.values())) 
        nWords = len(words)
        entropy = 0
        for f, amt in list(V.items()):
            it = amt * (-math.log10(f/float(nWords))) * (f/float(nWords))
            entropy = entropy + it
        #max = float(math.log10(self.N))
        return entropy
    
    
    def select_all(self, samplesInClasses):    
        
        namesOfClasses = list(samplesInClasses.keys())
        classes = list(samplesInClasses.keys())
        nClasses = len(classes)       
        samplesSelected = []
       
        for iClass in range(0,nClasses):
            documentsList = list(samplesInClasses[namesOfClasses[iClass]])
            nDocs = len(documentsList)              
            #while iDoc < nDocsTrain:
            for iDoc in range(0,nDocs):                
                if namesOfClasses[iClass] == "male":
                    nameClass = 1
                else:
                    nameClass = -1
                samplesSelected.append((documentsList[iDoc], nameClass))
                nDocs = nDocs - 1
           
        return samplesSelected                
    
        
    
    def bagging(self, samplesInClasses, nMaxSamples):    
        
        namesOfClasses = list(samplesInClasses.keys())
           
        #Descobrir quantas amostras em cada classe
        classes = list(samplesInClasses.keys())
        nClasses = len(classes)
        vNSamples = numpy.zeros(nClasses)
        for iClass in range(0,nClasses):
            vNSamples[iClass] = len(samplesInClasses[classes[iClass]])
       
        nMinSamples = int(numpy.min(vNSamples))
        if nMinSamples < nMaxSamples:
            nMaxSamples = nMinSamples
        
        samplesInClassesRed = {}
        samplesSelected = []
        for key in samplesInClasses.keys():
            samplesInClassesRed[key] = []
      
        for iClass in range(0,nClasses):
            documentsList = list(samplesInClasses[namesOfClasses[iClass]])
            nDocs = len(documentsList)              
           
            for iDoc in range(0,nMaxSamples):                  
                iDocRand = numpy.random.randint(0,nDocs-1)            
                textSample = documentsList.pop(iDocRand)
                nWordsInSample = len(textSample)                
                if namesOfClasses[iClass] == "male":
                    nameClass = 1
                else:
                    nameClass = -1
                samplesSelected.append((textSample, nameClass))
                nDocs = nDocs - 1
            samplesInClassesRed[namesOfClasses[iClass]] = documentsList
            
        return samplesSelected, samplesInClassesRed                 
            
    def load_and_process_samples(self, pathFilesF, pathFilesM,pathSave, setId):
        
        fileUtils = FileUtils(pathSave)
        if self.loadCSV == False:
            samplesInClasses = fileUtils.load_object("samples_in_classes_" + str(setId), "dict")
            samplesInClassesMA = fileUtils.load_object("samples_in_classes_MA_" + str(setId), "dict")
        else:
            samplesInClasses = self._read_csv(pathFilesF, pathFilesM)
            fileUtils.save_object(samplesInClasses, "samples_in_classes_"  + str(setId), "dict")
                        
            '''Transforma em meta-atributos'''
            samplesInClassesMA = self._extract_meta_attributes(samplesInClasses)
            fileUtils.save_object(samplesInClassesMA, "samples_in_classes_MA_"  + str(setId), "dict")
        return samplesInClasses, samplesInClassesMA
                    
    
    def _read_csv(self, pathFilesF, pathFilesM):
        '''
        '''
        textProcessing = TextProcessing()
       
        samplesInClasses = {}
        samplesInClasses["female"] = []
        samplesInClasses["male"] = []        
        
        samplesF = self._read_files(pathFilesF)
        samplesM = self._read_files(pathFilesM)
        
      
        iText = 0
        totalText = len(samplesM) + len(samplesF)
       
        for sample in samplesF:                
            nTokens = len(textProcessing.tokenize_one(sample))                
            if  nTokens > 1:
                #print("Texto " + str(iText) + " / " + str(totalText))                              
                samplesInClasses["female"].append(sample)                   
                iText = iText + 1                    
                #print(metaAttributes.all_meta_attributes)
                #input('-----------------------------')      
        
        for sample in samplesM:
            nTokens = len(textProcessing.tokenize_one(sample))
            if  nTokens > 1:
                #print("Texto " + str(iText) + " / " + str(totalText))                    
                samplesInClasses["male"].append(sample)
                iText = iText + 1        
                       
        return samplesInClasses
        #print(self.samplesInClasses["male"][0:10])
        #print(len(self.samplesInClasses["male"]))
    
    
    def _read_files(self,pathFiles):
        import sys
        textProcessing = TextProcessing()
        fileSearch = FileSearch()
        files = fileSearch.search_by_type(pathFiles, "csv")
        samples = []
        csv.field_size_limit(sys.maxsize)
        for file in files:       
            csvfile = open(file, newline='')
            print (file)
            csvreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            #print(csvreader)
            for row in csvreader:
                nTokens = len(textProcessing.tokenize_one(' '.join(row)))
                if  nTokens < 50:
                #print(row)
                    samples.append(' '.join(row))          
        
        return samples
    
   
    
    
    
    
    
    
if __name__ == '__main__':
    main()