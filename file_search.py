# -*- coding: utf-8 -*-
'''
Created on 18/11/2016

@author: Rodrigo
'''
import os
import numpy
from text_processing import TextProcessing

class FileSearch(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def search_by_type(self, pathFiles, targetTypes):
        textProcessing = TextProcessing()              
        filesFound = []
        for root, dirs, files in os.walk(pathFiles):
            #Verificar se existe arquivo e se é do tipos alvo
            if len(files)>0:
                for file in files:
                    tokens = textProcessing.tokenize_one(file.replace(".", " "))                    
                    if tokens[len(tokens)-1] in targetTypes:
                        filesFound.append(root+ "/" + file)        
        return filesFound
        
    def search_by_type_name(self, pathFiles, targetTypes, targetNames):
        textProcessing = TextProcessing()
        from models.text_analysis.nlp_basics.string_analysis import StringAnalysis
        stringAnalysis = StringAnalysis()         
        filesFound = []
        nNames = len(targetNames)
        for root, dirs, files in os.walk(pathFiles):
            #Verificar se existe arquivo e se é do tipos alvo
            if len(files)>0:
                for file in files:
                    tokens = textProcessing.tokenize_one(file.replace(".", " "))                                    
                    if tokens[len(tokens)-1] in targetTypes:
                        #Verificar os nomes
                        vFound = numpy.zeros(nNames)
                        for iName in range(0,nNames):
                            fileName = textProcessing.text_lower_one([tokens[0]])[0]
                            name = textProcessing.text_lower_one([targetNames[iName]])[0]
                            dist = stringAnalysis.string_in_string_dist(fileName, name)
                            if dist==1:
                                vFound[iName] = 1
                        '''Somente operação AND'''        
                        if numpy.sum(vFound) == nNames:
                            filesFound.append(root+ "/" + file)
        #print(len(filesFound))
        #input('------------------')
        return filesFound