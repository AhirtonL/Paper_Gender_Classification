# -*- coding: utf-8 -*-
"""
Created on Wed Apr 09 12:01:41 2014

@author: rpasti
"""
import pickle
import json
import os

class FileUtils():
    def __init__(self, root_path):
        self.root_path = root_path
        
    #Salva um JSON a partir de um dicionario
    def save_json(self,jsonName,dictInfo):
        file = open(self.root_path + jsonName + '.' + 'json', 'w+')        
        json.dump(dictInfo,file)
        file.close()
        
    #Carrega um JSON em um dicionario    
    def load_json(self,jsonName):
        file = open(self.root_path + jsonName + '.' + 'json', 'r')
        dictInfo = json.load(file)        
        file.close()
        return dictInfo
        
    #Salva um objeto ou estrutura em um arquivo
    def save_object(self, obj, objName, objType):
        file = open(self.root_path + objName + '.' + objType, 'wb+')
        pickle.dump(obj, file)
        file.close()
        
    #Carrega um objeto de um arquivo
    def load_object(self, objName,objType):
        file = open(self.root_path + objName + '.' + objType, 'rb') 
        obj = pickle.load(file)    
        file.close()
        return obj
