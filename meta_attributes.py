#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
__author__ = "Ahirton Lopes & Rodrigo Pasti"
__copyright__ = "Copyright 2015/2016, Mackenzie University"
__credits__ = ["Ahirton Lopes", "Rodrigo Pasti", "Leandro de Castro"]
__license__ = "None"
__version__ = "1.0"
__maintainer__ = "Ahirton Lopes"
__email__ = "ahirton.xd@gmail.com"
__status__ = "Beta"
"""

import re
import nltk
import math
import numpy
from text_processing import TextProcessing
from file_utils import FileUtils
import semantic_dictionaries


class MetaAttributes(object):
    def __init__(self, text):
        import time
        start = time.clock()
        '''
        -----------------------------------------------------------------------------------------------------------------------
        DEFINICAO DOS PARAMETROS DE CONTROLE
        -----------------------------------------------------------------------------------------------------------------------
        '''        
        tp = TextProcessing()
        
        self.nMaxLengthFreq = 16 
#       OBS1: Tamanho maximo de palavra a ser considerado na frequencia do tamanho de palavras       
        savePath = "/home/ahirton/Python/gender_classification/outputfiles/"
        #savePath = "/home/rpasti/workspace/gender_classification/outputfiles/"
        tagged = tp.tagging([tp.tokenize([text])[0]],savePath,"en")[0]
        fileUtils = FileUtils(savePath)
        
        text = re.sub("http","", text)
        self.raw = text
        
#        print tagged

        self.PARAGRAPHS = []
        self.SENTENCES = []
        self.WORDS = []
        delimiters = '\n','. \n', '! \n', '?\n', '.\n', '!\n', '?\n', '... \n' #, '... \n'#, ' \n ' #, " .\n", " !\n", ' ?\n'
        regexPattern = '|'.join(map(re.escape, delimiters))
       
        for paragraph in re.split(regexPattern,self.raw):        
            p = []
#            print ""
#            print paragraph            
#            raw_input(".----------------.FIM DE PARÁGRAFO----------------.")
            #sentences = tp.tokenize_sentence([paragraph])[0]
            for sentence in tp.tokenize_sentence([paragraph])[0]: 
#                print ""
#                print sentence
#                print tp.tagging(tp.tokenize([sentence]))
#                raw_input(".---------------..FIM DE FRASE...------.")
                words = tp.tokenize([sentence])[0]
                #words = tp.remove_punctuation([words])[0]
                self.WORDS.extend(words)
                self.SENTENCES.append(sentence)
                p.append(words)
#                print paragraph
#                print sentence
#                print words
#                print self.WORDS
#                raw_input('XXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            self.PARAGRAPHS.append(p)
            self.C = len(text)
            self.LOWER = MetaAttributes._count_char(text, "^[a-z_-]*$")
            self.UPPER = MetaAttributes._count_char(text, "^[A-Z_-]*$")
            self.NUMBERS = MetaAttributes._count_char(text, "^[\d]*$")
            self.WHITE = MetaAttributes._count_char(text, "^[ ]*$")
            self.TAB = MetaAttributes._count_char(text, "^[\t]*$")
            self.N = len(self.WORDS)
            self.SIZES = []
            self.FREQ = {}
        
        for w in self.WORDS:            
            self.SIZES.append(len(w))
            self.FREQ = dict(nltk.FreqDist(self.WORDS))
            self.V = dict(nltk.FreqDist(self.FREQ.values())) 
            self.VRICH = self.N - len(self.V)
            self.HXLEGO = []
            self.HXDISLEGO = []

        for w, t in self.FREQ.items():
            if t == 1:
                self.HXLEGO.append(w)
            elif t == 2:
                self.HXDISLEGO.append(w)
                
            self.TAGGED = tagged
            self.S = len(self.SENTENCES)
            
        self.pwdictionary = semantic_dictionaries.extended_positive()
        self.nwdictionary = semantic_dictionaries.extended_negative()
        self.neutralwdictionary = semantic_dictionaries.extended_neutral_words()
        self.LIWCdict = fileUtils.load_object("liwc", "dict")
        
         
        
        #self.N_EMOTICONS_NE = self._count_word_dictionary(self.raw, self.nedictionary)
        #self.N_EMOTICONS_POS= self._count_word_dictionary(self.raw, self.pedictionary)
    
    #@staticmethod
    def _count_word_dictionary(self, sample, dictionary):
        '''
        Conta quantas palavras de um dicionário ocorre em um texto
        '''
        countWord = 0
        for word in dictionary:
            findRes = sample.find(word)
            if findRes != -1 and sample[findRes-1] == " ":# and (sample[findRes+len(word)] == " " or :
                countWord = countWord + 1
                print(word)
                print(sample)
                input('----------------')
        return countWord
        
    
    @staticmethod
    def _count_char(text, regex = None):        
        if regex is None:
            return len(text)
        else:
            t = 0
            for c in text:
                if re.match(regex, c):
                    t = t + 1
            return t

    '''
    -----------------------------------------------------------------------------------------------------------------------
    DEFINICAO DOS META-ATRIBUTOS
    -----------------------------------------------------------------------------------------------------------------------
    '''
    
    '''
    -----------------------------------------------------------------------------------------------------------------------------------------------
    META-ATRIBUTOS BASEADOS NO Linguistic Inquiry and Word Count (LIWC) - 64 meta-atributos (baseados nas 64 categorias do dicionário em português)
    -----------------------------------------------------------------------------------------------------------------------------------------------
    '''
    
    @property
    def LIWC_N(self):
        keys = list(self.LIWCdict.keys())
        nKeys = len(keys) 
        vLIWC = numpy.zeros(nKeys)      
        for iKey in range(0,nKeys):
            for word in self.WORDS:
                if word in self.LIWCdict[keys[iKey]]:
                    vLIWC[iKey] = vLIWC[iKey] + 1
            vLIWC[iKey] = vLIWC[iKey]/self.N 
        return vLIWC
    
    
    '''
    -----------------------------------------------------------------------------------------------------------------------
    META-ATRIBUTOS BASEADOS EM ANALISE DE CARACTERES (C) - 16 meta-atributos 
    -----------------------------------------------------------------------------------------------------------------------
    '''

    @property
    def C1(self):
        return self.C 
#       C1: Quantidade total de caracteres (C)         
    @property
    def C2(self):
        return self.LOWER / float(self.C)
#       C2: Razão entre o número total de letras minúsculas (a-z) e o total de caracteres (C)
    @property
    def C3(self):
        return self.UPPER / float(self.C)
#       C3: Razão entre o número total de caracteres maiúsculos (A-Z) e o total de caracteres (C)
    @property
    def C4(self):
        return self.NUMBERS / float(self.C)
#       C4: Razão entre a quantidade de números (dígitos) e o total de caracteres (C)
    @property
    def C5(self):
        return self.WHITE / float(self.C)
#       C5: Razão entre o número de espaços em branco (white spaces) e o total de caracteres (C)
    @property
    def C6(self):
        return self.TAB / float(self.C)
#       C6: Razão entre o número de tabulações (tab spaces) e o total de caracteres (C)
    @property
    def C7(self):
        return MetaAttributes._count_char(self.raw, r'\'') / float(self.C)
#       C7: Razão entre o número de aspas (“) e o total de caracteres (C)
    @property
    def C8(self):
        return MetaAttributes._count_char(self.raw, r'\,') / float(self.C)
#       C8: Razão entre o número de vírgulas (,) e o total de caracteres (C)
    @property
    def C9(self):
        return MetaAttributes._count_char(self.raw, r'\:') / float(self.C)
#       C9: Razão entre o número de dois-pontos (:) e o total de caracteres (C)
    @property
    def C10(self):
        return MetaAttributes._count_char(self.raw, r'\;') / float(self.C)
#       C10: Razão entre o número de ponto-e-vírgulas (;) e o total de caracteres (C)
    @property
    def C11(self):
        return MetaAttributes._count_char(self.raw, r'\?') / float(self.C)
#       C11: Razão entre o número de pontos de interrogação simples (?) e o total de caracteres (C)
    @property
    def C12(self):
        return MetaAttributes._count_char(self.raw, r'\?\?+') / float(self.C)
#       C12: Razão entre o número de múltiplos pontos de interrogação (???) e o total de caracteres (C)
    @property
    def C13(self):
        return MetaAttributes._count_char(self.raw, r'\!') / float(self.C)
#       C13: Razão entre o número de pontos de exclamação simples (!) e o total de caracteres (C)
    @property
    def C14(self):
        return MetaAttributes._count_char(self.raw, r'\!\!+') / float(self.C)
#       C14: Razão entre o número de múltiplos pontos de exclamação (!!!) e o total de caracteres (C)
    @property
    def C15(self):
        return MetaAttributes._count_char(self.raw, r'\.') / float(self.C)
#       C15: Razão entre o número de pontos finais simples (.) e o total de caracteres (C)
    @property
    def C16(self):
        return MetaAttributes._count_char(self.raw, r'\.\.+') / float(self.C)     
#       C16: Razão entre o número de reticências (...) e o total de caracteres (C)     
    
    '''
    -----------------------------------------------------------------------------------------------------------------------
    META ATRIBUTOS BASEADOS EM ANÁLISE DE PALAVRAS (W) - 28 meta-atributos (12 + W13_N)
    -----------------------------------------------------------------------------------------------------------------------
    '''

    @property
    def W1(self):  
        return self.N
#       W1: Quantidade total de palavras (P)
    @property
    def W2(self):     
        n = []
        for w in self.WORDS:
            n.append(len(w))
        return float(numpy.mean(n))
#       W2: Média da quantidade de caracteres por palavra (P)
    @property
    def W3(self):
        return self.VRICH / float (self.N)
#       W3: Razão entre o número de palavras diferentes e o total de palavras (número total de palavras diferentes/P)
    @property
    def W4(self):    
        t = 0
        for s in self.SIZES:
            if s > 6:
                t = t + 1
        return t / float(self.N)
#       W4: Razão entre o número de palavras com mais de 6 caracteres e o total de palavras (P)
    @property
    def W5(self):
        t = 0
        for s in self.SIZES:
            if s <= 3:
                t = t + 1
        return t / float(self.N)
#       W5: Razão entre o número de palavras com 1 a 3 caracteres (palavras curtas) e o total de palavras (P)
    @property
    def W6(self):
        return len(self.HXLEGO) / float(self.N)
#       W6: Razão entre hapax legomena (palavra que ocorre uma única vez em todo um texto) e o número total de palavras (P)
    @property
    def W7(self):
        return len(self.V) / float(self.N)
#       W7: Razão entre hapax dislegomena (palavra que ocorre apenas duas vezes em todo um texto) e o número total de palavras (P)
    @property
    def W8(self):
        sum = 0       
        for f, amt in self.V.items():
            it = amt * math.pow(f/float(self.N), 2)
            sum = sum + it
        yule = (-1.0/self.N + sum)
#       print amt
#       print it       
#       print self.N
#       print sum
#       print yule
        return yule
#       W8: Medida K de Yule*
    @property
    def W9(self):
        simpson = 0
        i=1
#       OBS2: Sentenças de apenas uma palavra tornam a funcao indefinida
        for f, amt in self.V.items():
            it = amt *(i / float(self.N)) * ((i-1.0) / (self.N - 1.0))
            simpson = simpson + it
            i=i+1
#       print amt
#       print i
#       print self.N
#       print it 
#       print simpson        
        return simpson
#       W9: Medida D de Simpson*
    @property
    def W10(self):
        return len(self.V) / float(self.VRICH)
#       W9: Medida S de Sichel*
    @property
    def W11(self):
        hlego_count = len(self.HXLEGO)
        v_count = float(self.VRICH)
#       print hlego_count
#       print v_count        
#       W11: Medida R de Honore*        
        if hlego_count == v_count:
            return 0
        else:
            return (100.0 * math.log10(self.N)) / float(1.0-(hlego_count/float(v_count))) 
    @property
    def W12(self):
        entropy = 0
        for f, amt in self.V.items():
            it = amt * (-math.log10(f/float(self.N))) * (f/float(self.N))
            entropy = entropy + it
#       print f
#       print amt
#       print self.N
#       print it 
#       print entropy
#       max = float(math.log10(self.N))        
        return entropy# / max
#       W12: Medida de Entropia*
    @property
    def W13_N(self):
#       OBS3: A maior palavra em portugues possui tamanho 46, segundo o dicionário Houaiss
        vFreq = numpy.zeros([self.nMaxLengthFreq])      
        for item in nltk.FreqDist(self.SIZES).items():           
            if item[0] < self.nMaxLengthFreq:
                vFreq[item[0]-1] = item[1]/float(self.N)
#            if item[0] >50:
#                print self.raw
#                print nltk.FreqDist(self.SIZES).items()
#                print item
#                print vFreq
#                raw_input('+++++++++++')
        return vFreq
#       W13_N: Razão entre a distribuição de frequência do tamanho das palavras (16 meta-atributos diferentes) e o total de palavras (P)
   
    '''
    -----------------------------------------------------------------------------------------------------------------------
    META ATRIBUTOS BASEADOS EM ANALISE DA ESTRUTURA TEXTUAL (TS) - 10 meta-atributos
    -----------------------------------------------------------------------------------------------------------------------
    ''' 
   
    @property
    def TS1(self):
        return self.S
#       TS1: Quantidade total de frases (F)
    @property
    def TS2(self):
        return len(self.PARAGRAPHS)
#       TS2: Quantidade total de parágrafos
    @property
    def TS3(self):
        sents_per_paragraph = []
        for p in self.PARAGRAPHS:
            sents_per_paragraph.append(len(p))
        return numpy.average(sents_per_paragraph)
#       TS3: Média de frases (F) por parágrafo
    @property
    def TS4(self):
        words_per_paragraph = []
        for p in self.PARAGRAPHS:
            total_words = 0
            for s in p:
                for w in s:
                    if w.isalpha():
                        total_words = total_words + 1
            words_per_paragraph.append(total_words)
        return numpy.average(words_per_paragraph)
#       TS4: Média de palavras (P) por parágrafo
    @property
    def TS5(self):
        chars_per_paragraph = []
        for p in self.PARAGRAPHS:
            total_chars = 0
            for s in p:
                for w in s:
                    total_chars = total_chars + len(w)
            chars_per_paragraph.append(total_chars)
        return numpy.average(chars_per_paragraph)
#       TS5: Média de caracteres (C) por parágrafo
    @property
    def TS6(self):
        words_per_sentence = []
        for p in self.PARAGRAPHS:
            for s in p:
                total_words = 0
                for w in s:
                    if w.isalpha():
                        total_words = total_words + 1
                words_per_sentence.append(total_words)
        return numpy.average(words_per_sentence)
#       TS6: Média de palavras (P) por frase (F)
    @property
    def TS7(self):
        amt = 0
        for p in self.PARAGRAPHS:
            for s in p:
                if len(s)>0:
                    first_char = s[0][0]
                    if first_char.islower():
                        amt = amt + 1
        return amt / float(self.S)
#       TS7: Razão entre o número de frases iniciadas por letra minúscula (a-z) e o número total de frases (F)
    @property
    def TS8(self):
        amt = 0        
        for p in self.PARAGRAPHS:            
            for s in p:
                if len(s)>0:
                    first_char = s[0][0]
                    if first_char.isupper():
                        amt = amt + 1                
        return amt / float(self.S)       
#       TS8: Razão entre o número de frases iniciadas por letra maiúscula (A-Z) e o número total de frases (F)
    @property
    def TS9(self):
        blank = 0
        for p in self.PARAGRAPHS:
            if len(p) == 0:
                blank = blank + 1
        return blank / float(self.TS2)
#       TS9: Razão entre o número de linhas em branco e o total de parágrafos
    @property
    def TS10(self):
        lenghts = []
        for p in self.PARAGRAPHS:
            lenght = 0
            for s in p:                
                for w in s:
                    #if len(w)>0:                        
                    lenght = lenght + len(w)
            lenghts.append(lenght)
        return numpy.average(lenghts)
#       TS10: Quantidade média de caracteres em linhas não vazias
        
    '''
    -----------------------------------------------------------------------------------------------------------------------
    META ATRIBUTOS BASEADOS EM ANALISE DA MORFOLOGIA TEXTUAL (TM) - 6 meta-atributos
    -----------------------------------------------------------------------------------------------------------------------
    '''

    @property
    def TM1(self):
        articles = []
        for word, tag in self.TAGGED:
            if tag == 'ART':
                articles.append(word)
        countWords = nltk.FreqDist(articles)

        article_ratio = 0
        for count in countWords.values():
            article_ratio = article_ratio + count 
        article_ratio = article_ratio/float(self.N)
        return article_ratio
        # TM1: Razão entre o número de artigos e o total de palavras (P)
    
    @property
    def TM2(self):
        countWords = self._morfo_freq(['PROADJ', 'PRO-KS', 'PROPESS', 'PRO-KS-REL', 'PRO-SUB'])

        pronoun_ratio = 0
        for count in countWords.values():
            pronoun_ratio = pronoun_ratio + count
        pronoun_ratio = pronoun_ratio / float(self.N)
        return pronoun_ratio
#       TM2: Razão entre o número de pronomes e o total de palavras (P)
    @property
    def TM3(self):
        countWords = self._morfo_freq('VAUX')

        verb_ratio = 0
        for count in countWords.values():
            verb_ratio = verb_ratio + count
        verb_ratio = verb_ratio / float(self.N)
        return verb_ratio
#       TM3: Razão entre o número de verbos-auxiliares e o total de palavras (P)
    @property
    def TM4(self):
        countWords = self._morfo_freq(['KC', 'KS'])
        
        conj_ratio = 0
        for count in countWords.values():
            conj_ratio = conj_ratio + count
        conj_ratio = conj_ratio / float(self.N)
        return conj_ratio
#       TM4: Razão entre o número de conjunções e o total de palavras (P)
    @property
    def TM5(self):
        countWords = self._morfo_freq('IN')
        inter_ratio = 0
        for count in countWords.values():
            inter_ratio = inter_ratio + count 
        inter_ratio = inter_ratio/float(self.N)
        return inter_ratio
#       TM5: Razão entre o número de interjeições e o total de palavras (P)
    @property
    def TM6(self):
        countWords = self._morfo_freq('PREP')        
        prep_ratio = 0
        for count in countWords.values():
            prep_ratio = prep_ratio + count
        prep_ratio = prep_ratio / float(self.N)
        return prep_ratio
#       TM6: Razão entre o número de preposições e o total de palavras (P)    
    
    '''   
    -----------------------------------------------------------------------------------------------------------------------
    META ATRIBUTOS BASEADOS EM SOCIOLINGUISTICA (Psycholinguistic Cues) - 3 meta-atributos
    -----------------------------------------------------------------------------------------------------------------------
    A partir daqui atributos sociolinguisticos baseados em dicionario
    '''
#    def ma_dict_x(dictionary,tokenizedText):
    
#    nTokensInText = len(tokenizedText)
#    nWordsFound = 0    
#    for iToken in range(0,nTokensInText):
#        if tokenizedText[iToken] in dictionary:
#            nWordsFound = nWordsFound + 1
            
#    p = nWordsFound/float(nTokensInText)
#    return p
    
    @property
    def PC1(self):
    
        nTokensInText = len(self.WORDS)
        nWordsFound = 0    
        for iToken in range(0,nTokensInText):
            if self.WORDS[iToken] in self.pwdictionary:
                nWordsFound = nWordsFound + 1
           
        PW = nWordsFound/float(nTokensInText)
        return PW
    
    @property        
    def PC2(self):
    
        nTokensInText = len(self.WORDS)
        nWordsFound = 0    
        for iToken in range(0,nTokensInText):
            if self.WORDS[iToken] in self.nwdictionary:
                nWordsFound = nWordsFound + 1
           
        NW = nWordsFound/float(nTokensInText)
        return NW
    
    @property
    def PC3(self):
    
        nTokensInText = len(self.WORDS)
        nWordsFound = 0    
        for iToken in range(0,nTokensInText):
            if self.WORDS[iToken] in self.neutralwdictionary:
                nWordsFound = nWordsFound + 1
           
        NEUTRAL = nWordsFound/float(nTokensInText)
        return NEUTRAL  
            
   
        
    '''
    -----------------------------------------------------------------------------------------------------------------------

    -----------------------------------------------------------------------------------------------------------------------
    '''
    def _morfo_freq(self, part_of_speech):
        if isinstance(part_of_speech, str):
            part_of_speech = [part_of_speech]

        words = []
        for word, tag in self.TAGGED:
            if tag in part_of_speech:
                words.append(word)
        return nltk.FreqDist(words)
    
    '''
    -----------------------------------------------------------------------------------------------------------------------
    MECANISMO PARA CONSTRUÇÃO DE VETOR NUMPY DE CARACTERÍSTICAS (META-ATRIBUTOS)
    -----------------------------------------------------------------------------------------------------------------------
    '''
       
    #@property
    def all_meta_attributes(self, categories):
        vMA = numpy.array([])
#        lMA = []       
        print("C")
#        analise de caracteres (16)
        if categories["C"] == True:
    #        -------------------------------------------------------------------------
            for i in range(1,17):       
                
                vMA = numpy.append(vMA,getattr(self, 'C' + str(i)))
    #            lMA.append(('C' + str(i), vMA[-1]))
        print("W")
#        analise de palavras (13)
        if categories["W"] == True:
#        -------------------------------------------------------------------------
            for i in range(1,13):        
               # listMetrics.append(('f0'+str(i),getattr(metrics, 'f0' + str(i))))
                vMA = numpy.append(vMA,getattr(self, 'W' + str(i)))
    #            lMA.append(('W' + str(i), vMA[-1]))
           
    #        Atributo 13 (vetor com n atributos)
            m = self.W13_N    
            for i in range(0,m.shape[0]):
                vMA = numpy.append(vMA,m[i])            
    #            lMA.append(('W13_' + str(i), vMA[-1]))
        print("TS")    
        if categories["TS"] == True:
#        analise de estrutura do texto (10)  
#        -------------------------------------------------------------------------       
            for i in range(4,6):        
    #           listMetrics.append(('f'+str(i),getattr(metrics, 'f' + str(i))))
                vMA = numpy.append(vMA,getattr(self, 'TS' + str(i)))
    #            lMA.append(('TS' + str(i), vMA[-1]))
        print("TM")  
#        analise morfologica
        if categories["TM"] == True:
#        -------------------------------------------------------------------------
            for i in range(1,7):        
               
                vMA = numpy.append(vMA,getattr(self, 'TM' + str(i)))
     #           lMA.append(('TM' + str(i), vMA[-1]))
        
        print("PC")    
        if categories["PC"] == True:
#       dicionarios psicolinguisticos
    #       ---------------------------------------------------------------------------
            for i in range(1,4):        
               
                vMA = numpy.append(vMA,getattr(self, 'PC' + str(i)))
    #            lMA.append(('PC' + str(i), vMA[-1]))
        
        print("LIWC")
        if categories["LIWC"] == True:     
            m = self.LIWC_N    
            for i in range(0,m.shape[0]):
                vMA = numpy.append(vMA,m[i])          
 
        return vMA
     
    
   