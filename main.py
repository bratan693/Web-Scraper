import io
import sys
import os
from urllib.request import urlopen
import collections
from bs4 import BeautifulSoup
import requests
import json
import pickle
import index_constructor    # import invertedIndex


if __name__ == "__main__":
    invIndex = index_constructor.invertedIndex()
    invIndexDictionary = collections.defaultdict(dict)
    bonusWeightDictionary = collections.defaultdict(dict)
    basedir = "WEBPAGES_RAW/"
    library = "WEBPAGES_RAW/bookkeeping.json"
    wordDir = 'InvertedIndexDictionary.txt'

    
    wordIndex = open(wordDir, "r")
    # if the file containing the parsed words is empty
    if os.stat(wordDir).st_size < 10:
        print("File empty, proceeding to parsing docs")
        with open(library) as f:
            data = json.load(f)
        for key in data:
                filetoSearch = os.path.join(basedir, key)
                invIndex.parseWords(filetoSearch, key, invIndexDictionary, data[key],bonusWeightDictionary)
        wordIndex.close()
        file = open("InvertedIndexDictionary.txt", "w", encoding='utf-8')
        for key, value in invIndexDictionary.items():
            count = 0
            file.write("Word: {}, \n".format(key))
            for frequency in value:
                if(count == 0):
                    file.write("docid : {} \n".format(invIndexDictionary[key]))#invIndexDictionary[key][frequency]))
                    count+=1
                else: 
                    count+=1
        file.close()
        
        invIndex.pickleAllWords(invIndexDictionary)
        invIndex.pickleBonusWeights(bonusWeightDictionary)
        invIndex.calculateTF() #calculate TF 
        invIndex.calculateIDF()
        invIndex.createWeight()
        invIndex.getUrl()
        invIndex.calculateBonusWeights()
    else:  
        query = input("Query: ")
        while True:
            invIndex.getQuery(query)                     

