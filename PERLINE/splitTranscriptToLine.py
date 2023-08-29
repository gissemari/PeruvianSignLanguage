import nltk
import sys
import argparse

from os import listdir
from os.path import isfile, join, exists

parser = argparse.ArgumentParser(description='The Embedded Topic Model')
parser.add_argument('--rawScriptPath', type=str, default='./../../Data/Transcripts/all_together/', help='Path where per-line files are located')
parser.add_argument('--inputName', type=str, default='', help='Input File Name')
parser.add_argument('--pathRealLine', type=str, default='./../../Data/Transcripts/per_line/LineAutomated/', help='Path where per-line files are located')

args = parser.parse_args()


rawScriptPath=args.rawScriptPath#'./'
pathRealLine = args.pathRealLine

END_PUNCTUATION = '.?!'


count=0
for fName in listdir(rawScriptPath):
	nameFile = join(rawScriptPath, fName)
	outputName = pathRealLine+fName[:-4]+'_linea.txt'
	


	if exists(outputName):
		print(fName,"\texisted")
		continue

	if isfile(nameFile) and nameFile[-3:]=='txt':
		print(fName, "\tprocessing")
		

		#for line in fileIn:
		try:
			fileIn = open(nameFile,'r', encoding='utf-8')
			stream = fileIn.read()
		except UnicodeDecodeError:

			fileIn = open(nameFile,'r', encoding='iso-8859-1')
			stream = fileIn.read()

		#print(stream)
		fileOutput = open(outputName,'w', encoding='utf-8')
		arrStrings = nltk.sent_tokenize(stream)

		numSentences = len(arrStrings)
		print(numSentences)

		'''
		prevString = ''
		for i in range(numSentences):
			if i < numSentences-1 and arrStrings[i][-1] in END_PUNCTUATION:
				if arrStrings[i+1][0]!=',':
					if prevString!='':
						fileOutput.write(prevString+'\n')
					
				else:
					prevString += arrStrings[i+1]
					if i ==numSentences-1:
						fileOutput.write(prevString+'\n')
						break
			else:
				fileOutput.write(arrStrings[i-1]+'\n')
			prevString = arrStrings[i]
		'''

		prevString = ''#arrStrings[0]
		for i in range(numSentences):
			# if we are in the last sentence
			if i==numSentences-1:
				# and it starts with coma, we combine it with the previous
				if arrStrings[i][0]==',':
					prevString += ' ' + arrStrings[i]
					# and print it if it ends in punctuation
					if arrStrings[i][-1]  in END_PUNCTUATION:
						fileOutput.write(prevString+'\n')
				# and it does not start with a comma, we just print it
				else:
					fileOutput.write(arrStrings[i]+'\n')
			# if we are not in the last sentence
			else:
				# and the next sentence starts with a comma, we save the current sentence in prev
				if arrStrings[i+1][0]==',' or arrStrings[i][-1] not in END_PUNCTUATION:
					prevString += ' ' + arrStrings[i]
				# and if the next sentences is not  comma, we print current sentence and release prev
				else:		
					fileOutput.write(arrStrings[i]+'\n')
					prevString = ''



		fileIn.close()
		fileOutput.close()

	
