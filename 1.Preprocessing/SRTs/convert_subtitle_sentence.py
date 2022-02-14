#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pysrt
import nltk
import sys
import argparse


'''
This code aims to read an SRT generated with the subtitle.to/ tool (which generates an srt file by subtitles) and mix it with the correct transcription.
The transcription with punctuation is in .txt
Output: a transformed SRT, annotated by sentences, rather than by subtitles
'''
parser = argparse.ArgumentParser(description='The Embedded Topic Model')

parser.add_argument('--srtRawPath', type=str, default='./../../Data/AEC/SRT/SRT_raw/', help='Path where SRT raw is located')
parser.add_argument('--inputName', type=str, default='', help='Input File Name')
parser.add_argument('--pathRealLine', type=str, default='./../../Data/AEC/Transcripts/per_line/LineAutomated/', help='Path where per-line files are located')
parser.add_argument('--outputSRT', type=str, default='./../../Data/AEC/SRT/SRT_voice_sentences/', help='Path where to leave output')

args = parser.parse_args()

# Set punctuation that we are looking for
srtRawPath = args.srtRawPath
inputName = args.inputName
pathRealLine = args.pathRealLine
outputSRT = args.outputSRT
END_PUNCTUATION = '.?!'


# Read files and debugging files
srtOriginal = pysrt.open(srtRawPath+inputName+'.srt', encoding='utf-8')#, encoding='iso-8859-1'
fileLine = pathRealLine+inputName+'_linea.txt'
linesIterator = open(fileLine, encoding='utf-8')

srtTransformed = pysrt.SubRipFile()
srtTransformedName = outputSRT+inputName + '.srt'
srtTransformedName1  = outputSRT+inputName + '_trans1.srt'
srtTransformedName2 = outputSRT+inputName + '_trans2.srt'
alertNumLines = open(outputSRT+inputName+"_alerta.txt",'w')


# Number of lines in per-line file should be the same as the number of lines calculated in SRT-raw
numLinesOrig = sum(1 for line in linesIterator)
numSubs = len(srtOriginal)
first_sub = srtOriginal[0]


prevText = ''
prevEnd = first_sub.end
prevStart = first_sub.start
deletedLines = []

lastAdded = 0
lastNewIdx = 0
count = 0
newIndex = 1
case =0
#### srtOriginal[0].index


for line in srtOriginal:

	# For debugging purpuse, print what prevLine and current line values are	
	if count==0:
		print("Initial")
		print(count, prevText, " || " ,line.text,  prevEnd, line.start, line.end)
	elif prevText!='':
		print("Initial")
		print(count, prevText, prevText[-1], " || " ,line.text)
		print(prevEnd, line.start, line.end)


	# Check number of sentences of current line
	arrStrings = nltk.sent_tokenize(line.text)
	numSentences = len(arrStrings)
	numTotalWords = len(nltk.word_tokenize(line.text))
	numFirstWords = len(nltk.word_tokenize(arrStrings[0]))

	duration = (line.end - line.start).seconds#.to_time().to_seconds() 
	ratioTime = float(numFirstWords)/float(numTotalWords)

	# If there is only one sentence or no punctuation marks
	if numSentences==1:
		#Evalute the punctuation of the previous sentence

		case = 1

		## CASO 3: if the previous sentence is the first empty one, ends in punctuation mark or "Musica", do not combine
		if prevText =='' or prevText[-1] in END_PUNCTUATION or prevText[-8:]=='[Música]':
	
			# If the current line is not "Musica", update prevLine
			# Check if lastAddeds should be updated with count or other value

			if line.text.strip() !='[Música]':
				prevStart = line.start
				prevText = line.text
				prevEnd = line.end
				lastAdded = count

			# If current Line is Music, add the line to be deleted later
			else:
				deletedLines.append(count)
				lastAdded = count-1
				if lastAdded>0:
					srtOriginal[lastAdded].start = srtOriginal[lastAdded-1].end
				else:
					srtOriginal[lastAdded].start = srtOriginal[count].end

			case = 3		

		else: ## CASO 1: if the current line should be combine with the previous line

			# Except if the current is "Musica"
			if line.text.strip() =='[Música]':
				deletedLines.append(count)
				#prevText = line.text
				#prevEnd = line.end
				lastAdded = count-1
				case=0

			else:
				#srtOriginal[lastAdded].start = prevStart
				srtOriginal[lastAdded].text += " " + arrStrings[0] # introducing blank space
				srtOriginal[lastAdded].end = line.end#+= {'seconds':ratioTime * duration}

				prevText = srtOriginal[lastAdded].text
				prevEnd = srtOriginal[lastAdded].end

				deletedLines.append(count)
				# We do not update lastAdded

	#Analyze case where more than one sentence in line (2 points?)
	else:#if numSentences>=2:

		
		# Num sentences was>1 then we can find the first punctuation mark
		firstPunctuationIdx = line.text.find(arrStrings[0][-1])

		lineAux = line.text

		### CASO 4: the previus added line ends in a punctuation mark, no combination needed. Extra SRT lines should be created
		if  prevText =='' or prevText[-1] in END_PUNCTUATION :

			prevTimeStart = line.start
			prevTimeEnd = line.start + {'seconds':ratioTime * duration}
			for newSent in range(1,numSentences):

				#Updating text and end from current line
				#print(count, " to srt2 ", lineAux[:firstPunctuationIdx+1])
				print(count, " to srt2 ", arrStrings[newSent-1])
				newIndex = lastAdded+1
				
				#newLine = pysrt.SubRipItem(index=newIndex, start=prevEnd , end=prevEnd, text=lineAux[:firstPunctuationIdx+1])

				
				newLine = pysrt.SubRipItem(index=newIndex, start= prevTimeStart, end=prevTimeEnd, text=arrStrings[newSent-1])
				srtTransformed.append(newLine)
				newIndex+=1

				#print(lineAux,"|", lineAux[firstPunctuationIdx+1:])

				srtOriginal[count].text = arrStrings[newSent]
				srtOriginal[count].start = prevTimeEnd
				numFirstWords = len(nltk.word_tokenize(arrStrings[newSent]))
				duration = (line.end - srtOriginal[count].start).seconds#.to_time().to_seconds() 
				ratioTime = float(numFirstWords)/float(numTotalWords)

				divtime = srtOriginal[count].start + {'seconds':ratioTime * duration}
				srtOriginal[count].end = divtime

				#prevTimeStart

			srtOriginal[count].end = line.end
			prevText = srtOriginal[count].text#lineAux[firstPunctuationIdx+1:] #arrStringsCase[-1]
			prevEnd = srtOriginal[count].end
			
			lastAdded = count
			case = 4

		### CASO 2
		else:

			srtOriginal[lastAdded].text += " " + arrStrings[0] # introducing blank space
			srtOriginal[lastAdded].end += {'seconds':ratioTime * duration}

			srtOriginal[count].start = srtOriginal[lastAdded].end

			numFirstWords = len(nltk.word_tokenize(arrStrings[1]))
			duration = (line.end - srtOriginal[count].start).seconds#.to_time().to_seconds() 
			ratioTime = float(numFirstWords)/float(numTotalWords)


			srtOriginal[count].text = arrStrings[1]## lineAux[firstPunctuationIdx+1:]
			divtime = srtOriginal[count].start + {'seconds':ratioTime * duration}
			srtOriginal[count].end = divtime

			prevTimeStart = srtOriginal[count].start
			prevTimeEnd = srtOriginal[count].end

			for newSent in range(2,numSentences):
				
				#Updating text and end from current line
				print(count, " to srt3 ",arrStrings[newSent-1] )#  lineAux[:firstPunctuationIdx+1]
				newIndex = lastAdded+1
				newLine = pysrt.SubRipItem(index=newIndex, start=prevTimeStart , end=prevTimeEnd, text= arrStrings[newSent-1])
				srtTransformed.append(newLine)
				newIndex+=1


				#print(lineAux,"|", lineAux[firstPunctuationIdx+1:])
				srtOriginal[count].start = srtOriginal[count].end
				srtOriginal[count].text = arrStrings[newSent]## lineAux[firstPunctuationIdx+1:]

				#calculating duration
				numFirstWords = len(nltk.word_tokenize(arrStrings[newSent]))
				duration = (line.end - srtOriginal[count].start).seconds#.to_time().to_seconds() 
				ratioTime = float(numFirstWords)/float(numTotalWords)

				divtime = srtOriginal[count].start + {'seconds':ratioTime * duration}
				srtOriginal[count].end = divtime

				prevTimeStart = srtOriginal[count].start
				prevTimeEnd = srtOriginal[count].end

			srtOriginal[count].end = line.end
			prevText = srtOriginal[count].text

			newIndex+=1

			# No need to update end time
			lastAdded = count

			case = 2


	# For debugging purposes
	if count==0:
		print("End")
		print(count, prevText, " || " ,line.text,  prevEnd, line.start, line.end)
	elif prevText!='':
		print("End")
		print(count, lastAdded, prevText, prevText[-1], " || " ,line.text)
		print(prevEnd, line.start, line.end, "case ", case)

	print("\n")
	
	count+=1
	

### Deleting lines
countDeleted = 0
for i in deletedLines:
	#print(i, len(srtOriginal))
	del srtOriginal[i-countDeleted]
	countDeleted+=1


#for i in srtOriginal:
#	alertNumLines.write(i.text+'\n')



#### UNIFIYING THE SRT OBJECT FROM THE ORIGINAL LIST, THE ADDITIONAL AND THE FINAL
idxOrig = 0
idxAdditional = 0
newIdx  =1
srtFinal = pysrt.SubRipFile()

while idxOrig< len(srtOriginal) or idxAdditional< len(srtTransformed):

	if idxAdditional >= len(srtTransformed):
		srtOriginal[idxOrig].index = newIdx
		srtFinal.append(srtOriginal[idxOrig])
		idxOrig +=1
	elif idxOrig>= len(srtOriginal):#idxAdditional< len(srtTransformed):
		srtTransformed[idxAdditional].index = newIdx
		srtFinal.append(srtTransformed[idxAdditional])
		idxAdditional +=1
	else:
		lineOrig = srtOriginal[idxOrig].index
		lineAddi = srtTransformed[idxAdditional].index# + srtOriginal[0].index

		if lineOrig<=lineAddi:
			srtOriginal[idxOrig].index = newIdx
			srtFinal.append(srtOriginal[idxOrig])
			idxOrig +=1
		else:
			srtTransformed[idxAdditional].index = newIdx
			srtFinal.append(srtTransformed[idxAdditional])
			idxAdditional +=1

	newIdx +=1


numLinesFinal = len(srtFinal)
print(numLinesOrig,numLinesFinal)
linesIterator.seek(0)


### COPYING THE LINES FROM THE CORRECT TRANSCRIPT

if numLinesOrig == numLinesFinal:
	print("Same number of lines")

	count = 0
	for  line in linesIterator:
		if count<2:
			print(line)
		srtFinal[count].text = line#.encode(encoding='UTF-8')#,errors='strict'
		alertNumLines.write(srtFinal[count].text)
		count+=1

else:
	print("Different number of lines")
	for  srtObject in srtFinal:
		#srtFinal[count].text = line
		alertNumLines.write(srtObject.text+'\n')

### LAST CHECK ON TIME



### Save new SRT
srtOriginal.save(srtTransformedName1, encoding='utf-8')
srtTransformed.save(srtTransformedName2, encoding='utf-8')
srtFinal.save(srtTransformedName, encoding='utf-8')
