#### READING SRT
- I kept getting errors for reading saying that the encoding type was not working or failt to read certain byte
- There was no encoding to transform correctly. So I thought I can use a hard transformation and replacement: ú \xfa
- It seems like the error was caused by the pysrt version. I install it with pip instead of conda and it worked
- Some technical issues to read SRT:
	no encoding u'[M\xfasica]'
	iso-8859-1 u'[M\xc3\xbasica]'
	utf8 same
	'utf7' Error: codec can't decode byte 0xc3
	encoding='latin-2'
	The error was the pysrt version (fro conda to pip to install pysrt)

- Some technical issues to read per line files:
	historietas, porcentajes  en 'iso-8859-1'
	ira_alegria en 'utf-8'
- MISSING: add a way to recognize encoding of each file o try several, because I guess due to different machines of volunteers, it si given erros for some files with one enconding and other errors with other encodings for another files.


#### TOKENIZE SENTENCES
- First I thought I had to used function find to find every punctuation mark top split the
- However, I remembered that we can do the trick with the function sent_tokenize() in NLTK library
- Fortuately, sent_tokenize*() does not count a different sentence if exclamation or question mark comes before a comma.
- Anyways we still need END_PUNCTUATION set to verify if the last character of a sentence has it. When we are evaluating to combine sentences.
- A point after a number is not considered as the end of a sentence (4.), however a point and exclamation after % it is.
- Same number of lines: number of lines read in per-line file should be the same as the number of lines calculated from SRT-raw. Some more check could be done by count mark points, ignoring the ones that do not count for sentence '?,' with stringPipe.count('.')



#### CASES
- Structures:
1) srtOriginal
2) srtAdditional
3) srtFinal

- We figure out there exist 4 main cases for which to treat differentely the updates in the variables:
1) The previous sentences does not end with punctuation so it can accept the next line, that consist in only on sentnce. The current object in SRT list is deleted
2) The previous sentences does not end with punctuation so it can accept the next line, however the current sentence consist of more than one sentence. So we split the current sentence, combine the first one with the previous one. Then, update the current one with only the last sentence. The sentences in the middle are saved in the ADDITIONAL list.
3) The previous sentences does end with punctuation so it can NOT accept the next line. Current line consist of only one sentence and remains the same
4) The previous sentences does end with punctuation so it can NOT accept the next line. Current line consist of more than one sentence. So we split the current sentence, combine the first one with the previous one. Then, update the current one with only the last sentence. The sentences in the middle are saved in the ADDITIONAL list.


#### OUTPUT
- Check if number of lines to print is the same as the number of lines in the [file]_lines.txt from DRIVE.
- A SRT file with the proper punctuation marks and replaced lines with correct transcription.


#### RUN
Currently version 3 is the one that takes into account the latest updates

	python convert_subtitle_sentence.py --inputName historietas

- The first parameter is the name of the file
- The seond parameter is the path from where to get the correct transcript per line