from google_trans_new import google_translator
#only 100 requests per hour, so coreNLP
import argparse
import pysrt

parser = argparse.ArgumentParser(description='Translating SRTs')

parser.add_argument('--srtRawPath', type=str, default='./../Data/SRT/SRT_gestures/', help='Path where the original SRT is located')
parser.add_argument('--inputName', type=str, default='', help='Input File Name')

args = parser.parse_args()

# Set punctuation that we are looking for
srtRawPath = args.srtRawPath
inputName = args.inputName


srtOriginal = pysrt.open(srtRawPath+inputName+'.srt', encoding='utf-8')#, encoding='iso-8859-1'

srtTranslated = pysrt.SubRipFile()
srtTranslatedName = srtRawPath+'en2_'+inputName + '.srt'

translator = google_translator()

for line in srtOriginal:
    newIndex = line.index
    start = line.start
    end = line.end
    sentence = line.text
    translate_text = translator.translate(sentence,lang_src='es', lang_tgt='en') 
    # if json decode error: https://stackoverflow.com/questions/68214591/python-google-trans-new-translate-raises-error-jsondecodeerror-extra-data 

    newLine = pysrt.SubRipItem(index=newIndex, start= start, end=end, text=translate_text)
    srtTranslated.append(newLine)

srtTranslated.save(srtTranslatedName, encoding='utf-8')
