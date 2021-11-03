
#mkdir Logs

cp -r utils 5.Run
python 5.Run/rnn_classifier.py
rm -r 5.Run/utils

echo "Press ENTER to exit:"
read ENTER
