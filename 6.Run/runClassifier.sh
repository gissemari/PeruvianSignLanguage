#mkdir Logs

cp -r models 6.Run
cp -r utils 6.Run
python 6.Run/rnn_classifier.py --pose --hands --face
rm -r 6.Run/utils
rm -r 6.Run/models

echo "Press ENTER to exit:"
read ENTER
