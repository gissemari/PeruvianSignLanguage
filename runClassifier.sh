#mkdir Logs

cp -r models 5.Run
cp -r utils 5.Run
python 5.Run/rnn_classifier.py --pose --hands --face
rm -r 5.Run/utils
rm -r 5.Run/models

echo "Press ENTER to exit:"
read ENTER
