
python 2.Segmentation/segmentPerSRT.py --rawVideoPath /home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/Videos/RawVideo/ --srtPath /home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/SRT/SRT_gestures/ --outputVideoPath /home/gissella/Documents/Research/SignLanguage/PeruvianSignLanguaje/Data/Videos/Segmented_gestures/ --flgGesture 1

python 3.Translation/FrameToKeypoint/ConvertVideoToKeypoint.py --holistic

echo "Press 'q' to exit"
count=0
while : ; do
read -n 1 k <&1
if [[ $k = q ]] ; then
printf "\nQuitting from the program\n"
break
else
((count=$count+1))
printf "\nIterate for $count times\n"
echo "Press 'q' to exit"
fi
done