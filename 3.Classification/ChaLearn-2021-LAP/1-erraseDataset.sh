rm data/train*.csv
rm data/val*.csv
rm project/train*.csv
echo "csv removed"
rm -r project/data/kp/train/*.*
rm -r project/data/kp/val/*.*
echo "keypoints removed"
rm -r project/data/mp4/val/*.*
rm -r project/data/mp4/val/*
rm -r project/data/mp4/train/*.*
rm -r project/data/mp4/train/*
echo "videos removed"
rm -r project/data/kpflow2/train/*.*
rm -r project/data/kpflow2/val/*.*
echo "keypoint flow removed"
