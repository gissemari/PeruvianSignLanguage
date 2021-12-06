rm data/*.csv
rm project/*.csv
echo "csv removed"
rm -r project/data/kp/train/*
rm -r project/data/kp/val/*
rm -r project/data/kp/test/*
echo "keypoints removed"
rm -r project/data/mp4/test/*
rm -r project/data/mp4/val/*
rm -r project/data/mp4/train/*
echo "videos removed"
rm -r project/data/kpflow2/train/*
rm -r project/data/kpflow2/test/*
rm -r project/data/kpflow2/val/*
echo "keypoint flow removed"
