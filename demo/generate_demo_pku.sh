vidname=$1
path='/data/stars/share/PKU-MMD/original_videos/'
FPS=30
savepath='/data/stars/user/sasharma/PKU_poseattnet/demo/original_frames/'

# extract frames from .avi file.
mkdir -p $savepath$vidname
ffmpeg -loglevel panic -i $path$vidname'.avi' -start_number 0 -vf scale=640:480 -q:v 1 -r $FPS $savepath$vidname'/%5d.jpg'
echo "Frames have been extracted for "$vidname

# create demo video
python ./generate_video_pku.py --videoname $vidname
echo "Frames have been modified to include annotations"

ffmpeg -r $FPS -i 'modified_frames/'$vidname'/%5d.jpg' -vb 20M 'Predicted_'$vidname'.mp4'
echo "Demo video is generated!!"
