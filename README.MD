How to use:
- Download the zip
- Edit the settings in to_spinda.py if you want (default should be fine for use)
- If you didn't change the default filenames place a "test.mp4" in the res folder
- Run the to_spinda.py (this will take a while. Progress can be seen in console. Change settings (mainly resolution, RESIZE_FACTOR or STEP_MULTIPLIER) if it takes to much time to render)
- Watch your video in spinda! Video will be saved as "test_output.mp4" in res

Modules to install:
- Pillow
- moviepy

Currently the output video has no sound. Sound can be added with ffmpeg:
 ffmpeg -i test_output.mp4 -i audio.mp3 -map 0 -map 1:a -c:v copy -shortest output_with_sound.mp4

Questions? Either over github or comment under my video: https://youtu.be/aprNIWir4aU
