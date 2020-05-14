import os
cmd = "ffmpeg -f concat -safe 0 -i info.txt -c copy 2_of_each.avi"
os.system(cmd)
