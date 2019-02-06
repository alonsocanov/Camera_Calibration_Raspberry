import numpy as np
import cv2
import glob


print("Camera Potos PC program is to take a videowith a preview")
cv2.namedWindow('Preview')
live_vid = cv2.VideoCapture(0)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.mp4',fourcc, 20.0, (640,480))

if live_vid.isOpened(): # try to get the first frame
    rval, frame = live_vid.read()
else:
    rval = False

while rval:
    cv2.imshow("Preview", frame)
    rval, frame = live_vid.read()

    out.write(frame)
    key = cv2.waitKey(20) & 0xFF
    # exit on ESC
    if key == ord('q'):
        break
live_vid.release()
cv2.destroyWindow("Preview")
print('End of Camera Phtos PC')