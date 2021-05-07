# VirtualTouch
This repo contains my "Virtual Touch" script. This script is designed to be an adaptable, low cost, and accessible alternative to touch screens in public places. The system is designed to work with a simple webcam and no expensive or proprietary hardware or software. The reason for this is to ensure the system is accessible to any company or retail outlet looking to add an extra level of safety to their customers without investing heavily in commercial systems, and contributing to the E-waste problem by replacing existing systems with complex hardware.

This script depends upon the following python libraries:
cv2,
numpy,
time,
mouse,
ctypes,
math.

The system is designed to segment the region of interest into 2 segments, the hand and background. Because of this, the system will work best with a solid background as long as the colour is constant. It is also recommended that the system is set up such that there is no light coming from either sides or facing the camera. Ideally, the system is used in a well lit area, with a front facing LED panel.

Instructions:
Place your hand, closed fisted with your index finger pointing upward. Direct the cursor with the tip of your finger, and raise your thumb outward to "click". A video example of this can be seen: https://youtu.be/Ux1onjVMbhw

**IMPORTANT**
This script will control your mouse and so it is important to read this before running. The script is initially set to NOT control
the mouse. Running the script as it currently exists, will open 3 windows which can be used to verify the system works for you in your system and environment. If the system works as intended, your fingertip will be correctly detected and a visible red dot will appear on it in the windows. Similarly, clicking will be indicated in the command line if it is being correctly detected. To exit the script, click upon any of the windows and press escape to end the script. It is also worth noting in the event you enable the mouse control, and the system malfunctions for any reason, pressing the windows button and entering task manager will allow you to stop the script as opening task manager disables this script from controlling the mouse.

Uncommenting line 440 will enable mouse control:
#clickTime = UpdateMousePosition(X,Y,click,RatioLength,RatioWidth,clickTime,1)

You may also comment the "imshow" commands in lines 468-470, to remove the windows, however another method of closing the script would need to be added.
