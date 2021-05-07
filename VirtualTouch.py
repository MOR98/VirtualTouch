#----------------------------------------------------------------------------------------
#
#Libraries
#
#----------------------------------------------------------------------------------------
import cv2
import numpy as np
import time
import mouse
import ctypes
import math as m

#----------------------------------------------------------------------------------------
#
#Init/Globals
#
#----------------------------------------------------------------------------------------

user32 = ctypes.windll.user32
ScreenWidth,ScreenHeigth = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
AspectRatio = ScreenWidth/ScreenHeigth
AspectRatio = m.ceil(AspectRatio)
xDiff = 200

#----------------------------------------------------------------------------------------
#
#These are the criteria for the K means algorithim, such that it iterates 1000 times 
#with accuracy 0.01 and uses 2 clusters
#
#----------------------------------------------------------------------------------------
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.01)
k = 2

#----------------------------------------------------------------------------------------
#
#These global lists are used to store the finger count, X and Y points across multiple
#iterations of the loop
#
#----------------------------------------------------------------------------------------

FingerCountHistory = [0,0,0]
XHistory = [0,0,0,0,0,0,0,0,0,0,0]
YHistory = [0,0,0,0,0,0,0,0,0,0,0]

#----------------------------------------------------------------------------------------
#
#Set an initial value to the clickTime variable
#
#----------------------------------------------------------------------------------------

clickTime = time.time()
print(clickTime)

#----------------------------------------------------------------------------------------
#define the camera to be the first camera device seen by the computer
#
#----------------------------------------------------------------------------------------

camera = cv2.VideoCapture(0)


def DefinePointingArea(img,Ratio,L,W,scale):
	#------------------------------------------------------------------------------------
	#
	#This function takes in the image, the aspect ratio of the screen, the dimensions of
	#the screen, and the user set scale to crop the image as desired.
	#
	#------------------------------------------------------------------------------------

	#------------------------------------------------------------------------------------
	#
	#First get the image dimensions, crop the image using the ratio between the camera 
	#and the screen, and the scale, before returning the ROI and the ratios.
	#
	#------------------------------------------------------------------------------------

	Y, X = img.shape[:2]
	xL = m.ceil(X/scale)
	yL = m.ceil(xL/AspectRatio)
	Rl = m.ceil(L/xL)
	Rw = m.ceil(W/yL)
	img = img[0:yL+xDiff,X-xL-50:X]
	return img,Rl,Rw,xL,yL

def KmeansThreshold(ROI):
	#------------------------------------------------------------------------------------
	#
	#This function takes in the ROI and returns the contour, the segmented image, the 
	#centre of the hand and a boolean value as to whether there is a hand or not
	#
	#------------------------------------------------------------------------------------

	#------------------------------------------------------------------------------------
	#Perform set up and preprocessing upon the input image
	#
	#------------------------------------------------------------------------------------
	
	image = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
	pixel_values = image.reshape((-1, 3))
	pixel_values = np.float32(pixel_values)


	#------------------------------------------------------------------------------------
	#
	#Perform the k means algorithim upon the data
	#
	#------------------------------------------------------------------------------------

	ret,labels,(centers)=cv2.kmeans(pixel_values,
									k,
									None,
									criteria,
									10,
									cv2.KMEANS_RANDOM_CENTERS)	


	#------------------------------------------------------------------------------------
	#
	#Order the 2 colours by the value of the first element, this is to keep the 
	#segmented region consistent each loop as the first and second value could take
	#on either contour.
	#
	#------------------------------------------------------------------------------------
	centers = np.uint8(centers)
	tempa = centers[0]
	tempb = centers[1]
	if(centers[0,0]>centers[1,0]):
		tempa = centers[1]
		tempb = centers[0]
	

	#------------------------------------------------------------------------------------
	#
	#Set upper and lower limits for the HSV image
	#
	#------------------------------------------------------------------------------------

	hl,sl,vl = (tempa[0]-1), (tempa[1]-1), (tempa[2]-1)
	hu,su,vu = (tempa[0]+1), (tempa[1]+1), (tempa[2]+1)
	lower = np.array([hl,sl,vl], dtype = "uint8")
	upper = np.array([hu,su,vu], dtype = "uint8")

	#------------------------------------------------------------------------------------
	#
	#Create a segmented image from the data collected from k means algorithim
	#
	#------------------------------------------------------------------------------------
	labels = labels.flatten()
	segmented_image = centers[labels.flatten()]
	segmented_image = segmented_image.reshape(image.shape)

	#------------------------------------------------------------------------------------
	#
	#Threshold the segmented image based on the upper and lower limit set, and attempt
	# to find contours in this image
	#
	#------------------------------------------------------------------------------------

	Skin = cv2.inRange(segmented_image, lower, upper)
	Skin = cv2.blur(Skin, (2,2))
	contours, hierarchy = cv2.findContours(Skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


	#------------------------------------------------------------------------------------
	#If contours exist in this image, we thresholded the correct cluster and proceed
	#
	#------------------------------------------------------------------------------------

	if contours:

		#--------------------------------------------------------------------------------
		#
		#Take the largest contour, create and draw the hull 
		#(2 are needed as one cannot be visualised)
		#
		#--------------------------------------------------------------------------------

		contours = max(contours, key=lambda x: cv2.contourArea(x))

		hull = cv2.convexHull(contours,returnPoints = False)
		hullt = cv2.convexHull(contours)
		cv2.drawContours(segmented_image, [hullt], -1, (0, 255, 255), -1)
		cv2.drawContours(segmented_image, [contours], -1, (255,255,0), 2)


	
	else:

		#--------------------------------------------------------------------------------
		#
		#If there was no contour, swap the limits and attempt to threshold again.
		#
		#--------------------------------------------------------------------------------
		tempa = centers[0]
		tempb = centers[1]
		if(centers[0,0]<centers[1,0]):
			tempa = centers[1]
			tempb = centers[0]

		rl,gl,bl = (tempa[0]-1), (tempa[1]-1), (tempa[2]-1)
		ru,gu,bu = (tempa[0]+1), (tempa[1]+1), (tempa[2]+1)
		
		lower = np.array([rl,gl,bl], dtype = "uint8")
		upper = np.array([ru,gu,bu], dtype = "uint8")

		#--------------------------------------------------------------------------------
		#
		#Create a segmented image from the data collected from k means algorithim
		#
		#--------------------------------------------------------------------------------
		labels = labels.flatten()
		segmented_image = centers[labels.flatten()]
		segmented_image = segmented_image.reshape(image.shape)

		#--------------------------------------------------------------------------------
		#
		#Threshold the segmented image based on the upper and lower limit set, and  
		#attempt to find contours in this image
		#
		#--------------------------------------------------------------------------------

		Skin = cv2.inRange(segmented_image, lower, upper)
		Skin = cv2.blur(Skin, (5,5))
		contours, hierarchy = cv2.findContours(Skin, 
											   cv2.RETR_TREE,
											   cv2.CHAIN_APPROX_NONE)

		if contours:
			#----------------------------------------------------------------------------
			#
			#Take the largest contour, create and draw the hull 
			#(2 are needed as one cannot be visualised)
			#
			#----------------------------------------------------------------------------
			contours = max(contours, key=lambda x: cv2.contourArea(x))
			hull = cv2.convexHull(contours,returnPoints = False)
			hullt = cv2.convexHull(contours)
			cv2.drawContours(segmented_image, [hullt], -1, (0, 255, 255), -1)
			cv2.drawContours(segmented_image, [contours], -1, (255,255,0), 2)
		else:
			#----------------------------------------------------------------------------
			#
			#At this point neither methods worked, the camera may be covered or 
			#obstructed, return empty
			#
			#----------------------------------------------------------------------------
			return 0,0,0,0


	#------------------------------------------------------------------------------------
	#
	#Check if the area of the "hand" is greater than 1/3 of the image, in which case 
	#assume it is not the hand	
	#
	#------------------------------------------------------------------------------------
	Y, X = ROI.shape[:2]
	areaROI = X*Y
	areaContour = cv2.contourArea(contours)
	handInFrame = True 
	if(areaContour > (areaROI/3)):
		handInFrame = False

	return contours,segmented_image,hull,handInFrame

def DefectDetermination(contours,img,hull,display):
	#------------------------------------------------------------------------------------
	#
	#Attempt to find convexity defects in the image
	#
	#------------------------------------------------------------------------------------
	if(contours is not None and hull is not None):
		try:
			defects = cv2.convexityDefects(contours,hull)
		except:
			return 0,0,0
		

		if defects is not None:
			cnt = 0
			#----------------------------------------------------------------------------
			#
			#Unpack and Check each "defect" and check if they are convex, if they 
			#are convex, they are assumed to be a finger, increment counter
			#
			#----------------------------------------------------------------------------
			for i in range(defects.shape[0]):
				s, e, f, d = defects[i][0]
				start = tuple(contours[s][0])
				end = tuple(contours[e][0])
				far = tuple(contours[f][0])
				#------------------------------------------------------------------------
				#
				#Determine if convex using the cosine rule
				#
				#------------------------------------------------------------------------
				a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
				b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
				c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
				angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
				if angle <= np.pi / 2:  
				    cnt += 1
				#------------------------------------------------------------------------
				#
				#The top most point of the contour is assumed to be the finger tip
				#
				#------------------------------------------------------------------------
				X,Y = tuple(contours[contours[:, :, 1].argmin()][0])
				
			if cnt > 0:
				cnt = cnt+1

			return cnt,X,Y
		else:
			return 0,0,0
	else:
		return 0,0,0

def AveragePoint(X,Y,fingers):
	#------------------------------------------------------------------------------------
	#
	#This function creates the average value for the finger point, as well as updating 
	#the finger count#history list.	
	#
	#------------------------------------------------------------------------------------

	XHistory[1:] = XHistory[0:10]
	XHistory[0] = X
	YHistory[1:] = YHistory[0:10]
	YHistory[0] = Y

	FingerCountHistory[1:] = FingerCountHistory[0:2]
	FingerCountHistory[0] = fingers

	tempx = 0
	tempy = 0
	i = 0
	for i in range(0,5):
		tempx = tempx + XHistory[i]
		tempy = tempy + YHistory[i]

	meanX = int(tempx/5)
	meanY = int(tempy/5)
	click = 1
	#------------------------------------------------------------------------------------
	#
	#If any of the 3 counts were not 2, it is not assumed to be a click
	#
	#------------------------------------------------------------------------------------
	for count in range(0,len(FingerCountHistory)):
		if(FingerCountHistory[count] != 2):
			click = 0
			break
		
	return meanX,meanY,click



def UpdateMousePosition(X,Y,click,Rl,Rw,clickTime,roundFactor):
	#------------------------------------------------------------------------------------
	#
	#Correct the X,Y coordinates based off of the ratio of the ROI to the 
	#screen, and the roundFactor,move and Click as needed
	#
	#------------------------------------------------------------------------------------
	
	X = (round((X*Rl)/roundFactor))*roundFactor
	Y = (round((Y*Rw)/roundFactor))*roundFactor
	
	mouse.move(X-xDiff, Y, absolute=True, duration=0)

	#------------------------------------------------------------------------------------
	#
	#If the user has not clicked in the previous 2 seconds,and the indicator 
	#is to click, click!
	#
	#------------------------------------------------------------------------------------
	timeSinceClick = time.time() - clickTime
	if(click and timeSinceClick>1):
		mouse.click(button = "left")
		clickTime = time.time()
	return clickTime


#----------------------------------------------------------------------------------------
#
#Set up for the main loop with a loopcounter and initializing some other variables.
#
#----------------------------------------------------------------------------------------
loops = 0
prevTestTime = time.time()
FPS = 0
loopTime = time.time()
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

while True: 

	#------------------------------------------------------------------------------------
	#
	#Capture the image, flip to counter the mirror effect, and grab 
	#some basic info on the image size
	#
	#------------------------------------------------------------------------------------
	ret, inputImage = camera.read()
	inputImage = cv2.flip(inputImage,1)
	frameHeight, frameWidth = inputImage.shape[:2]

	if (inputImage is not None):
		#--------------------------------------------------------------------------------
		#
		#If there is an image, create the ROI, threshold the ROI, finding the contours
		#
		#--------------------------------------------------------------------------------
		loops = loops+1
		ROI,RatioWidth,RatioLength,ROIWidth,ROIHeight = DefinePointingArea(inputImage,
																		   AspectRatio,
																		   ScreenWidth,
																		   ScreenHeigth,
																		   scale = 2)
		contours, segmented_image, hull,handInFrame = KmeansThreshold(ROI)

		if(handInFrame):
			#----------------------------------------------------------------------------
			#
			#If there is hand in the image, find the finger tip, and count the fingers
			#
			#----------------------------------------------------------------------------
			NoFingers, X,Y = DefectDetermination(contours,ROI,hull,1)
			X,Y,click = AveragePoint(X,Y,NoFingers)	
			cv2.circle(ROI,(X,Y),10,[0,0,255],-1)

			#----------------------------------------------------------------------------
			#
			#Update the mouse position as appropriate
			#
			#----------------------------------------------------------------------------
			#clickTime = UpdateMousePosition(X,Y,click,RatioLength,RatioWidth,clickTime,1)
			if(click):
				print("Click")
			
	   	#--------------------------------------------------------------------------------
		#
		#Check and display FPS
		#
		#--------------------------------------------------------------------------------
		if (loops == 10):
			print("_________")
			loops = 0
			TestTime = time.time()
			FPS = 1/((TestTime - prevTestTime)/10)
			prevTestTime = TestTime
		cv2.putText(inputImage,
					"{:.2f}".format(FPS), 
					bottomLeftCornerOfText,
					font,fontScale,
					fontColor,
					lineType)

	
		#--------------------------------------------------------------------------------
		#
		#Display the frames as appropriate
		#
		#--------------------------------------------------------------------------------
		cv2.imshow("K Means Contour",segmented_image)
		cv2.imshow("ROI",ROI)
		cv2.imshow("Image", inputImage)


	#------------------------------------------------------------------------------------
	#
	#If the user hits escape end the program
	#
	#------------------------------------------------------------------------------------
	if cv2.waitKey(1) == 27:
	   		break

	#------------------------------------------------------------------------------------
	#
	#Busy wait to ensure 10FPS
	#
	#------------------------------------------------------------------------------------
	while(time.time() < (loopTime + 0.1)):
		dummy = 0
	loopTime = time.time()

#----------------------------------------------------------------------------------------
#
#IUpon the end of the script, close the windows and release the camera from use
#
#----------------------------------------------------------------------------------------
camera.release()
cv2.destroyAllWindows()