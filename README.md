# VIDEO LINK!!!
https://youtu.be/EBr9OlxyP9c

# Raspberry-pi-Camera-Calibration--PYTHON



# Steps to setup and calibrate the pi camera or any camera( Basically for Pin hole type of cameras)

1. Enable the camera option in rpi config 

   $ sudo raspi-config


Now reboot


Note : Getting pi camera to work with mjpeg sreamer is optional, using mjpeg streamer you just get the camera feed working and you can watch the video on th web browser. In case if you dont want to use mjpeg streamer you can just get the camera configured and additional settings in the calibration code to complete the calibration process





2. Download the mjpeg streamer

     $ git clone https://github.com/jacksonliam/mjpg-streamer


3. Now cmake the mjpeg streamer

     $ git clone https://github.com/jacksonliam/mjpg-streamer


4. Now get in the mjpeg stream at the home location

     $ cd mjpg-streamer
     $ cd mjpg-streamer-experimental

5. Now run the make command 

     $ make

     $ sudo make install

6. Lets check the camera is working or not so run the command
   get back to root folder

     $ mjpg_streamer -o "output_http.so -w ./www" -i "input_raspicam.so"
     
To watch the video stream on the browser use the link

     $ http://127.0.0.1:8080/?action=stream
     
7. Download all the dronekit mavlink mavproxy over pi
next is to install imutils which will  help to increase the speed up of the python script mainly with the camera frame rates.
(actually imutil is required for the camera components like to get the camera vectors).

     $ sudo pip install imutils


8. Now open any editor like vi or nano 

     $ sudo nano /usr/local/lib/python2.7/dist-packages/imutils/video/webcamvideostream.py
     
 After the file executes, we need to change and add few things

Add the resolution:

        def __init__(self, src=0, name="WebcamVideoStream", width=640, height=480):

Now also add:

     self.stream = cv2.VideoCapture(src)
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                (self.grabbed, self.frame) = self.stream.read()

Save and exit


9. Istall open cv numpy

     $ sudo pip install python-opencv python-numpy



10. Measure the size of the single chess board block in mm


After this run the caliberate.py executable file

    $ python calibrate.py --mm 26 --width 640 --height 480


'26' is he size of the marker block  640 480 are strictly adviced to use with this particular setup



USE THIS COMMAND TO TURN OFF THE COLOUR AUTO CORRECTION IF USING RPI NOIR V2.1 WITH TINT PURPLE IMAGES

    $ sudo vcdbg set awb_mode 0

Update the setup with this command every time the rpi reboots 

# GREAT YOU GOT IT DONE!!!







 
