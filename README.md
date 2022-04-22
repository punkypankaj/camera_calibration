# Raspberry-pi-Camera-Calibration--PYTHON



# Steps to setup and calibrate the pi camera or any camera( Basically for Pin hole type of cameras)

1. enable the camera option in rpi config 

   sudo raspi-config


now reboot


Note : getting pi camera to work with mjpeg sreamer is optional, using mjpeg streamer you just get the camera feed working and you can watch the video on th web browser.




  
2. now download the mjpeg streamer
  git clone https://github.com/jacksonliam/mjpg-streamer


3. now cmake the mjpeg streamer
   git clone https://github.com/jacksonliam/mjpg-streamer


4.now get in the mjpeg stream at the home location

   cd mjpg-streamer
   cd mjpg-streamer-experimental

5 now run the make command 

\\  make
    sudo make install

6. now lets check the camera is working or not so run the command
   get back to root folder

   mjpg_streamer -o "output_http.so -w ./www" -i "input_raspicam.so"


7 now download all the dronekit mavlink mavproxy using pi
next is to install imutils which will  help to increase the speed up of the python script maily with the camera frame rates.
( actually imutil is required for the camera components like to getthe camera vectors).

sudo pip install imutils


8. now open any editor like vi or nano im using nano

sudo nano /usr/local/lib/python2.7/dist-packages/imutils/video/webcamvideostream.py
 after the file executes now  we need to change and add few things

add the resolution

        def __init__(self, src=0, name="WebcamVideoStream", width=640, height=480):

now also add

     self.stream = cv2.VideoCapture(src)
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                (self.grabbed, self.frame) = self.stream.read()

save and exit


9. now install open cv numpy

sudo pip install python-opencv python-numpy



10. now measure the size of the single chess board block in mm


after this run run the caliberate.py

python calibration.py --mm 22 --width 640 --height 480


22. is the size of the marker block  640 480 are strictly adviced with use with this particulat setup



USE THIS COMMAND TO TURN OFFTHE COLOUR AUTO CORRECTION IF USING RPI NOIR V2.1 WITH TINT PURPLE IMAGES

  sudo vcdbg set awb_mode 0

update the setup with this command every time the rpi reboots 







 
