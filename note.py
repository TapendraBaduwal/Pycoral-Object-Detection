#https://github.com/TannerGilbert/Google-Coral-Edge-TPU/blob/master/pycoral_object_detection.py
#https://coral.ai/docs/accelerator/get-started#requirements
#Install the PyCoral API
sudo apt-get update

sudo apt-get install python3-pycoral

#Add our Debian package repository to your system:

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update

#Install the Edge TPU runtime:

sudo apt-get install libedgetpu1-std

#Install the PyCoral library
sudo apt-get install python3-pycoral

#Finally Run code
python3 -m object_detection 
or 
sudo python3 -m object_detection

#sudo permission.... sudo hanna pardaina 
sudo usermod -aG plugdev $USER
sudo reboot now

###### Single Shot Detector Algorithm #########
1.SSD framework used VGG-16(Visual Geometry Group) as the base network
in which 16  convolutional layers + Relu layer +pooling layers 
 finally we get convolution feature map metrix.

2.After getting convolution feature map metrix from above,
we pass this matrix to next two or more convolution layer with multiscale,
and pass on  fully connected classification layer  to detect object.

3.Boundary boxes baunda image frame matrix  ma object or face vako thau ma different pixel
hunxan so teyo thau ma cluster banxa ra cluster ko centroide ko ori pari Boundary boxes
creat garinxa.