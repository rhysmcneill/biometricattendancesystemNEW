import os
import pickle
from datetime import datetime, time

import dlib
import numpy as np
from django.shortcuts import render, redirect
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import threading
from sklearn.preprocessing import LabelEncoder
from django.views.decorators.http import condition
from sklearn.svm._libsvm import predict

from .forms import UserRegForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
from django.views.decorators import gzip
import threading
import io
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import matplotlib
# Create your views here.

def username_present(username):
    if User.objects.filter(username=username).exists():
        return True

    return False

# @register(request) takes a form request and ensures it is POST method
@login_required
def register(request):
    if request.method == 'POST':
        form = UserRegForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}')
            return redirect('/home')
    else:
        form = UserRegForm()
    return render(request, "users/register.html", {'form': form })

def loginUser(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user.is_active:
            # Redirecting to the required login according to user status.
            if user.is_superuser or user.is_staff:
                login(request, user)
                return redirect('/contactus')  # or your url name
            else:
                login(request, user)
                return redirect('/howitworks')
    else:
        return render(request, 'users/login.html', {})
# Login required to access employee dashboard
@login_required
def dashboard(request):
    return render(request, 'users/dashboard.html')

# Login required to access admin dashboard
@login_required
def admindashboard(request):
    return render(request, 'users/adminDashboard.html')


def logout_view(request):
    logout(request)
    return redirect('/home')

class mycamera(object):

    def __init__(self):
        self.frames = cv2.VideoCapture(0)

    def __del__(self):
        self.frames.release()

    def get_jpg_frame(self):
        is_captured, frame = self.frames.read()
        retval, jframe = cv2.imencode('.jpg', frame)
        return jframe.tobytes()

def livefeed():
    camera_object = mycamera()
    while True:
        jframe_bytes = camera_object.get_jpg_frame()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + jframe_bytes + b'\r\n\r\n')

@login_required
@condition(etag_func=None)
def display_livefeed(self):
     return StreamingHttpResponse(
            livefeed(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )


def create_dataset(username):
    id = username
    if (os.path.exists('image_data/Train_Data/{}/'.format(id)) == False):
        os.makedirs('image_data/Train_Data/{}/'.format(id))
    directory = 'image_data/Train_Data/{}/'.format(id)

    # Detect face
    # Loading the HOG face detector and the shape predictpr for allignment

    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()

    # capture images from the webcam and process and detect the face
    # Initialize the video stream
    print("[INFO] Initializing Video stream")
    vs = cv2.VideoCapture(0)
    # time.sleep(2.0) ####CHECK######

    # Our identifier
    # We will put the id here and we will store the id with a face, so that later we can identify whose face it is

    # Our dataset naming counter
    sampleNum = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while (True):
        # Capturing the image
        # vs.read each frame
        _, img = vs.read()
        # Resize each image
        img = imutils.resize(img, width=1000)
        # the returned img is a colored image but for the classifier to work we need a greyscale image
        # to convert
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # To store the faces
        # This will detect all the images in the current frame, and it will return the coordinates of the faces
        # Takes in image and some other parameter for accurate result
        faces = detector(gray, 0)
        # In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        print("[INFO] Adding face detection images...")
        for (x, y, w, h) in faces:


            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            sampleNum = sampleNum + 1
            # Saving the image dataset, but only the face part, cropping the rest

            if (x, y, w, h) is None:
                print("face is none")
                continue

            cv2.imwrite(directory + '/' + str(sampleNum) + '.jpg', img)
            # cv2.imshow("Image Captured",face_aligned)
            # @params the initial point of the rectangle will be x,y and
            # @params end point will be x+width and y+height
            # @params along with color of the rectangle
            # @params thickness of the rectangle

            # Before continuing to the next loop, I want to give it a little pause
            # waitKey of 100 millisecond
            cv2.waitKey(50)

        # Showing the image in another window
        # Creates a window with window name "Face" and with the image img
        cv2.imshow("Add Images", img)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        cv2.waitKey(1)
        # To get out of the loop
        if (sampleNum > 300):
            break

    # Stoping the videostream
    # destroying all the windows
    cv2.destroyAllWindows()


@login_required
def add_photos(request):
    if not request.user.is_authenticated:
        messages.warning(request, f'Employee not authorised')
        return redirect('dashboard')
    elif request.user.is_authenticated:
        user = request.user
        create_dataset(user)
        messages.success(request, f'Photos added successfully')
        return redirect('dashboard')
    else:
        messages.warning(request, f'Error: dataset not created')
        return render(request, 'dashboard')
