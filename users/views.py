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


@condition(etag_func=None)
def display_livefeed(self):
     return StreamingHttpResponse(
            livefeed(),
            content_type='multipart/x-mixed-replace; boundary=frame'
        )
