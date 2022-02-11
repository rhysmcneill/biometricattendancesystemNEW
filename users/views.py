import os
import pickle
from datetime import datetime, time
from pyexpat import model

import dlib
import face_recognition
import numpy as np
from imutils import face_utils
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from django.views.decorators.http import condition
from sklearn.svm import SVC
from sklearn.svm._libsvm import predict, predict_proba
from sklearn.linear_model import LinearRegression
from face_recognition.face_recognition_cli import image_files_in_folder
from .forms import UserRegForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import imutils
from users.models import is_Present, clocked_Time


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
            return redirect('home')
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
                return redirect('/adminDashboard')  # or your url name
            else:
                login(request, user)
                return redirect('/howitworks')
    else:
        return render(request, 'users/login.html', {})


# Login required to access employee dashboard
@login_required
def dashboard(request):
    return render(request, 'users/dashboard.html')


def loading_training(request):
    return render(request, 'users/trainDatasetLoader.html')

# Login required to access admin dashboard
@login_required
def admindashboard(request):
    return render(request, 'users/adminDashboard.html')


def logout_view(request):
    logout(request)
    return redirect('home')


def display_livefeed(request):
    return render(request, 'users/vidStream.html')


def create_dataset(username):
    id = username
    if (os.path.exists('image_data\Train_Data\{}\\'.format(id)) == False):
        os.makedirs('image_data\Train_Data\{}\\'.format(id))
    directory = 'image_data\Train_Data\{}\\'.format(id)

    # Detect face
    # Load HOG

    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    # Initialise cv2 video stream
    print("[INFO] Initializing Video stream")
    video = cv2.VideoCapture(0)

    # dataset counter
    dataset_counter = 0
    # Capturing the faces one by one and detect the faces and showing it on the window
    while (True):
        # Capturing the image
        # video.read each frame
        _, img = video.read()
        # Resize each image
        img = imutils.resize(img, width=1000)
        # convert img to grey for the ML Classifier to work
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # This will detect all the images in the current frame, and it will return the coordinates of the faces
        # Takes in image and some other parameter for accurate result
        faces = detector(gray, 0)
        # @faces - this ensures there can be multiple faces so we have to get each and every face and draw a rectangle around it.
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        print("[INFO] Adding face detection images...")
        for (x, y, w, h) in faces:


            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Whenever the program captures the face, we will write that is a folder
            # Before capturing the face, we need to tell the script whose face it is
            # For that we will need an identifier, here we call it id
            # So now we captured a face, we need to write it in a file
            dataset_counter = dataset_counter + 1
            # Saving the image dataset, but only the face part, cropping the rest

            if (x, y, w, h) is None:
                print("face is none")
                continue

            cv2.imwrite(directory + '/' + str(dataset_counter) + '.jpg', img)
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
        if (dataset_counter > 300):
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


def train(request):

    # Dataset directory for training.
    dataset_directory = 'image_data\Train_Data'

    person_counter = 0
    face_encodings = []
    person_names = []
    counter = 0
    label_id = 0
    images = []
    labels = []
    names = {}

    # Checks for amount of personnel in dataset and their images
    for person_name in os.listdir(dataset_directory):
        current_directory = os.path.join(dataset_directory, person_name)
        if not os.path.isdir(current_directory):
            continue
        for img in image_files_in_folder(current_directory):
            person_counter += 1

    # Checks and joins person_name to directory and encodes a 128-dimension face encoding for each face in the image.
    for person_name in os.listdir(dataset_directory):
        print(str(person_name))
        names[label_id] = person_names
        current_directory = os.path.join(dataset_directory, person_name)
        if not os.path.isdir(current_directory):
            continue
        for img in image_files_in_folder(current_directory):
            print(str(img))
            read_image = cv2.imread(img)
            label = label_id
            images.append(cv2.imread(img, 0))
            labels.append(int(label)  )
            try:
                # 128-dimension encoding
                face_encodings.append((face_recognition.face_encodings(read_image)[0]).tolist())

                person_names.append(person_name) # Appending names to list
                counter += 1

            except:
                print("Image Removed from Directory")
                os.remove(img)
        label_id += 1

    # Preparing data for trainging
    (images, labels) = [np.array(lis) for lis in [images, labels]]

    # training the model
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    model.save(r'mainsite\\templates\mainsite\\trainningData.yml')


@login_required
def training_dataset(request):
    if not request.user.is_authenticated:
        messages.warning(request, f'User not authorised')
        return redirect('admindashboard')
    elif request.user.is_superuser or request.user.is_staff:
        train(request)
        messages.success(request, f'Training Complete.')
        return redirect('admindashboard')
    else:
        messages.warning(request, f'Error - please see logs for details.')
        return render(request, 'admindashboard')


def face_recognition_mark_in(request):
    # Dataset directory
    dataset_directory = 'image_data\Train_Data'
    person_counter = 0
    id = 0
    name = {}
    count = dict()
    is_present = dict()
    current_time = dict()
    username = request.user.username

    # Gets all the personnel within the system
    for person_name in os.listdir(dataset_directory):
        current_directory = os.path.join(dataset_directory, person_name)
        name[id] = person_name
        if not os.path.isdir(current_directory):
            continue
        for img in image_files_in_folder(current_directory):
            person_counter += 1
        id += 1

    # Gets the trained dataset and initialises the face recogniser/detector
    # We use Local Binary Patterns Histogram algorithm to recognise faces in the feed
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(r'mainsite\\templates\mainsite\\trainningData.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialise camera
    webcam = cv2.VideoCapture(0)
    while True:
        (_, frame) = webcam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            for employee in dataset_directory:
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (130, 100))
                # Predicting faces based on model
                prediction = model.predict(face_resize)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # If prediction < 80 then dont mark employee in - < 80% is not valid
                if prediction[1] < 80:
                    count[username] = 0
                # If prediction <= 120 and > 80 then mark employee in - > 80% is valid
                elif prediction[1] <= 120:
                    cv2.putText(frame, '% s - %.0f' %
                                (name[prediction[0]], prediction[1]), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                    is_present[username] = True
                    current_time[username] = datetime.now()
                    count[username] = count.get(username, 0) + 1
                else:
                    cv2.putText(frame, 'Unknown Entity',
                                (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        cv2.imshow('Facial Recogniser - Attendance (Press q to quit)', frame)

        key = cv2.waitKey(50) & 0xFF
        if (key == ord("q")):
            break

    webcam.release()
    cv2.destroyAllWindows()
    update_db_attendance_in(is_present)  # update db with attendance


@login_required
def mark_attendance_in(request):
    if not request.user.is_authenticated:
        messages.warning(request, f'User not authorised')
        return redirect('home')
    elif request.user.is_superuser or request.user.is_staff:
        messages.warning(request, f'Admin cannot use this functionality')
        return redirect('home')
    elif request.user.is_authenticated:
        face_recognition_mark_in(request)
        messages.success(request, f'Attendance successful')
        return redirect('home')
    else:
        messages.warning(request, f'Error - please see logs for details.')
        return redirect(request, 'home')


def update_db_attendance_in(present):
    today = datetime.today()
    time = datetime.now()
    for employee in present:
        user = User.objects.get(username=employee)
        try:
            attendance = is_Present.objects.get(user=user, date=today)
        except:
            attendance = None

        if attendance is None:
            if present[employee] == True:
                a = is_Present(user=user, date=today, is_present=True)
                a.save()
            else:
                a = is_Present(user=user, date=today, is_present=False)
                a.save()
        else:
            if present[employee] == True:
                attendance.present = True
                attendance.save(update_fields=['is_present'])
        if present[employee] == True:
            a = clocked_Time(user=user, date=today, time=time, signed_out=False)
            a.save()
