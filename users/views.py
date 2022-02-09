import os
import pickle
from datetime import datetime, time
import dlib
import face_recognition
import numpy as np
from imutils import face_utils
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
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
    if (os.path.exists('image_data/Train_Data/{}/'.format(id)) == False):
        os.makedirs('image_data/Train_Data/{}/'.format(id))
    directory = 'image_data/Train_Data/{}/'.format(id)

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


def data_points(data_points, all_person_names):
    all_data_points = TSNE(n_components=2,
                      learning_rate='auto').fit_transform(data_points)

    for i, names in enumerate(set(all_person_names)):
        idx = all_person_names == names
        plt.scatter(all_data_points[idx, 0], all_data_points[idx, 1], label=names)

    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Scatter graph for employee facial data', fontdict=None, loc='center',)
    matplotlib.rcParams.update({'figure.autolayout': True})
    plt.tight_layout()
    plt.savefig('users/static/users/images/training_visualisation.png')
    plt.close()


def train(request):

    # Dataset directory for training.
    dataset_directory = 'image_data/Train_Data'

    person_counter = 0
    face_encodings = []
    person_names = []
    counter = 0

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
        current_directory = os.path.join(dataset_directory, person_name)
        if not os.path.isdir(current_directory):
            continue
        for img in image_files_in_folder(current_directory):
            print(str(img))
            read_image = cv2.imread(img)
            try:
                # 128-dimension encoding
                face_encodings.append((face_recognition.face_encodings(read_image)[0]).tolist())

                person_names.append(person_name) # Appending names to list
                counter += 1
            except:
                print("Image Removed from Directory")
                os.remove(img)

    # Add all dataset names to a NumPy array
    all_person_names = np.array(person_names)
    le = LabelEncoder() # Label encoder
    le.fit(person_names)
    normalised_encoding = le.transform(person_names)
    all_face_encodings = np.array(face_encodings)
    np.save('image_data/classification.npy', le.classes_)
    # Using linear support vector classification to split data into classes and train the model
    support_vector_class = SVC(kernel='linear', probability=True)
    support_vector_class.fit(all_face_encodings, normalised_encoding)
    support_vector_class_directory = "image_data/support_vector_class.sav"
    with open(support_vector_class_directory, 'wb') as file:
        pickle.dump(support_vector_class, file)

    data_points(all_face_encodings, all_person_names)

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

@login_required
def face_recognition_mark_in(request):

    # Detect face by loading HOG
    print("[INFO] Loading the facial detector")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('mainsite/templates/mainsite/shape_predictor_68_face_landmarks.dat')
    support_vector_class_directory = "image_data/support_vector_class.sav"

    with open(support_vector_class_directory, 'rb') as file:
        support_vector_class = pickle.load(file)
    fa = FaceAligner(predictor, desiredFaceWidth=100)
    le = LabelEncoder()
    le.classes_ = np.load('image_data/classification.npy')
    encodings = np.zeros((1, 128))
    all_faces = len(support_vector_class.predict_proba(encodings)[0])

    count = dict()
    is_present = dict()
    current_time = dict()
    start = dict()
    for i in range(all_faces):
        count[le.inverse_transform([i])[0]] = 0
        is_present[le.inverse_transform([i])[0]] = False

    # Initialise cv2 video stream
    print("[INFO] Initializing Video stream")
    video = cv2.VideoCapture(0)


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
        # @faces - this ensures there can be multiple faces so we have to get each and every face and draw a rectangle around its

        for face in faces:
            print("INFO : inside for loop")
            (x, y, w, h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(img, gray, face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            (prediction, probability) = predict(face_aligned, support_vector_class)

            if (prediction != [-1]):

                    employee_name = le.inverse_transform(np.ravel([prediction]))[0] # Turns labels into original encoding
                    prediction = employee_name
                    if count[prediction] == 0:
                        start[prediction] = time.time()
                        count[prediction] = count.get(prediction, 0) + 1

                    if count[prediction] == 4 and (time.time() - start[prediction]) > 1.2:
                        count[prediction] = 0
                    else:
                        is_present[prediction] = True
                        current_time[prediction] = datetime.datetime.now()
                        count[prediction] = count.get(prediction, 0) + 1
                        print(prediction, is_present[prediction], count[prediction])
                    cv2.putText(img, str(employee_name) + str(probability), (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            else:
                employee_name = "unknown"
                cv2.putText(img, str(employee_name), (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Showing the image in another window
        # Creates a window with window name "Mark Attendance - In - Press q to exit" and with the image img
        cv2.imshow("Mark Attendance - In - Press q to exit", img)
        # Before closing it we need to give a wait command, otherwise the open cv wont work
        # @params with the millisecond of delay 1
        # cv2.waitKey(1)
        # To get out of the loop
        key = cv2.waitKey(50) & 0xFF
        if (key == ord("q")):
            break

    # Stoping the videostream
    # destroying all the windows
    cv2.destroyAllWindows()
