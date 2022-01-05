from django.shortcuts import render
from django.core.mail import send_mail
from django.conf import settings
# Create your views here.

# General requests for main webapp pages
def home(request):
    return render(request, "mainsite/home.html")
def base(request):
    return render(request, "mainsite/base.html")
def howitworks(request):
    return render(request, "mainsite/howitworks.html")
def contactus(request):
    if request.method == "POST":
        messageName = request.POST['message_name']
        messageEmail = request.POST['message_email']
        messagSubject = request.POST['message_subject']
        message = request.POST['message']

        # send email to admin to retrieve contact message
        send_mail(
            messageName, # Email subject
            message, #Email message
            messageEmail, #Sender email address
            ['rhysdjmcneill@gmail.com'], # Receiver email
            fail_silently=False
        )
        return render(request, "mainsite/contactus.html", {'message_name' : messageName}) # Creates dictionary for username -> referenced in contactus.html success msg
    else:
        return render(request, "mainsite/contactus.html")

