from django import forms

# Create your forms here.

class ContactForm(forms.Form):
    name = forms.CharField(max_length=100)
    email = forms.EmailField()
    subject = forms.CharField(max_length=255)
    message = forms.CharField(widget = forms.Textarea, max_length = 2000)
