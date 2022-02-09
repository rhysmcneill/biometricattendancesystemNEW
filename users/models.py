from django.contrib.auth.models import User
from django.db import models

# Create your models here.

import datetime


class Meta:
    model = User
    fields = ("username", 'email', 'password1', 'password2')


class is_Present(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(default=datetime.date.today)
    is_present = models.BooleanField(default=False)


class clocked_Time(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    date = models.DateField(default=datetime.date.today)
    time = models.DateTimeField(null=True, blank=True)
    signed_out = models.BooleanField(default=False)
