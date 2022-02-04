from django.contrib.auth.models import User
from django.db import models

# Create your models here.

import datetime


class Meta:
    model = User
    fields = ("username", 'email', 'password1', 'password2')
