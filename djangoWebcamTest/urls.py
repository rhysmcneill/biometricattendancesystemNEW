"""djangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from users.views import register, train  # import for users app
from users import views as user_views
from mainsite.views import home, howitworks, contactus # Import for mainsite app

urlpatterns = [
    path('admin/', admin.site.urls),
    path('register/', user_views.register, name='register'),
    path('', home, name="home"),
    path('howitworks', howitworks, name="howitworks"),
    path('contactus', contactus, name="contactus"),
    path('users/', include('django.contrib.auth.urls')),
    path('users/', include('users.urls')),
    path('login/', auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
    path('dashboard/', user_views.dashboard, name='dashboard'),
    path('admindashboard/', user_views.admindashboard, name='admindashboard'),
    path('vidstream/', user_views.add_photos, name='vidStream'),
    path('training/', user_views.training_dataset, name='training')
]
