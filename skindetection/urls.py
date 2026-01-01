"""
URL configuration for skindetection app.
"""
from django.urls import path
from . import views

app_name = 'skindetection'

urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('upload/', views.upload, name='upload'),
    path('api/predict/', views.api_predict, name='api_predict'),
    path('api/login/', views.api_login, name='api_login'),
    path('api/signup/', views.api_signup, name='api_signup'),
    path('api/profile/', views.api_profile, name='api_profile'),
]

