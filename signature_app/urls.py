from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_page, name='upload_page'),
    path('authenticate/', views.authenticate_signature, name='authenticate'),
]