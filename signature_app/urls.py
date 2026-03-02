# signature_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_hub, name='index_hub'),
    path('upload/', views.upload_page, name='upload_page'),
    path('live/', views.live_page, name='live_page'),
    
    
    # This matches the 'action' in your upload.html form
    path('authenticate_file/', views.authenticate_file, name='authenticate_file'),
    path('authenticate_live/', views.authenticate_live, name='authenticate_live'),
    path('authenticate-image/', views.authenticate_by_image, name='authenticate_image'),
    path('results/', views.show_results, name='show_results'),

]