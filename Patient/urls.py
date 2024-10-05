from django.urls import path 
from .import views 

urlpatterns=[
    path('index',views.index,name='index'),
    path('signup',views.signup,name='signup'),
    path('signin',views.signin,name='signin'),
    path('parkinson',views.parkinson,name='parkinson'),
    path('test',views.test,name='test')
]