from . import views
from django.urls import include,path

urlpatterns = [
    path('', views.login, name = "login"),
    path('test', views.testing, name = "test"),
    path('register', views.register, name = "register"),
    path('user/<str:name>',views.user,name = "user"),
]
