from django.db import models

# Create your models here.

class Login(models.Model):
    username = models.CharField(max_length = 40)
    password = models.CharField(max_length  = 40)
    name = models.CharField(max_length = 40)
