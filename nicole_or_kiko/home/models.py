from django import forms
from django.db import models

# Create your models here.


class InputSentence(forms.Form):
    sentence = forms.CharField()
