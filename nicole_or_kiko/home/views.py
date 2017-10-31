import os
import pickle

import MeCab
import numpy as np

from sklearn.externals import joblib

from django.shortcuts import render
from nicole_or_kiko.settings import BASE_DIR
from home.models import InputSentence

# load classifier
clf = joblib.load(os.path.join(BASE_DIR, "classifier", "classifier.pkl"))


def index(request):

    sentence = InputSentence()

    context = {
        "sentence": sentence
    }
    return render(request, "home/index.html", context)


def result(request):

    sentence = request.POST["sentence"]

    pred = predict(sentence)

    context = {
        "sentence": sentence,
        "pred": pred
    }

    return render(request, "home/result.html", context)


def predict(sentence):

    mt = MeCab.Tagger("-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    X = np.array([mt.parse(sentence)])

    return clf.predict(X)
