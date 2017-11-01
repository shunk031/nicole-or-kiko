import json
import os
import pickle
import string

import MeCab
import numpy as np

from sklearn.externals import joblib
from sklearn.utils import check_random_state

from lime.lime_text import LimeTextExplainer

from django.shortcuts import render
from nicole_or_kiko.settings import BASE_DIR
from nicole_or_kiko import logging
from home.models import InputSentence

logger = logging.get_logger(__name__)

# load classifier
CLF = joblib.load(os.path.join(BASE_DIR, "classifier", "classifier.pkl"))

with open(os.path.join(BASE_DIR, "classifier", "label_list.pkl"), "rb") as rf:
    LABEL_LIST = pickle.load(rf)


def index(request):

    sentence = InputSentence()

    context = {
        "sentence": sentence
    }
    return render(request, "home/index.html", context)


def result(request):

    sentence = request.POST["sentence"]

    pred = predict(sentence)[0]

    pred_proba = predict_proba(sentence)
    logger.info("Predict proba: {}".format(pred_proba))
    explainer = LimeTextExplainer(class_names=LABEL_LIST)
    exp_instance = explainer.explain_instance(wakati(sentence), CLF.predict_proba)

    exp_labels = exp_instance.available_labels()

    random_id = id_generator(size=15, random_state=check_random_state(114514))

    top_div = '''
    <div class="lime top_div" id="top_div{}"></div>
    '''.format(random_id)

    # 事後確率を表示
    predict_proba_js = """
    var pp_div = top_div.append('div').classed('lime predict_proba', true);
    var pp_svg = pp_div.append('svg').style('width', '100%%');
    var pp = new lime.PredictProba(pp_svg, {}, {});
    """.format(jsonize(LABEL_LIST),
               jsonize(list(exp_instance.predict_proba.astype(float))))

    predict_value_js = ""

    # 予測に寄与した単語を表示
    exp_js = """
    var exp_div;
    var exp = new lime.Explanation({});
    """.format(jsonize(LABEL_LIST))

    for label in exp_labels:
        exp = jsonize(exp_instance.as_list())
        exp_js += """
        exp_div = top_div.append('div').classed('lime explanation', true);
        exp.show({}, {}, exp_div);
        """.format(exp, label)

    raw_js = """
    var raw_div = top_div.append('div');
    """
    logger.info("LABEL_LIST: {}".format(LABEL_LIST))
    logger.info("exp_labels: {}".format(exp_labels))
    logger.info("exp_instance.local_exp: {}".format(exp_instance.local_exp))

    # センテンスに対するアテンションを可視化
    html_data = exp_instance.local_exp[exp_labels[0]]
    raw_js += exp_instance.domain_mapper.visualize_instance_html(
        html_data, exp_labels[0],
        "raw_div",
        "exp")

    context = {
        "sentence": sentence,
        "pred": LABEL_LIST[pred],
        "pred_proba": pred_proba[0, 0],
        "as_list": jsonize(exp_instance.as_list()),
        # "bundle_js": bundle_js,
        "top_div": top_div,
        "random_id": random_id,
        "predict_proba_js": predict_proba_js,
        "predict_value_js": predict_value_js,
        "exp_js": exp_js,
        "raw_js": raw_js,
    }

    return render(request, "home/result.html", context)


def wakati(sentence):
    mt = MeCab.Tagger("-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
    return mt.parse(sentence)


def predict(sentence):

    return CLF.predict([wakati(sentence)])


def predict_proba(sentence):

    return CLF.predict_proba([wakati(sentence)])


def id_generator(size=15, random_state=None):

    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))


def jsonize(x):
    return json.dumps(x, ensure_ascii=False)
