import os
import json
import pickle

import MeCab
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from django.core.management.base import BaseCommand

from nicole_or_kiko import logging

logger = logging.get_logger(__name__)

ROOT_DIR = os.path.dirname(os.path.realpath("__name__"))
DATASET_DIR = os.path.join(ROOT_DIR, "instagram_scraper")


class Command(BaseCommand):

    def handle(self, *args, **options):

        dataset_dict = self.__load_instagram_dataset()
        label_list = list(set(dataset_dict["label"]))
        label_list.sort()

        mt = MeCab.Tagger("-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd")
        X = [mt.parse(caption) for caption in dataset_dict["caption"]]
        y = [label_list.index(label) for label in dataset_dict["label"]]

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=114514)

        pipe_svc = Pipeline([("vectorizer", TfidfVectorizer()),
                             ("clf", SVC(random_state=114514))])

        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        param_grid = [{"clf__C": param_range, "clf__kernel": ["linear"]}]

        logger.info("[GridSearch] Start grid search")
        gs = GridSearchCV(estimator=pipe_svc,
                          param_grid=param_grid,
                          scoring="accuracy",
                          cv=10,
                          n_jobs=4)
        gs = gs.fit(X_train, y_train)

        logger.info("[GridSearch] Best score: {:.3f}".format(gs.best_score_))
        logger.info("[GridSearch] Best params: {}".format(gs.best_params_))

        clf = gs.best_estimator_
        clf.fit(X_train, y_train)
        logger.info("Test accuracy: {:.3f}".format(clf.score(X_test, y_test)))

        clf.fit(X, y)
        dump_path = os.path.join(ROOT_DIR, "classifier", "classifier.pkl")
        joblib.dump(clf, dump_path)

        logger.info("Dump to {}".format(dump_path))

    def __load_instagram_dataset(self):

        files = os.listdir(DATASET_DIR)
        user_dirs = list(filter(lambda x: os.path.isdir(os.path.join(DATASET_DIR, x)), files))

        json_filepaths = []
        for user_dir in user_dirs:

            # 各ユーザーのディレクトリを取得する
            files = os.listdir(os.path.join(DATASET_DIR, user_dir))
            # .json のファイルを探す
            json_file = list(filter(lambda x: x.endswith(".json"), files))
            # .jsonのフルパスを取得する
            json_file = map(lambda x: os.path.join(DATASET_DIR, user_dir, x), json_file)

            json_filepaths.extend(json_file)

        dataset_dict = {
            "caption": [],
            "label": []
        }

        for json_filepath, user in zip(json_filepaths, user_dirs):

            with open(json_filepath, "r") as rf:
                posts = json.load(rf)

            captions = [post["caption"]["text"] for post in posts if post["caption"] is not None]
            labels = [user for _ in range(len(captions))]

            dataset_dict["caption"].extend(captions)
            dataset_dict["label"].extend(labels)

        return dataset_dict
