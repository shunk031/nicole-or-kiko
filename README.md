# Nicole or Kiko

for demo of campus festival

## Requirements

- Python 3.6.2
- pyenv-virtualenv

## Setup pyenv-virtualenv

``` shell
pyenv virtualenv 3.6.2 nicole-or-kiko
```

## Install requirements

```shell
pip install -r python_requirements.txt
```

## Crawl and scrap from Instagram

``` shell
python manage.py scrap_instagram 2525nicole2 --media-metadata
python manage.py scrap_instagram i_am_kiko --media-metadata
```

## Make Linear SVM Classifier

``` shell
python manage.py make_classifier
```

## Run server

``` shell
python manage.py runserver
# and then access http://localhost:8000/home
```
