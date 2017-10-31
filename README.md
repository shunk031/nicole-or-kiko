# Nicole or Kiko

## Requirements

- Python 3.6.2
- pyenv-virtualenv

## Setup pyenv-virtualenv

``` shell
pyenv virtualenv 3.6.2 nicole-or-kiko
```

## Setup django

``` shell
django-admin startproject nicole_or_kiko
python manage.py startapp home
python manage.py startapp instagram_scraper
python manage.py startapp classifier
```

## Crawl and scrape from Instagram

``` shell
cd nicole_or_kiko/instagram_scraper
instagram-scraper 2525nicole2 --media-metadata
instagram-scraper i_am_kiko --media-metadata
```

## Make Linear SVM Classifier

``` shell
python manage.py make_classifier
```

