import logging

from django.conf import settings


def get_logger(name):
    return logging.getLogger("{}.{}".format(settings.LOGGING_PREFIX, name))
