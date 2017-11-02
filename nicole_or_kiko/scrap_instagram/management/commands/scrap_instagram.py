import re

from instagram_scraper.app import InstagramScraper
from django.core.management.base import BaseCommand


class Command(BaseCommand):

    def add_arguments(self, parser):

        parser.add_argument('username', help='Instagram user(s) to scrape', nargs='*')
        parser.add_argument('--destination', '-d', default='./scrap_instagram/static/scrap_instagram/downloaded', help='Download destination')
        parser.add_argument('--login-user', '--login_user', '-u', default=None, help='Instagram login user')
        parser.add_argument('--login-pass', '--login_pass', '-p', default=None, help='Instagram login password')
        parser.add_argument('--login-only', '--login_only', '-l', default=False, action='store_true', help='Disable anonymous fallback if login fails')
        parser.add_argument('--filename', '-f', help='Path to a file containing a list of users to scrape')
        parser.add_argument('--quiet', '-q', default=False, action='store_true', help='Be quiet while scraping')
        parser.add_argument('--maximum', '-m', type=int, default=0, help='Maximum number of items to scrape')
        parser.add_argument('--retain-username', '--retain_username', '-n', action='store_true', default=True,
                            help='Creates username subdirectory when destination flag is set')
        parser.add_argument('--media-metadata', '--media_metadata', action='store_true', default=False, help='Save media metadata to json file')
        parser.add_argument('--include-location', '--include_location', action='store_true', default=False, help='Include location data when saving media metadata')
        parser.add_argument('--media-types', '--media_types', '-t', nargs='+', default=['image', 'video', 'story'], help='Specify media types to scrape')
        parser.add_argument('--latest', action='store_true', default=False, help='Scrape new media since the last scrape')
        parser.add_argument('--tag', action='store_true', default=False, help='Scrape media using a hashtag')
        parser.add_argument('--filter', default=None, help='Filter by tags in user posts', nargs='*')
        parser.add_argument('--location', action='store_true', default=False, help='Scrape media using a location-id')
        parser.add_argument('--search-location', action='store_true', default=False, help='Search for locations by name')
        parser.add_argument('--comments', action='store_true', default=False, help='Save post comments to json file')
        parser.add_argument('--verbose', type=int, default=0, help='Logging verbosity level')

    def handle(self, *args, **kwargs):

        if (kwargs["login_user"] and kwargs["login_pass"] is None) or (kwargs["login_user"] is None and kwargs["login_pass"]):
            raise ValueError('Must provide login user AND password')

        if not kwargs["username"] and kwargs["filename"] is None:
            raise ValueError('Must provide username(s) OR a file containing a list of username(s)')
        elif kwargs["username"] and kwargs["filename"]:
            raise ValueError('Must provide only one of the following: username(s) OR a filename containing username(s)')

        if kwargs["tag"] and kwargs["location"]:
            raise ValueError('Must provide only one of the following: hashtag OR location')

        if kwargs["tag"] and kwargs["filter"]:
            raise ValueError('Filters apply to user posts')

        if kwargs["filename"]:
            kwargs["usernames"] = InstagramScraper.parse_file_usernames(kwargs["filename"])
        else:
            kwargs["usernames"] = InstagramScraper.parse_delimited_str(','.join(kwargs["username"]))

        if kwargs["media_types"] and len(kwargs["media_types"]) == 1 and re.compile(r'[,;\s]+').findall(kwargs["media_types"][0]):
            kwargs["media_types"] = InstagramScraper.parse_delimited_str(kwargs["media_types"][0])

        scraper = InstagramScraper(**kwargs)

        if kwargs["tag"]:
            scraper.scrape_hashtag()
        elif kwargs["location"]:
            scraper.scrape_location()
        elif kwargs["search_location"]:
            scraper.search_locations()
        else:
            scraper.scrape()
