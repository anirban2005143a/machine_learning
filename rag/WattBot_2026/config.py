import os
from dotenv import load_dotenv

load_dotenv()

settings = {
    'ELASTIC_SEARCH_URL' : os.getenv('ELASTIC_SEARCH_URL'),
    'ELASTIC_SEARCH_USER' : os.getenv('ELASTIC_SEARCH_USER'),
    'ELASTIC_SEARCH_PASS' : os.getenv('ELASTIC_SEARCH_PASS')
}