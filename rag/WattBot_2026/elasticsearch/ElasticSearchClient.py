from elasticsearch import Elasticsearch
from config import settings
from elasticsearch.synonyms import SYNONYMS_SET

class ElasticsearchClient:

    def __init__(self):
        self.client = None

    # -------------------------
    # Connection
    # -------------------------

    def connect(self):
        if self.client is not None:
            return

        self.client = Elasticsearch(
            settings["ELASTIC_SEARCH_URL"],
            basic_auth=(
                settings["ELASTIC_SEARCH_USER"],
                settings["ELASTIC_SEARCH_PASS"],
            ),
            verify_certs=False,
        )

        if not self.client.ping():
            raise ConnectionError("Unable to connect to Elasticsearch")

        print("Elasticsearch connected successfully")

    def get_client(self):
        if self.client is None:
            raise ConnectionError("Elasticsearch is not connected")

        return self.client

    # -------------------------
    # Index operations
    # -------------------------

    def index_exists(self, index_name):
        if not index_name:
            raise ValueError("Index name not provided")

        return self.get_client().indices.exists(index=index_name)

    def create_index(self, index_name, force_recreate=False):
        if not index_name:
            raise ValueError("Index name not provided")

        # Check existing index
        if self.index_exists(index_name):

            if not force_recreate:
                raise Exception(f"Index already exists: {index_name}")

            print(f"Deleting existing index: {index_name}")

            self.delete_index(index_name)

        index_config = {
            "settings": {
                "analysis": {
                    "filter": {
                        "my_synonyms": {
                            "type": "synonym_graph",
                            "synonyms": SYNONYMS_SET,
                        },
                        "my_stemmer": {
                            "type": "stemmer",
                            "name": "english",
                        },
                    },
                    "analyzer": {
                        #    1. Used when storing the
                        "my_index_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "my_stemmer"],
                        },
                        #    2. Used only when a user types in the search bar
                        "my_search_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "my_synonyms", "my_stemmer"],
                        },
                    },
                },
            },
            "mappings": {
                # IMPORTANT:
                # Unknown fields are allowed.
                # Elasticsearch dynamically maps them.
                "dynamic": True,
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "my_index_analyzer",
                    },
                    "authors": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "abstract": {
                        "type": "text",
                        "analyzer": "my_index_analyzer",
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "my_index_analyzer",
                    },
                    "year": {"type": "integer"},
                    "venue": {"type": "keyword"},
                    "paper_id": {"type": "keyword"},
                    "title_embedding": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "content_embedding": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                },
            },
        }

        self.get_client().indices.create(
            index=index_name,
            **index_config,
        )

        print(f"Index created: {index_name}")

    def delete_index(self, index_name):

        if not self.index_exists(index_name):
            print(f"Index does not exist: {index_name}")
            return

        self.get_client().indices.delete(index=index_name)

        print(f"Index deleted: {index_name}")

    def get_all_indexes(self):

        return self.get_client().cat.indices(format="json")

    # -------------------------
    # Document operations
    # -------------------------

    def add_document(self, index_name, document, document_id=None):

        response = self.get_client().index(
            index=index_name,
            id=document_id,
            document=document,
        )

        return response

    def get_document(self, index_name, document_id):

        return self.get_client().get(
            index=index_name,
            id=document_id,
        )

    def delete_document(self, index_name, document_id):

        return self.get_client().delete(
            index=index_name,
            id=document_id,
        )

    # -------------------------
    # Search
    # -------------------------

    def search(self, index_name, query):

        return self.get_client().search(
            index=index_name,
            query=query,
        )


# --------------------------------
# Singleton-style instance
# --------------------------------

es = ElasticsearchClient()
