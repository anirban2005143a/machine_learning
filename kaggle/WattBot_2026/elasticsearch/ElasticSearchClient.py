from elasticsearch import Elasticsearch
from config import settings
from .synonyms import SYNONYMS_SET


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

        # ---------------------------------------------------------
        # Check existing index
        # ---------------------------------------------------------
        if self.index_exists(index_name):

            if not force_recreate:
                raise Exception(f"Index already exists: {index_name}")

            print(f"Deleting existing index: {index_name}")
            self.delete_index(index_name)

        # ---------------------------------------------------------
        # Index configuration
        # ---------------------------------------------------------
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
                        # Analyzer used while indexing searchable text
                        "my_index_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "my_stemmer",
                            ],
                        },
                        # Analyzer used while searching
                        "my_search_analyzer": {
                            "tokenizer": "standard",
                            "filter": [
                                "lowercase",
                                "my_synonyms",
                                "my_stemmer",
                            ],
                        },
                    },
                },
            },
            "mappings": {
                "dynamic": True,
                "properties": {
                    # =====================================================
                    # LangChain Document
                    # =====================================================
                    "page_content": {
                        "type": "text",
                        "analyzer": "my_index_analyzer",
                        "search_analyzer": "my_search_analyzer",
                    },
                    # =====================================================
                    # Metadata
                    # =====================================================
                    "metadata": {
                        "type": "object",
                        "dynamic": True,
                        "properties": {
                            # -------------------------------------------------
                            # ID
                            # Exact identifier + filtering
                            # -------------------------------------------------
                            "id": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                    }
                                },
                            },
                            # -------------------------------------------------
                            # Type
                            # Example: paper, report
                            # Exact filtering
                            # -------------------------------------------------
                            "type": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                    }
                                },
                            },
                            # -------------------------------------------------
                            # Title
                            # Searchable only
                            # No filtering
                            # -------------------------------------------------
                            "title": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                            },
                            # -------------------------------------------------
                            # Year
                            # Searchable + filtering + range queries
                            # -------------------------------------------------
                            "year": {
                                "type": "integer",
                            },
                            # -------------------------------------------------
                            # Citation
                            # Searchable only
                            # No filtering
                            # -------------------------------------------------
                            "citation": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                            },
                            # -------------------------------------------------
                            # URL
                            # Exact matching + filtering
                            # -------------------------------------------------
                            "url": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                    }
                                },
                            },
                            # -------------------------------------------------
                            # Peer review status
                            # Example: yes, no, preprint
                            # Exact filtering
                            # -------------------------------------------------
                            "peer_reviewed": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                    }
                                },
                            },
                            # -------------------------------------------------
                            # Venue evidence
                            # Searchable + exact filtering
                            # -------------------------------------------------
                            "venue_evidence": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                    }
                                },
                            },
                            # -------------------------------------------------
                            # Venue
                            # Searchable only
                            # No filtering
                            # -------------------------------------------------
                            "venue": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                            },
                            # -------------------------------------------------
                            # Section headings
                            # Searchable + exact filtering
                            # -------------------------------------------------
                            "section_headings": {
                                "type": "text",
                                "analyzer": "my_index_analyzer",
                                "search_analyzer": "my_search_analyzer",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword",
                                    }
                                },
                            },
                        },
                    },
                    # =====================================================
                    # Chunk embedding
                    # =====================================================
                    "content_embedding": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine",
                    },
                },
            },
        }

        # ---------------------------------------------------------
        # Create index
        # ---------------------------------------------------------
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
