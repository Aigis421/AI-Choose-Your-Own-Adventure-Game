# adapters/cassandra_vector_store_adapter.py
class CassandraVectorStoreAdapter:
    def __init__(self, cassandra_util):
        self.cassandra_util = cassandra_util

    def store_vector(self, doc_id, vector):
        # Use CassandraUtil to store the vector
        self.cassandra_util.store_vector(doc_id, vector)

    def retrieve_vector(self, doc_id):
        # Use CassandraUtil to retrieve the vector
        return self.cassandra_util.retrieve_vector(doc_id)

    # Add additional methods if LangChain requires more functionalities
    # like search_vectors, etc.
