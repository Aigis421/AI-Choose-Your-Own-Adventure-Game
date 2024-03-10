from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from langchain.memory import CassandraChatMessageHistory
from langchain.chains import LLMChain
class CassandraUtil:
    def __init__(self, cloud_config, client_id, client_secret, keyspace):
        auth_provider = PlainTextAuthProvider(client_id, client_secret)
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        self.session = cluster.connect()
        self.keyspace = keyspace

    def create_message_history(self, session_id, ttl_seconds=3600):
        message_history = CassandraChatMessageHistory(
            session_id=session_id,
            session=self.session,
            keyspace=self.keyspace,
            ttl_seconds=ttl_seconds
        )
        return message_history

    def store_vector(self, doc_id, vector):
        """
        Stores a vector in Cassandra for a given document ID.
        :param doc_id: The document identifier.
        :param vector: The vector to store (list or similar iterable).
        """
        query = "INSERT INTO vectors (doc_id, vector) VALUES (%s, %s)"
        self.session.execute(query, (doc_id, vector))

    def retrieve_vector(self, doc_id):
        """
        Retrieves a stored vector from Cassandra by document ID.
        :param doc_id: The document identifier.
        :return: The retrieved vector or None if not found.
        """
        query = "SELECT vector FROM vectors WHERE doc_id = %s"
        result = self.session.execute(query, (doc_id,))
        return result.one().vector if result else None