from .summarization import generate_summary, retrieve_summary
from cassandra_utils import CassandraUtil

def process_and_store_chat_history(cassandra_util, session_id, chat_history, new_input, new_output):
    # Concatenate new input and output with existing chat history
    updated_history = chat_history + "\nHuman: " + new_input + "\nAI: " + new_output

    # Summarize the updated chat history
    summarized_history = generate_summary(updated_history)

    # Store the summarized history in Cassandra
    cassandra_util.store_chat_history(session_id, summarized_history)

    return summarized_history

def retrieve_and_expand_chat_history(cassandra_util, session_id):
    # Retrieve summarized chat history from Cassandra
    summarized_history = cassandra_util.retrieve_chat_history(session_id)

    # Check if the history is empty or not found
    if not summarized_history:
        return ""

    # Logic to expand or utilize summarized history
    # This could be as simple as returning the summarized history,
    # or it could involve more complex logic, such as:
    # - Parsing the summarized history to extract key information
    # - Combining it with additional data from other sources
    # - Expanding it with details stored elsewhere
    # 
    # Example of a simple expansion (you can replace it with more complex logic as needed)
    expanded_history = "Summary of Previous Conversations:\n" + summarized_history

    # Optionally, you might want to process the expanded history further,
    # like breaking it into manageable parts, tagging, or categorizing it.

    return expanded_history