import openai

def generate_summary(text):
    """
    Generates a summary for the given text using OpenAI's GPT model.
    """
    response = openai.Completion.create(
        engine="text-davinci-003",  # Or whichever GPT model you're using
        prompt=f"Summarize the following text:\n\n{text}",
        max_tokens=150  # Adjust based on your needs
    )
    summary = response.choices[0].text.strip()
    return summary

def retrieve_summary(cassandra_util, summary_id):
    """
    Retrieves a summary from Cassandra using the provided summary ID.
    """
    query = "SELECT summary_text FROM summaries WHERE id = %s"
    result = cassandra_util.session.execute(query, (summary_id,))
    return result.one().summary_text if result else None