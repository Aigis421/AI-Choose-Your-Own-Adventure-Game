from cassandra_utils import CassandraUtil
from adapters.cassandra_vector_store_adapter import CassandraVectorStoreAdapter
from openai_utils import OpenAIUtil
from data_processing.summarization import generate_summary, retrieve_summary
from data_processing.chat_management import process_and_store_chat_history, retrieve_and_expand_chat_history
import tiktoken
from tokenizers import Tokenizer
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, RetrievalQAWithSourcesChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import json
import re
import uuid

# class PromptTemplate:
#     def __init__(self, input_variables, template):
#         self.input_variables = input_variables
#         self.template = template

#     def render(self, **kwargs):
#         return self.template.format(**kwargs)

# Token Counting Function
# Note: This is a simplified version. OpenAI's actual tokenization may count tokens differently.
# def count_tokens(text):
#     return len(text.split())

# Chat History Truncation Function
# Truncates the chat history to fit within the token limit
# def truncate_chat_history(chat_history, max_length):
#     # Split the history into lines
#     lines = chat_history.split('\n')
#     # Keep adding lines until the max_length is reached
#     truncated_history = ''
#     for line in lines[::-1]:  # Start from the most recent
#         if count_tokens(truncated_history + line) + TRUNCATE_THRESHOLD < max_length:
#             truncated_history = line + '\n' + truncated_history
#         else:
#             break
#     return truncated_history.strip()

# def count_tokens(text):
#     # Approximating that each word and space is a token
#     return len(text.split()) + text.count(' ')

def count_tokens(text, encoding):
    return len(encoding.encode(text))

# def truncate_to_token_limit(text, token_limit=4096):
#     tokens = text.split()
#     while count_tokens(' '.join(tokens)) > token_limit:
#         tokens.pop()
#     return ' '.join(tokens)

def truncate_to_token_limit(text, encoding, token_limit=4096):
    tokens = encoding.encode(text)
    if len(tokens) > token_limit:
        # Truncate the tokens and decode back to text
        truncated_tokens = tokens[:token_limit]
        return encoding.decode(truncated_tokens)
    return text

def get_valid_input(prompt):
    while True:
        user_input = input(prompt).strip()
        if user_input:
            return user_input
        else:
            print("Please enter a valid response.")

# def extract_choices(response):
#     # Example pattern to extract choices - adjust according to your AI's response format
#     pattern = r'\[Choice (\d+): (.*?)\]'
#     return re.findall(pattern, response)
#     return [(num, text) for num, text in matches if len(matches) == 2]


# def display_choices(choices):
#     for number, choice in choices:
#         print(f"{number}: {choice}")
#     return choices


with open("cya_openai_game-token.json") as f:
    secrets = json.load(f)

cloud_config= {'secure_connect_bundle': 'secure-connect-cya-openai-game.zip'}
CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]
ASTRA_DB_KEYSPACE = "insert database"
OPENAI_API_KEY = "insert api key"


# auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
# cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
# session = cluster.connect()
# Initialize Cassandra Utility
cassandra_util = CassandraUtil(cloud_config, CLIENT_ID, CLIENT_SECRET, ASTRA_DB_KEYSPACE)

# Initialize the Vector Store Adapter with Cassandra Utility
cassandra_vector_store = CassandraVectorStoreAdapter(cassandra_util)

openai_util = OpenAIUtil(OPENAI_API_KEY)
enc = tiktoken.encoding_for_model("gpt-4")

# message_history = CassandraChatMessageHistory(
#     session_id="anything",
#     session=session,
#     keyspace=ASTRA_DB_KEYSPACE,
#     ttl_seconds=3600
# )
message_history = cassandra_util.create_message_history("anything")
cass_buff_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history)

template = """
Guiding a text adventure in Sithguard. Choices shape the path; organizations and foes lurk. 
Respond to user input with guidance, choices, and narrative descriptions. The story evolves towards an epic conclusion.

Rules:
1. Assign a class to the player determining abilities and options.
2. Present a choice of weapons, including player's custom choices.
3. Offer multiple paths, with both success and failure outcomes.
4. On failure, describe the outcome, ending with "The End." to conclude the game.
5. Respond based on player's previous input for a coherent narrative.
6. Store items players keep for inventory management.
7. Maintain a health bar for the player.
8. Provide an ending only when all quests are completed.
9. Counter player actions logically, yielding only when it makes sense.
10. Limit maximum context to within 4097 tokens every time.
11. Keep player's inventory updated and interactive.

Game State: {chat_history}
Summaries: {summaries}
Player Input: {human_input}
AI Response:
"""
# cass_buff_memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     chat_memory=message_history
# )



prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "summaries"],
    template=template
)
# llm = OpenAI(model_name="gpt-3.5-turbo", max_tokens=1000, temperature=0)
# prompt = {
#     "template": "Your input: {input}",
#     "variables": {"input": "Placeholder for user input"}
# }
chain_type_kwargs = {"prompt": prompt}  # 'prompt' is your conversation template or structure
# Setup the chain
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm = OpenAI(openai_api_key=OPENAI_API_KEY),
    vectorstore=cassandra_vector_store,  # Assuming 'index.store' is your vector database store
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
    reduce_k_below_max_tokens=True
)
# llm = OpenAI(openai_api_key=OPENAI_API_KEY)
# llm_chain = LLMChain(
#     llm=llm,
#     prompt=prompt,
#     memory=cass_buff_memory
# )

# choice = "start"

# while True:
#     response = llm_chain.predict(human_input=choice)
#     print(response.strip())

#     if "The End." in response:
#         break

#     choice = input("Your reply: ")


# In the game loop
# tokenizer = CharBPETokenizer(unk_token="[UNK]")

session_id = str(uuid.uuid4())  # Generates a unique session ID
chat_history = ""
choice = ""
human_input = "{human_input}"
 # Initialize chat history; load from DB if resuming a session
MAX_TOKENS = 4096  # For GPT-3 models, adjust if using a different model
TRUNCATE_THRESHOLD = 100  # A buffer to prevent exceeding the token limit
while True:
    try:
        # if not choice:
            # choice = get_valid_input("Please choose your starting class [swordfighter, pirate, mage, priest, or any you'd like]: ")
        user_input = get_valid_input("Your reply: ") if chat_history else get_valid_input("Please choose your starting class [swordfighter, pirate, mage, priest, or any you'd like]: ")

        # chat_history = truncate_to_token_limit(chat_history, enc, MAX_TOKENS)

        # chat_history += f"\nHuman: {user_input}"
        chat_history = process_and_store_chat_history(cassandra_util, session_id, chat_history, user_input, response)        
        # current_tokens = count_tokens(chat_history + human_input, enc)
        # if current_tokens > MAX_TOKENS - TRUNCATE_THRESHOLD:
        #     # Logic to truncate chat_history
        #     truncated_chat_history = truncate_to_token_limit(chat_history, MAX_TOKENS - TRUNCATE_THRESHOLD)
        #     chat_history = truncated_chat_history + human_input
        # else:
        #     chat_history += human_input

        # if count_tokens(chat_history, enc) > 4096:
        #     chat_history = truncate_to_token_limit(chat_history, enc)

        # full_prompt = chat_history + "\nAI:"

        combined_text = f"{chat_history}\nAI:"
        if count_tokens(combined_text, enc) > MAX_TOKENS:
            combined_text = truncate_to_token_limit(combined_text, enc, MAX_TOKENS)

        prompt_text = f"{chat_history}\nAI:"
        prompt = PromptTemplate(input_variables=["chat_history", "human_input"], template=template)
        llm_chain = openai_util.create_llm_chain(prompt, cass_buff_memory)

        # Now generate the prompt with the possibly truncated chat history
        # prompt = prompt.update(chat_history=combined_text, human_input=user_input)  # Adjust this line as needed
        response = llm_chain.run(chat_history)
        # response = llm_chain.predict(human_input=prompt_text)
        response_text = response.strip()
        print("AI:", response_text)

        chat_history += f"\nAI: {response_text}"
        
        chat_history = truncate_to_token_limit(chat_history, enc, MAX_TOKENS)

        if "The End." in response_text or "Congratulations, adventurer!" in response_text:
            break
        

    except Exception as e:
        print(f"An error occurred: {e}")
        break
        # Handle the error or offer the player a chance to retry/continue

  # After AI response, always ask for user input
        # choice = get_valid_input("Your reply: ")

        # choices = extract_choices(response_text)
        # if choices:
        #     displayed_choices = display_choices(choices)
        #     choice_number = get_valid_input("Choose an option: ")
        #     choice = next((text for num, text in displayed_choices if num == choice_number), None)
        # else:
        #     choice = get_valid_input("Your reply: ")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break
        # Handle the error or offer the player a chance to retry/continue
