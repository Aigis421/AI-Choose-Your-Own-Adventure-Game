import json
from modules.vector_store_util import VectorStoreUtil
from modules.llm_util import LLMUtil
from modules.prompts import STORY_PROMPT_TEMPLATE


class GameEngine:
    def __init__(self, llm, story_prompt_template):
        self.vector_util = VectorStoreUtil()
        self.llm = llm
        self.story_prompt_template = story_prompt_template
        self.player_state = {"location": "start", "inventory": []}

    def load_story_docs(self, path="data/story_docs.json"):
        with open(path, "r", encoding="utf-8") as f:
            story_data = json.load(f)
        texts = [chunk["text"] for chunk in story_data]
        metadata = [{"source": "story", "chapter": i} for i in range(len(story_data))]
        self.vector_util.add_documents(texts, metadata)
        self.vector_util.save()

    def summarize_context(self, full_context):
        """Return a condensed summary to prevent token overflow."""
        # Could call your LLM here to summarize past events
        # For now, just truncate to last 2000 characters
        return full_context[-2000:]

    def run(self):
        print("🌌 Welcome to the AI Adventure!\nType 'quit' to exit.\n")
        retriever = self.vector_util.as_retriever(k=2)
        summaries = ""
            # --- INITIAL NARRATION ---
        opening_prompt = STORY_PROMPT_TEMPLATE.format(
            context="",  # No previous context yet
            summaries=summaries,
            human_input="The game begins."  # LLM sees this as the first player action
        )
        opening_text = self.llm.generate(opening_prompt)
        print(f"\n{opening_text}\n")

        # Optional: update summaries with the opening narration
        summaries = self.summarize_context(opening_text)
        
        while True:
            user_input = input(">> ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("Game Over. Thanks for playing!")
                break
            
            # Retrieve context documents
            context_docs = retriever.invoke(user_input)
            
            #Handle empty context gracefully
            if context_docs:
                context_text = "\n".join([d.page_content for d in context_docs])
            else:
                context_text = "(No previous story context. Your actions shape the adventure!)"
            
            # Format the prompt with the player's input
            prompt = STORY_PROMPT_TEMPLATE.format(
                context=context_text,
                summaries=summaries,
                human_input=user_input
                )
            # Generate story output
            story_output = self.llm.generate(prompt)

            # Generate story output
            print(f"\n{story_output}\n")

            summaries = self.summarize_context(context_text + "\n" + story_output)