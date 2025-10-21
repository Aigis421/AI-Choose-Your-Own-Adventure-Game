# cya.py

from modules.llm_util import LLMUtil
from modules.prompts import STORY_PROMPT_TEMPLATE
from modules.game_engine import GameEngine

def main():
    llm = LLMUtil()
    engine = GameEngine(llm, STORY_PROMPT_TEMPLATE)
    engine.load_story_docs()
    engine.run()

if __name__ == "__main__":
    main()
