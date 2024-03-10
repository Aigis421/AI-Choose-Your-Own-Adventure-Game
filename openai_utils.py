from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
class OpenAIUtil:
    def __init__(self, api_key):
        self.llm = OpenAI(openai_api_key=api_key)

    def create_llm_chain(self, prompt_template, cass_buff_memory):
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            memory=cass_buff_memory
        )
        return llm_chain