import os
import re
import json
import tqdm
import torch
import time
import argparse
import transformers
from transformers import AutoTokenizer, pipeline
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
import sys
sys.path.append("src")
from utils import RetrievalSystem
from template import *

if openai.api_key is None:
    from config import config
    openai.api_type = config["api_type"]
    openai.api_base = config["api_base"]
    openai.api_version = config["api_version"]
    openai.api_key = config["api_key"]

class MedRAG:
    def __init__(self, llm_name="llama-2-70b-chat-hf", rag=True, retriever_name="MedCPT", corpus_name="Textbooks", db_dir="./corpus", cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                          "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}

        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir)
        else:
            self.retrieval_system = None

        if "llama-2-70b-chat-hf" in llm_name.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name, cache_dir=self.cache_dir)
            self.model = pipeline("text-generation", model=llm_name, cache_dir=self.cache_dir)
            self.max_length = 4096  # Set appropriate max length for the LLaMA model
            self.context_length = 3072  # Set appropriate context length for the LLaMA model
        else:
            # Load other models as necessary
            pass

    def answer(self, question, options=None, k=32, rrf_k=100, save_dir=None):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        save_dir (str): directory to save the results
        '''

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = '' # double check this later!!!!!! See if new prompt tempates are needed.

        # retrieve relevant snippets
        if self.rag:
            retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)
            contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
            if len(contexts) == 0:
                contexts = [""]
            if "openai" in self.llm_name.lower():
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
            else:
                contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages)
                answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores
            

    def generate(self, messages):
        # Generation logic using LLaMA or other models
        if "llama-2-70b-chat-hf" in self.llm_name.lower():
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            response = self.model(prompt, max_length=self.max_length, truncation=True)
            ans = response[0]["generated_text"]
        else:
            # Handling for other models
            pass
        return ans

    # Other methods as before
