from typing import List, Union

from pydantic import Field
from typing import Optional, Union, Sequence,TYPE_CHECKING
from pydantic import Field,BaseModel,field_validator
from agentscope.message import Msg
from agentscope.agents import AgentBase
from agentscope.parsers import ParserBase
from agentscope.memory import MemoryBase,TemporaryMemory
from datetime import datetime as dt
from twitter_utils import TwitterMessage,TwitterPage,InfoBox
from twitter_env import TwitterEnvironment,Registry,BaseEnvironment
from logging import getLogger
from agentscope.memory import MemoryBase,TemporaryMemory


from twitter_utils import TwitterMessage, TwitterPage, InfoBox
from agentscope.models.openai_model import OpenAIChatWrapper
from tqdm import tqdm
import json
import os
import re

import string
import numpy as np
import openai
from openai import OpenAI





memory_registry = Registry(name="MemoryRegistry")
@memory_registry.register("vectorstore")
class VectorStoreMemory(TemporaryMemory):

    """

    The main difference of this class with chat_history is that this class treat memory as a dict

    treat message.content as memory

    Attributes:
        messages (List[Message]) : used to store messages, message.content is the key of embeddings.
        embedding2memory (dict) : `key` is the embedding and `value` is the message
        memory2embedding (dict) : `key` is the message and `value` is the embedding
        llm (BaseLLM) : llm used to get embeddings


    Methods:
        add_message : Additionally, add the embedding to embeddings

    """

    messages: List[Msg] = Field(default=[])
    embedding2memory: dict = {}
    memory2embedding: dict = {}

    def add_message(self, messages: List[Msg]) -> None:
        for message in messages:
            self.messages.append(message)
            memory_embedding = self.get_embeddings(self.llm, message.content)
            self.embedding2memory[memory_embedding] = message.content
            self.memory2embedding[message.content] = memory_embedding

    def to_string(self, add_sender_prefix: bool = False) -> str:
        messages = self.messages
        if add_sender_prefix:
            return "\n".join(
                [
                    f"[{message.name}]: {message.content}"
                    if message.name != ""
                    else message.content
                    for message in messages
                ]
            )
        else:
            return "\n".join([message.content for message in messages])

    def reset(self) -> None:
        self.messages = []










class PersonalMemory(TemporaryMemory):
    

    def __init__(
        self,
        messages: List[Msg] = Field(default=[]),
        memory_path: str = None,
        target: str = "HK Peace",
        top_k: str = 5,
        deadline: str = None,
        model: str = "gpt-3.5-turbo",
        has_summary: bool = False,
        max_summary_length: int = 200,
        summary: str = "",
        SUMMARIZATION_PROMPT = '''Your task is to create a concise running summary of observations in the provided text, focusing on key and potentially important information to remember.

Please avoid repeating the observations and pay attention to the person's overall leanings. Keep the summary concise in one sentence.

Observations:
"""
{new_events}
"""
''',
        RETRIVEAL_QUERY='''What is your opinion on {target} or other political and social issues?''',
    ) -> None:
        """
        Temporary memory module for conversation.

        Args:
            embedding_model (Union[str, Callable])
                if the temporary memory needs to be embedded,
                then either pass the name of embedding model or
                the embedding model itself.
        """
        super().__init__()
        self.messages = messages
        self.memory_path = memory_path      
        self.target = target
        self.top_k = top_k
        self.deadline = deadline
        self.model = model
        self.has_summary = has_summary
        self.max_summary_length = max_summary_length
        self.summary = summary
        self.SUMMARIZATION_PROMPT = SUMMARIZATION_PROMPT
        self.RETRIVEAL_QUERY = RETRIVEAL_QUERY

    async def summarize(self):
        self.has_summary = True
        messages = self.messages
        print('summarize personal experience:', len(messages))
        if len(messages)==0:
            self.summary=''
            return
        texts = self.to_string(add_sender_prefix=True)
        prompt = self.SUMMARIZATION_PROMPT.format(
            new_events=texts
        )
        response = await openai.ChatCompletion.acreate(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_summary_length,
            temperature=0.5,
        )
        self.summary =  response["choices"][0]["message"]["content"]   
        message = Msg(content=self.summary)
        self.add([message])
    
    def to_string(self, add_sender_prefix: bool = False) -> str:   
        if add_sender_prefix:
            return "\n".join(
                [
                    f"[{message.name}] posted a tweet: {message.content}"
                    if message.name != ""
                    else message.content
                    for message in self.messages
                ]
            )
        else:
            return "\n".join([message.content for message in self.messages])