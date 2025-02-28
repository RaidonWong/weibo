from typing import List, Union

from pydantic import Field
from typing import Optional, Union, Sequence,TYPE_CHECKING
from pydantic import Field,BaseModel,field_validator
from agentscope.message import Msg
from agentscope.agents import AgentBase
from agentscope.parsers import ParserBase
from agentscope.memory import MemoryBase,TemporaryMemory
from datetime import datetime as dt
from twitter_utils import TwitterMessage,TwitterPage,InfoBox,Registry
from twitter_env import TwitterEnvironment,BaseEnvironment
from logging import getLogger
from agentscope.memory import MemoryBase,TemporaryMemory
from myllm import BaseLLM,get_embedding
import json
import os
from typing import Iterable, Sequence
from typing import Optional
from typing import Union
from typing import Callable
from aiohttp import ClientSession
from loguru import logger

from agentscope.memory import MemoryBase
from agentscope.manager import ModelManager
from agentscope.serialize import serialize, deserialize
from agentscope.service.retrieval.retrieval_from_list import retrieve_from_list
from agentscope.service.retrieval.similarity import Embedding
from agentscope.message import Msg
from agentscope.rpc import AsyncResult
from twitter_utils import TwitterMessage, TwitterPage, InfoBox
from agentscope.models.openai_model import OpenAIChatWrapper
from tqdm import tqdm
from abc import abstractmethod
import json
import os
import re

import string
import numpy as np
import openai
from openai import OpenAI,AsyncOpenAI
aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)


class BaseMemory(BaseModel):
    @abstractmethod
    def add_message(self, messages: List[Msg]) -> None:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def to_messages(self) -> List[dict]:
        pass


memory_registry = Registry(name="MemoryRegistry")


@memory_registry.register("twitter")
class TwitterMemory(BaseMemory):
    class Config:
        arbitrary_types_allowed = True

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
    llm: BaseLLM

    
    # memory_size: int = 10


    def add_message(self, messages: List[Msg]) -> None:
        for message in messages:
            if message.content in self.memory2embedding:continue
            if isinstance(message, TwitterMessage):
                memory_embedding = message.embedding
            else:
                memory_embedding = get_embedding(self.llm, message.content)
            self.messages.append(message)
            self.embedding2memory[memory_embedding] = message.content
            self.memory2embedding[message.content] = memory_embedding
        # self.messages = self.messages[-self.memory_size:]

    def to_string(self, add_sender_prefix: bool = False) -> str:
        messages = self.messages
        if len(messages)>30:
            messages = messages[-30:]
        for m in messages:
            if 'PersonWithCold' in m.content or 'PersonInNeed' in m.content or 'cold' in m.content or 'diagnosis' in m.content:
                messages.remove(m)
        if add_sender_prefix:
            return "\n".join(
                [
                    f"[{message.name}]: {message.content}"
                    if message.name != ""
                    else message.content
                    for message in self.messages
                ]
            )
        else:
            return "\n".join([message.content for message in self.messages])


    def reset(self) -> None:
        self.messages = []
        
    def add(
        self,
        memories: Union[Sequence[Msg], Msg, None],
        embed: bool = False,
    ) -> None:
        """
        Adding new memory fragment, depending on how the memory are stored
        Args:
            memories (`Union[Sequence[Msg], Msg, None]`):
                Memories to be added.
            embed (`bool`):
                Whether to generate embedding for the new added memories
        """
        if memories is None:
            return

        if not isinstance(memories, Sequence):
            record_memories = [memories]
        else:
            record_memories = memories

        # FIXME: a single message may be inserted multiple times
        # Assert the message types
        memories_idx = set(_.id for _ in self.messages if hasattr(_, "id"))
        for memory_unit in record_memories:
            # in case this is a PlaceholderMessage, try to update
            # the values first
            if isinstance(memory_unit, AsyncResult):
                memory_unit = memory_unit.result()

            if not isinstance(memory_unit, Msg):
                raise ValueError(
                    f"Cannot add {type(memory_unit)} to memory, "
                    f"must be a Msg object.",
                )

            # Add to memory if it's new
            if memory_unit.id not in memories_idx:
                if embed:
                    if self.embedding_model:
                        # TODO: embed only content or its string representation
                        memory_unit.embedding = self.embedding_model(
                            [memory_unit],
                            return_embedding_only=True,
                        )
                    else:
                        raise RuntimeError("Embedding model is not provided.")
                self.messages.append(memory_unit)

    def delete(self, index: Union[Iterable, int]) -> None:
        """
        Delete memory fragment, depending on how the memory are stored
        and matched
        Args:
            index (Union[Iterable, int]):
                indices of the memory fragments to delete
        """
        if self.size() == 0:
            logger.warning(
                "The memory is empty, and the delete operation is "
                "skipping.",
            )
            return

        if isinstance(index, int):
            index = [index]

        if isinstance(index, list):
            index = set(index)

            invalid_index = [_ for _ in index if _ >= self.size() or _ < 0]
            if len(invalid_index) > 0:
                logger.warning(
                    f"Skip delete operation for the invalid "
                    f"index {invalid_index}",
                )

            self.messages = [
                _ for i, _ in enumerate(self.messages) if i not in index
            ]
        else:
            raise NotImplementedError(
                "index type only supports {None, int, list}",
            )

    def export(
        self,
        file_path: Optional[str] = None,
        to_mem: bool = False,
    ) -> Optional[list]:
        """
        Export memory, depending on how the memory are stored
        Args:
            file_path (Optional[str]):
                file path to save the memory to. The messages will
                be serialized and written to the file.
            to_mem (Optional[str]):
                if True, just return the list of messages in memory
        Notice: this method prevents file_path is None when to_mem
        is False.
        """
        if to_mem:
            return self.messages

        if to_mem is False and file_path is not None:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(serialize(self.messages))
        else:
            raise NotImplementedError(
                "file type only supports "
                "{json, yaml, pkl}, default is json",
            )
        return None

    def load(
        self,
        memories: Union[str, list[Msg], Msg],
        overwrite: bool = False,
    ) -> None:
        """
        Load memory, depending on how the memory are passed, design to load
        from both file or dict
        Args:
            memories (Union[str, list[Msg], Msg]):
                memories to be loaded.
                If it is in str type, it will be first checked if it is a
                file; otherwise it will be deserialized as messages.
                Otherwise, memories must be either in message type or list
                 of messages.
            overwrite (bool):
                if True, clear the current memory before loading the new ones;
                if False, memories will be appended to the old one at the end.
        """
        if isinstance(memories, str):
            if os.path.isfile(memories):
                with open(memories, "r", encoding="utf-8") as f:
                    load_memories = deserialize(f.read())
            else:
                try:
                    load_memories = deserialize(memories)
                    if not isinstance(load_memories, dict) and not isinstance(
                        load_memories,
                        list,
                    ):
                        logger.warning(
                            "The memory loaded by json.loads is "
                            "neither a dict nor a list, which may "
                            "cause unpredictable errors.",
                        )
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Cannot load [{memories}] via " f"json.loads.",
                        e.doc,
                        e.pos,
                    )
        elif isinstance(memories, list):
            for unit in memories:
                if not isinstance(unit, Msg):
                    raise TypeError(
                        f"Expect a list of Msg objects, but get {type(unit)} "
                        f"instead.",
                    )
            load_memories = memories
        elif isinstance(memories, Msg):
            load_memories = [memories]
        else:
            raise TypeError(
                f"The type of memories to be loaded is not supported. "
                f"Expect str, list[Msg], or Msg, but get {type(memories)}.",
            )

        # overwrite the original memories after loading the new ones
        if overwrite:
            self.clear()

        self.add(load_memories)

    def clear(self) -> None:
        """Clean memory, depending on how the memory are stored"""
        self.messages = []

    def size(self) -> int:
        """Returns the number of memory segments in memory."""
        return len(self.messages)

    def retrieve_by_embedding(
        self,
        query: Union[str, Embedding],
        metric: Callable[[Embedding, Embedding], float],
        top_k: int = 1,
        preserve_order: bool = True,
        embedding_model: Callable[[Union[str, dict]], Embedding] = None,
    ) -> list[dict]:
        """Retrieve memory by their embeddings.

        Args:
            query (`Union[str, Embedding]`):
                Query string or embedding.
            metric (`Callable[[Embedding, Embedding], float]`):
                A metric to compute the relevance between embeddings of query
                and memory. In default, higher relevance means better match,
                and you can set `reverse` to `True` to reverse the order.
            top_k (`int`, defaults to `1`):
                The number of memory units to retrieve.
            preserve_order (`bool`, defaults to `True`):
                Whether to preserve the original order of the retrieved memory
                units.
            embedding_model (`Callable[[Union[str, dict]], Embedding]`, \
                defaults to `None`):
                A callable object to embed the memory unit. If not provided, it
                will use the default embedding model.

        Returns:
            `list[dict]`: a list of retrieved memory units in
            specific order.
        """

        retrieved_items = retrieve_from_list(
            query,
            self.get_embeddings(embedding_model or self.embedding_model),
            metric,
            top_k,
            self.embedding_model,
            preserve_order,
        ).content

        # obtain the corresponding memory item
        response = []
        for score, index, _ in retrieved_items:
            response.append(
                {
                    "score": score,
                    "index": index,
                    "memory": self.messages[index],
                },
            )

        return response

    def get_embeddings(
        self,
        embedding_model: Callable[[Union[str, dict]], Embedding] = None,
    ) -> list:
        """Get embeddings of all memory units. If `embedding_model` is
        provided, the memory units that doesn't have `embedding` attribute
        will be embedded. Otherwise, its embedding will be `None`.

        Args:
            embedding_model
                (`Callable[[Union[str, dict]], Embedding]`, defaults to
                `None`):
                Embedding model or embedding vector.

        Returns:
            `list[Union[Embedding, None]]`: List of embeddings or None.
        """
        embeddings = []
        for memory_unit in self.messages:
            if memory_unit.embedding is None and embedding_model is not None:
                # embedding
                # TODO: embed only content or its string representation
                memory_unit.embedding = embedding_model(memory_unit)
            embeddings.append(memory_unit.embedding)
        return embeddings

    def get_memory(
        self,
        recent_n: Optional[int] = None,
        filter_func: Optional[Callable[[int, dict], bool]] = None,
    ) -> list:
        """Retrieve memory.

        Args:
            recent_n (`Optional[int]`, default `None`):
                The last number of memories to return.
            filter_func
                (`Callable[[int, dict], bool]`, default to `None`):
                The function to filter memories, which take the index and
                memory unit as input, and return a boolean value.
        """
        # extract the recent `recent_n` entries in memories
        if recent_n is None:
            memories = self.messages
        else:
            if recent_n > self.size():
                logger.warning(
                    "The retrieved number of memories {} is "
                    "greater than the total number of memories {"
                    "}",
                    recent_n,
                    self.size(),
                )
            memories = self.messages[-recent_n:]

        # filter the memories
        if filter_func is not None:
            memories = [_ for i, _ in enumerate(memories) if filter_func(i, _)]

        return memories





@memory_registry.register("vectorstore")
class VectorStoreMemory(BaseMemory):
    class Config:
        arbitrary_types_allowed = True

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
    llm: BaseLLM

    def add_message(self, messages: List[Msg]) -> None:
        for message in messages:
            self.messages.append(message)
            memory_embedding = get_embedding(self.llm, message.content)
            self.embedding2memory[memory_embedding] = message.content
            self.memory2embedding[message.content] = memory_embedding

    def to_string(self, add_sender_prefix: bool = False) -> str:
        messages = self.messages
        for m in messages:
            if 'PersonWithCold' in m.content or 'PersonInNeed' in m.content:
                messages.remove(m)
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






@memory_registry.register("personal_history")
class PersonalMemory(BaseMemory):
    class Config:
        arbitrary_types_allowed = True
    messages: List[Msg] = Field(default=[])
    memory_path: str = None
    target: str = "support HK peace"
    top_k: str = 5
    deadline: str = None
    model: str = "gpt-3.5-turbo"
    has_summary: bool = False
    max_summary_length: int = 200
    summary: str = ""
    SUMMARIZATION_PROMPT: str= '''Your task is to create a concise running summary of observations in the provided text, focusing on key and potentially important information to remember.

Please avoid repeating the observations and pay attention to the person's overall leanings. Keep the summary concise in one sentence.

Observations:
"""
{new_events}
"""
'''
    RETRIVEAL_QUERY:str='''What is your opinion on {target} or other political and social issues?'''

    def __init__(self, memory_path, target, top_k, deadline, llm):
        super().__init__()
        self.memory_path = memory_path
        self.target = target
        self.top_k = top_k
        self.deadline = deadline
        self.model = llm
        # load the historical tweets of the user
        if self.memory_path is not None and os.path.exists(self.memory_path):
            print('load ',self.memory_path)
            df = open(self.memory_path,'r',errors='ignore').readlines()
            content_set = set()
            for d in df:
                d = json.loads(d)
                content = d["rawContent"]
                content = re.sub(r"\n+", "\n", content)
                content.replace('\n',' ')
                if content in content_set or len(content.split())<10:continue
                content_set.add(content)
                post_time = d["date"][:19]
                if post_time>self.deadline:continue
                sender = d["user"]["username"]
                if sender !=self.memory_path.split('/')[-1][:-4]:continue
                message = TwitterMessage(name="memory",role="assistant",content=content, post_time=post_time, sender=sender)
                self.messages.append(message)
            
        else:
            print(self.memory_path,' does not exist!')

    def add_message(self, messages: List[Msg]) -> None:
        for message in messages:
            self.messages.append(message)

    def reset(self) -> None:
        self.messages = []

 

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
        async with ClientSession(trust_env=True) as session:
            response = await aclient.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            max_tokens=self.max_summary_length,
            temperature=0.5,
            max_completion_tokens=10,
        )
            
            self.summary =  response.choices[0].message.content   
            message = Msg(content=self.summary)
            self.add_message([message])
        
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