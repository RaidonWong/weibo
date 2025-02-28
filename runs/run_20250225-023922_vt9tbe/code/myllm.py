from abc import abstractmethod
from typing import Dict, Any
from twitter_utils import Registry
from pydantic import BaseModel, Field
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import ast
import os
import numpy as np
from aiohttp import ClientSession
from typing import Dict, List, Optional, Union
from mylogging import logger
from jsonrepair import JsonRepair
import openai


LOCAL_LLMS = [
    "llama-2-7b-chat-hf",
    "llama-2-13b-chat-hf",
    "vicuna-7b-v1.1",
    "vicuna-7b-v1.3",
    "vicuna-7b-v1.5",
    "vicuna-13b-v1.5",
    "mistral-7b-instruct-v0.1"
]
LOCAL_LLMS_MAPPING = {
    "llama-2-7b-chat-hf": "/remote-home/share/LLM_CKPT/Llama-2-7b-chat-hf/",
    "llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "vicuna-7b-v1.1": "/remote-home/share/LLM_CKPT/vicuna-7B-v1.1/",
    "vicuna-7b-v1.3": "/remote-home/share/LLM_CKPT/vicuna-7B-v1.3/",
    "vicuna-7b-v1.5": "/remote-home/share/LLM_CKPT/vicuna-7b-v1.5/",
    "vicuna-13b-v1.5": "/remote-home/share/LLM_CKPT/vicuna-13b-v1.5",
    "mistral-7b-instruct-v0.1":"/remote-home/share/LLM_CKPT/Mistral-7B-Instruct-v0.1/",
}

class LLMResult(BaseModel):
    content: str = ""
    function_name: str = ""
    function_arguments: Any = None
    send_tokens: int = 0
    recv_tokens: int = 0
    total_tokens: int = 0


class BaseModelArgs(BaseModel):
    pass


class BaseLLM(BaseModel):
    args: BaseModelArgs = Field(default_factory=BaseModelArgs)
    max_retry: int = Field(default=3)

    @abstractmethod
    def get_spend(self) -> float:
        """
        Number of USD spent
        """
        return -1.0

    @abstractmethod
    def generate_response(self, **kwargs) -> LLMResult:
        pass

    @abstractmethod
    def agenerate_response(self, **kwargs) -> LLMResult:
        pass


class BaseChatModel(BaseLLM):
    pass


class BaseCompletionModel(BaseLLM):
    pass


class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-3.5-turbo")
    deployment_id: str = Field(default='text-embedding-ada-002')
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)



llm_registry = Registry(name="LLMRegistry")


@llm_registry.register("gpt-3.5-turbo")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.model_dump()
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logger.warn(f"Unused arguments: {kwargs}")
        if args["model"] in LOCAL_LLMS:
            openai.api_base = "http://localhost:5000/v1"
        super().__init__(args=args, max_retry=max_retry)

    @classmethod
    def send_token_limit(self, model: str) -> int:
        send_token_limit_dict = {
            "gpt-3.5-turbo": 4096,
            "gpt-35-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "llama-2-7b-chat-hf": 4096,
            "vicuna-7b-v1.5":2048,
            "vicuna-7b-v1.3":2048,
        }

        return send_token_limit_dict[model]

    # def _construct_messages(self, history: List[Message]):
    #     return history + [{"role": "user", "content": query}]
    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def generate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)
        try:
            # Execute function call
            if functions != []:
                response = openai.ChatCompletion.create(
                    messages=messages,
                    functions=functions,
                    **self.args.model_dump(),
                )
                if response["choices"][0]["message"].get("function_call") is not None:
                    self.collect_metrics(response)
                    return LLMResult(
                        content=response["choices"][0]["message"].get("content", ""),
                        function_name=response["choices"][0]["message"][
                            "function_call"
                        ]["name"],
                        function_arguments=ast.literal_eval(
                            response["choices"][0]["message"]["function_call"][
                                "arguments"
                            ]
                        ),
                        send_tokens=response["usage"]["prompt_tokens"],
                        recv_tokens=response["usage"]["completion_tokens"],
                        total_tokens=response["usage"]["total_tokens"],
                    )
                else:
                    self.collect_metrics(response)
                    return LLMResult(
                        content=response["choices"][0]["message"]["content"],
                        send_tokens=response["usage"]["prompt_tokens"],
                        recv_tokens=response["usage"]["completion_tokens"],
                        total_tokens=response["usage"]["total_tokens"],
                    )

            else:
                response = openai.ChatCompletion.create(
                    messages=messages,
                    **self.args.dict(),
                )
                self.collect_metrics(response)
                return LLMResult(
                    content=response["choices"][0]["message"]["content"],
                    send_tokens=response["usage"]["prompt_tokens"],
                    recv_tokens=response["usage"]["completion_tokens"],
                    total_tokens=response["usage"]["total_tokens"],
                )
        except ( KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            raise

    @retry(
        stop=stop_after_attempt(20),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def agenerate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)

        try:
            if functions != []:
                async with ClientSession(trust_env=True) as session:
        # 这里已经不需要再设置 aiosession
                    response = await openai.ChatCompletion.acreate(
                        messages=messages,
                        functions=functions,
                        **self.args.model_dump(),  # 使用参数传递给 API
        )
                if response["choices"][0]["message"].get("function_call") is not None:
                    function_name = response["choices"][0]["message"]["function_call"][
                        "name"
                    ]
                    valid_function = False
                    if function_name.startswith("function."):
                        function_name = function_name.replace("function.", "")
                    elif function_name.startswith("functions."):
                        function_name = function_name.replace("functions.", "")
                    for function in functions:
                        if function["name"] == function_name:
                            valid_function = True
                            break
                    if not valid_function:
                        logger.warn(
                            f"The returned function name {function_name} is not in the list of valid functions. Retrying..."
                        )
                        raise ValueError(
                            f"The returned function name {function_name} is not in the list of valid functions."
                        )
                    try:
                        arguments = ast.literal_eval(
                            response["choices"][0]["message"]["function_call"][
                                "arguments"
                            ]
                        )
                    except:
                        try:
                            arguments = ast.literal_eval(
                                JsonRepair(
                                    response["choices"][0]["message"]["function_call"][
                                        "arguments"
                                    ]
                                ).repair()
                            )
                        except:
                            logger.warn(
                                "The returned argument in function call is not valid json. Retrying..."
                            )
                            raise ValueError(
                                "The returned argument in function call is not valid json."
                            )
                    self.collect_metrics(response)
                    return LLMResult(
                        function_name=function_name,
                        function_arguments=arguments,
                        send_tokens=response["usage"]["prompt_tokens"],
                        recv_tokens=response["usage"]["completion_tokens"],
                        total_tokens=response["usage"]["total_tokens"],
                    )

                else:
                    self.collect_metrics(response)
                    return LLMResult(
                        content=response["choices"][0]["message"]["content"],
                        send_tokens=response["usage"]["prompt_tokens"],
                        recv_tokens=response["usage"]["completion_tokens"],
                        total_tokens=response["usage"]["total_tokens"],
                    )

            else:
                async with ClientSession(trust_env=True) as session:
                    response = await openai.ChatCompletion.acreate(
                        messages=messages,
                        **self.args.model_dump(),
        )
                self.collect_metrics(response)
                return LLMResult(
                    content=response["choices"][0]["message"]["content"],
                    send_tokens=response["usage"]["prompt_tokens"],
                    recv_tokens=response["usage"]["completion_tokens"],
                    total_tokens=response["usage"]["total_tokens"],
                )
        except (KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
            raise

    def construct_messages(
        self, prepend_prompt: str, history: List[dict], append_prompt: str
    ):
        messages = []
        if prepend_prompt != "":
            messages.append({"role": "system", "content": prepend_prompt})
        if len(history) > 0:
            messages += history
        if append_prompt != "":
            messages.append({"role": "user", "content": append_prompt})
        return messages

    def wrap_prompt(self,messages):
        """
        Wrap the messages with prompts for open-source LLMs like vicuna
        """
        pass


    def collect_metrics(self, response):
        self.total_prompt_tokens += response["usage"]["prompt_tokens"]
        self.total_completion_tokens += response["usage"]["completion_tokens"]

    def get_spend(self) -> int:
        input_cost_map = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-3.5-turbo-16k": 0.003,
            "gpt-3.5-turbo-0613": 0.0015,
            "gpt-3.5-turbo-0125": 0.006,
            "gpt-3.5-turbo-16k-0613": 0.003,
            "gpt-4": 0.03,
            "gpt-4-0613": 0.03,
            "gpt-4-32k": 0.06,
            "llama-2-7b-chat-hf": 0.0,
            "vicuna-7b-v1.5":0.0,
            "vicuna-7b-v1.3":0.0,
        }

        output_cost_map = {
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.004,
            "gpt-3.5-turbo-0613": 0.002,
            "gpt-3.5-turbo-0125": 0.018,
            "gpt-3.5-turbo-16k-0613": 0.004,
            "gpt-4": 0.06,
            "gpt-4-0613": 0.06,
            "gpt-4-32k": 0.12,
            "llama-2-7b-chat-hf": 0.0,
            "vicuna-7b-v1.5":0.0,
            "vicuna-7b-v1.3":0.0,
        }

        model = self.args.model
        if model not in input_cost_map or model not in output_cost_map:
            raise ValueError(f"Model type {model} not supported")

        return (
            self.total_prompt_tokens * input_cost_map[model] / 1000.0
            + self.total_completion_tokens * output_cost_map[model] / 1000.0
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def get_embedding(llm, text: str, attempts=3) -> np.array:
    try:
        text = text.replace("\n", " ")
        if llm.args.model.startswith('gpt'):
            if openai.api_type == "azure":
                embedding = openai.Embedding.create(
                    input=[text], deployment_id="text-embedding-ada-002"
                )["data"][0]["embedding"]
            else:
                embedding = openai.Embedding.create(
                    input=[text], model="text-embedding-ada-002"
                )["data"][0]["embedding"]
            return tuple(embedding)
        else:
            # for local models
            embedding = openai.Embedding.create(
                input=[text], model=llm.args.model
            )["data"][0]["embedding"]
            return tuple(embedding)            
    except Exception as e:
        attempts += 1
        logger.error(f"Error {e} when requesting openai models. Retrying")
        raise




async def aget_embedding(llm, text: str, attempts=3) -> np.array:
    try:
        text = text.replace("\n", " ")
        if llm.args.model.startswith('gpt'):
            if openai.api_type == "azure":
                embedding = openai.Embedding.create(
                    input=[text], deployment_id="text-embedding-ada-002"
                )["data"][0]["embedding"]
            else:
                async with ClientSession(trust_env=True) as session:
        # 不再需要手动设置 aiosession
                    embedding = await openai.Embedding.acreate(
                        input=[text], 
                        model="text-embedding-ada-002"
                )["data"][0]["embedding"]
            return tuple(embedding)
        else:
            # for local models
            embedding = openai.Embedding.create(
                input=[text], model=llm.args.model
            )["data"][0]["embedding"]
            return tuple(embedding)            
    except Exception as e:
        attempts += 1
        logger.error(f"Error {e} when requesting openai models. Retrying")
        raise
