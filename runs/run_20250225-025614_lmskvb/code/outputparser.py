from __future__ import annotations

import re
from abc import abstractmethod
import json
from typing import Union, List, Tuple, NamedTuple, TYPE_CHECKING

from twitter_utils import Registry

from pydantic import BaseModel
from myllm import LLMResult
if TYPE_CHECKING:
    from agentscope.agents import AgentBase
    from twitter_env import BaseEnvironment


class AgentAction(NamedTuple):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class AgentCriticism(NamedTuple):
    """Agent's criticism."""

    is_agree: bool
    criticism: str
    sender_agent: object = None


class OutputParserError(Exception):
    """Exception raised when parsing output from a command fails."""

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "Failed to parse output of the model:%s\n " % self.message


class OutputParser(BaseModel):
    """Base class for output parsers."""

    @abstractmethod
    def parse(self, output: LLMResult) -> NamedTuple:
        pass

output_parser_registry = Registry(name="OutputParserRegistry")
@output_parser_registry.register("twitter")
class TwitterParser(OutputParser):
    def parse(self, output: LLMResult) -> Union[AgentAction, AgentFinish]:
        text = output.content
        text = text.replace('Post(', 'post(').replace('Retweet(','retweet(').replace('Reply(','reply(').replace('Like(','like(')
        cleaned_output = text.strip()
        cleaned_output = re.sub(r"\n+", "\n", cleaned_output)
        cleaned_output = text.replace('\_','_')
        cleaned_output = cleaned_output.split("\n")
        while '' in cleaned_output:
            cleaned_output.remove('')
        if not (
            (
            len(cleaned_output) >= 3
            and cleaned_output[1].startswith("Thought:")
            and cleaned_output[2].startswith("Action:")
            # and cleaned_output[3].startswith("Attitude:")
            )
        or
            (
            len(cleaned_output) >= 2    
            and cleaned_output[0].startswith("Thought:")
            and cleaned_output[1].startswith("Action:")
            # and cleaned_output[2].startswith("Attitude:")
            )
        ):
            raise OutputParserError(text)

        if cleaned_output[1].startswith("Action:"):
            action = cleaned_output[1][len("Action:") :].strip()
            if action.endswith('))'):action = action[:-1]
        else:
            action = cleaned_output[2][len("Action:") :].strip()
            if action.endswith('))'):action = action[:-1]
        
        return AgentFinish({"output": action,}, text)

