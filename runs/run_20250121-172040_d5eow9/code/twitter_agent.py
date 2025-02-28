# -*- coding: utf-8 -*-
"""An agent that replies in a dictionary format."""
from typing import Optional, Union, Sequence
from pydantic import Field,BaseModel
from agentscope.message import Msg
from agentscope.agents import AgentBase
from agentscope.parsers import ParserBase
from agentscope.memory import MemoryBase
from datetime import datetime as dt
from twitter_utils import TwitterMessage,TwitterPage,InfoBox

class TwitterAgent(AgentBase):
    """An agent that generates response in a dict format, where user can
    specify the required fields in the response via specifying the parser

    About parser, please refer to our
    [tutorial](https://modelscope.github.io/agentscope/en/tutorial/203-parser.html)

    For usage example, please refer to the example of werewolf in
    `examples/game_werewolf`"""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model_config_name: str,
        use_memory: bool = True,
        max_retries: Optional[int] = 3,
    ) -> None:
        """Initialize the dict dialog agent.

        Arguments:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`, defaults to `None`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`, defaults to None):
                The name of the model config, which is used to load model from
                configuration.
            use_memory (`bool`, defaults to `True`):
                Whether the agent has memory.
            max_retries (`Optional[int]`, defaults to `None`):
                The maximum number of retries when failed to parse the model
                output.
        """  # noqa
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
            use_memory=use_memory,
        )

        self.parser = None
        self.max_retries = max_retries

    def set_parser(self, parser: ParserBase) -> None:
        """Set response parser, which will provide 1) format instruction; 2)
        response parsing; 3) filtering fields when returning message, storing
        message in memory. So developers only need to change the
        parser, and the agent will work as expected.
        """
        self.parser = parser

    async def reply(self, x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
      
        # record the input if needed
        if self.memory:
            self.memory.add(x)

        # prepare prompt
        prompt = self.model.format(
            Msg("system", self.sys_prompt, role="system"),
            self.memory
            and self.memory.get_memory()
            or x,  # type: ignore[arg-type]
            Msg("system", self.parser.format_instruction, "system"),
        )

        # call llm
        raw_response = self.model(prompt)


        self.speak(raw_response.stream or raw_response.text)

        # Parsing the raw response
        res = self.parser.parse(raw_response)

        # Filter the parsed response by keys for storing in memory, returning
        # in the reply function, and feeding into the metadata field in the
        # returned message object.
        if self.memory:
            self.memory.add(
                Msg(self.name, self.parser.to_memory(res.parsed), "assistant"),
            )

        msg = Msg(
            self.name,
            content=self.parser.to_content(res.parsed),
            role="assistant",
            metadata=self.parser.to_metadata(res.parsed),
        )

        return msg
 