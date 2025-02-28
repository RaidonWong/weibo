# -*- coding: utf-8 -*-
"""An agent that replies in a dictionary format."""
from typing import Optional, Union, Sequence,TYPE_CHECKING,List,Set
from pydantic import Field,BaseModel,field_validator
from agentscope.message import Msg
from agentscope.agents import AgentBase,DictDialogAgent

from agentscope.memory import MemoryBase,TemporaryMemory
from datetime import datetime as dt
from twitter_utils import TwitterMessage,TwitterPage,InfoBox,agenerate_response,Registry
from twitter_env import TwitterEnvironment,BaseEnvironment
from logging import getLogger
from outputparser import OutputParser
from agentscope.memory import MemoryBase
from twittermemory import PersonalMemory
from reflection import BaseMemoryManipulator, Reflection
from string import Template
from myllm import BaseLLM
from twittermemory import BaseMemory
from agentscope.manager import ModelManager
from agentscope.models import ModelResponse

logger = getLogger(__file__)



task_prompt='''In terms of how you actually perform the action, you take action by calling functions. Currently, there are the following functions that can be called.
    - do_nothing(): Do nothing. There is nothing that you like to respond to.
    - post(content): Post a tweet. `content` is the sentence that you will post.
    - retweet(content, author, original_tweet_id, original_tweet). Retweet or quote an existing tweet in your twitter page. `content` is the statements that you add when retweeting. If you want to say nothing, set `content` to None. `author` is the author of the tweet that you want to retweet, it should be the concrete name. `original_tweet_id` and `original_tweet` are the id and content of the retweeted tweet.
    - reply(content, author, original_tweet_id). Reply to an existing tweet in your twitter page or reply one of replies in your notifications, but don't reply to yourself and those not in your tweet page. `content` is what you will reply to the original tweet or other comments. `author` is the author of the original tweet or comment that you want to reply to. `original_tweet_id` is the id of the original tweet.
    - like(author, original_tweet_id). Press like on an existing tweet in your twitter page. `author` is the author of the original tweet that you like. `original_tweet_id` is the id of the original tweet.

    Call one function at a time, please give a thought before calling these actions, i.e., use the following format strictly:

    [OPTION 1]
    Thought: None of the observation attract my attention, I need to:
    Action: do_nothing()

    [OPTION 2]
    Thought: due to `xxx`, I need to:
    Action: post(content="Stop this farce!")

    [OPTION 3]
    Thought: due to `xxx`, I need to:
    Action: retweet(content="I agree with you", author="zzz", original_tweet_id="0", original_tweet="kkk")

    [OPTION 4]
    Thought: due to `xxx`, I need to:
    Action: reply(content="yyy", author="zzz", original_tweet_id="0")

    [OPTION 5]
    Thought: due to `xxx`, I need to:
    Action: like(author="zzz", original_tweet_id="1")

    Now begin your actions as the agent. Remember give a thought after and `Thought:` and only write one function call after `Action:`
    Based on the above history, what will you, ${agent_name}, do next?'''












agent_registry = Registry(name="AgentRegistry")
@agent_registry.register("twitter")
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
        sys_prompt: str="Please action in your role.",
        model_config_name: str = None,
        use_memory: bool = True,
        role_description: str = Field(default=""),
        max_retries: Optional[int] = 3,
        prompt_template: str = Field(default=""),
        async_mode: bool =True,
        current_time: str = None,
        environment: BaseEnvironment = None,
        step_cnt: int = 0,
        page: TwitterPage = None,
        info_box: InfoBox = None,
        output_parser: OutputParser=None,
        personal_history: MemoryBase = None,
        retrieved_memory:str = "",
        data_collector = {},
        receiver: Set[str] = Field(default=set({"all"})),
        atts: list = [],
        llm:BaseLLM=None,
        
        memory: BaseMemory = Field(default_factory=PersonalMemory),
        memory_manipulator: Optional[BaseMemoryManipulator] = None,
    
        context_prompt_template: str = Field(default=""),

        manipulated_memory: str = Field(
            default="", description="one fragment used in prompt construction"
        )
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

        if data_collector is None:
            data_collector = {}  # 防止可变类型的默认值共享问题

        
        if personal_history is None:
            personal_history = PersonalMemory()
        
        # 初始化属性
        self.async_mode = async_mode
        self.current_time = current_time
        self.environment = environment
        self.step_cnt = step_cnt
        self.page = page
        self.info_box = info_box
        self.personal_history = personal_history
        self.retrieved_memory = retrieved_memory
        self.data_collector = data_collector
        self.prompt_template: str = Field(default="")
        self.context_prompt_template = context_prompt_template
        self.manipulated_memory = manipulated_memory
        self.memory_manipulator=memory_manipulator
        self.role_description: str = Field(default="")
        self.max_retry: int = 2
        self.memory=memory
        self.output_parser: OutputParser=output_parser
        self.receiver: Set[str] = Field(default=set({"all"}))
        
       

    @field_validator("current_time")
    def convert_str_to_dt(cls, current_time):
        if not isinstance(current_time, str):
            raise ValueError("current_time should be str")
        try:
            return dt.strptime(current_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError(f"Invalid date format for current_time. Expected format: '%Y-%m-%d %H:%M:%S', but got: {current_time}")


    async def get_personal_experience(self):
        # retrieve and summarize personal experience
        if not self.personal_history.has_summary:
            await self.personal_history.summarize()
        

    






    def _fill_prompt_template(self, env_description: str = "") -> str:
        """Fill the placeholders in the prompt template

        In the twitter agent, these placeholders are supported:
        - ${agent_name}: the name of the agent
        - ${env_description}: the description of the environment
        - ${role_description}: the description of the role of the agent
        - ${personal_history}: the personal experience (tweets) of the user
        - ${chat_history}: the chat history (about this movement) of the agent
        - ${tweet_page}: the tweet page the agent can see
        - ${trigger_news}: desc of the trigger event
        - ${info_box}: replies in notifications
        """
        input_arguments = {
            "agent_name": self.name,
            "role_description": self.role_description,
            "personal_history": self.personal_history.summary,
            # "chat_history": self.memory.to_string(add_sender_prefix=True),
            "chat_history":self.retrieved_memory,
            "current_time": self.current_time,
            "trigger_news": env_description,
            "tweet_page":self.page.to_string() if self.page else "",
            "info_box": self.info_box.to_string()
        }
        return Template(self.context_prompt_template).safe_substitute(input_arguments)

    def _fill_context_for_retrieval(self, env_description: str = "") -> str:
        """Fill the placeholders in the prompt template

        In the twitter agent, these placeholders are supported:
        - ${agent_name}: the name of the agent
        - ${env_description}: the description of the environment
        - ${role_description}: the description of the role of the agent
        - ${personal_history}: the personal experience (tweets) of the user
        - ${chat_history}: the chat history (about this movement) of the agent
        - ${tweet_page}: the tweet page the agent can see
        - ${trigger_news}: desc of the trigger event
        - ${info_box}: replies in notifications
        """
        input_arguments = {
            "agent_name": self.name,
            "role_description": self.role_description,
            "personal_history": self.personal_history.summary,
            "current_time": self.current_time,
            "trigger_news": env_description,
            "tweet_page":self.page.to_string() if self.page else "",
            "info_box": self.info_box.to_string()
        }
        return Template(self.context_prompt_template).safe_substitute(input_arguments)

    def add_message_to_memory(self, messages: List[Msg]) -> None:
        self.memory.add(messages)

    def add_message_to_tweet_page(self, messages: List[Msg]) -> None:
        self.page.add(messages)

    def add_message_to_info_box(self, messages: List[Msg]) -> None:
        self.info_box.add(messages)

    def reset(self, environment: BaseEnvironment) -> None:
        """Reset the agent"""
        self.environment = environment
        self.memory.reset()
        self.memory_manipulator.agent = self
        self.memory_manipulator.memory = self.memory


    
    async def acontext_test(self, context_config) -> Msg:
        """Test the agent given a specific context"""

        self.manipulated_memory = self.memory_manipulator.manipulate_memory()
        if not self.personal_history.has_summary:
            await self.personal_history.summarize()

        input_arguments = {
            "agent_name": self.name,
            "role_description": self.role_description,
            "personal_history": self.personal_history.to_string(add_sender_prefix=True),
            "chat_history": self.memory.to_string(add_sender_prefix=True),
            "current_time": context_config.pop("current_time", ""),
            "trigger_news": context_config.pop("trigger_news", ""),
            "tweet_page":context_config.pop("tweet_page", ""),
            "info_box": context_config.pop("info_box", ""),
        }
        prompt = Template(self.prompt_template).safe_substitute(input_arguments)

        parsed_response, reaction, target, parent_id = None, None, None, None
        for i in range(self.max_retry):
            try:
                response = await agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)

                if "post(" in parsed_response.return_values["output"]:
                    reaction = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'post'
                elif "retweet(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'retweet'
                elif "reply(" in parsed_response.return_values["output"]:
                    reaction, target, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )      
                    reaction_type = 'reply'
                elif "like(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )    
                    reaction_type = 'like'          
                elif "do_nothing(" in parsed_response.return_values["output"]:
                    reaction, target, reaction_type = None, None, None
                else:
                    raise Exception(
                        f"no valid parsed_response detected, "
                        f"cur response {parsed_response.return_values['output']}"
                    )
                break

            except Exception as e:
                logger.error(e)
                logger.warning("Retrying...")
                continue

        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")

        if reaction is None:
            reaction = "Silence"
            reaction_type = 'other'
       

        message = TwitterMessage(
            content="" if reaction is None else reaction,
            post_time = self.current_time,
            msg_type = reaction_type,
            sender = self.name,
            receiver = self.get_receiver()
            if target is None
            else self.get_valid_receiver(target),
            parent_id = parent_id,
        )
        parsed_response = {'response':parsed_response}
        return prompt, message, parsed_response


    








    
    def set_parser(self, parser:OutputParser) -> None:
        """Set response parser, which will provide 1) format instruction; 2)
        response parsing; 3) filtering fields when returning message, storing
        message in memory. So developers only need to change the
        parser, and the agent will work as expected.
        """
        self.parser = parser

    async def reply(self, current_time: dt, env_description: str = "",x: Optional[Union[Msg, Sequence[Msg]]] = None) -> Msg:
        
        self.current_time = current_time

        #self.manipulated_memory=self.memory_manipulator.manipulate_memory()

        # record the input if needed
        if self.memory:
            self.memory.add(x)
        
        env_prompt = self._fill_prompt_template(env_description)
        
        # prepare prompt
        prompt = self.model.format(
            Msg("system", task_prompt, "system"),
            Msg("system", env_prompt, "system"),
            Msg("system", self.sys_prompt, role="system"),
            self.memory
            and Msg("system", content=self.memory.to_string(), role="system")
            or x,  # type: ignore[arg-type]
        )
        
        parsed_response, reaction, target, parent_id = None, None, None, None
        
        # call llm
        #raw_response = self.model(prompt)
        for i in range(self.max_retry):
            try:
                raw_response = await agenerate_response(prompt)
                parsed_response = self.output_parser.parse(raw_response)

                if "post(" in parsed_response.return_values["output"]:
                    reaction = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'post'
                elif "retweet(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )
                    reaction_type = 'retweet'
                elif "reply(" in parsed_response.return_values["output"]:
                    reaction, target, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )      
                    reaction_type = 'reply'
                elif "like(" in parsed_response.return_values["output"]:
                    reaction, parent_id = eval(
                        "self._" + parsed_response.return_values["output"].strip()
                    )    
                    reaction_type = 'like'          
                elif "do_nothing(" in parsed_response.return_values["output"]:
                    reaction, target, reaction_type = None, None, None
                else:
                    raise Exception(
                        f"no valid parsed_response detected, "
                        f"cur response {parsed_response.return_values['output']}"
                    )
                break
            except Exception as e:
                logger.error(e)
                logger.warning("Retrying...")
                continue
        if parsed_response is None:
            logger.error(f"{self.name} failed to generate valid response.")

        if reaction is None:
            reaction = "Silence"
            reaction_type = 'other'

        message = TwitterMessage(
            name=self.name,
            content="" if reaction is None else reaction,
            post_time = current_time,
            role="assistant",
            msg_type = reaction_type,
            sender = self.name,
            receiver = self.get_receiver()
            if target is None
            else self.get_valid_receiver(target),
            parent_id = parent_id,
        )
        if env_description!="":
            msg = Msg(content=env_description, sender='News', role='system',name='News')
            self.add_message_to_memory([msg])
        
        self.step_cnt += 1
        self.data_collector[self.environment.cnt_turn]={}
        self.data_collector[self.environment.cnt_turn]['prompt']=prompt
        self.data_collector[self.environment.cnt_turn]['response']=message
        self.data_collector[self.environment.cnt_turn]['parsed_response']=parsed_response
        
        return message



        """self.speak(raw_response.stream or raw_response.text)

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

        return msg"""
    


    def get_receiver(self) -> Set[str]:
        return self.receiver
    
    def set_receiver(self, receiver: Union[Set[str], str]) -> None:
        if isinstance(receiver, str):
            self.receiver = set({receiver})
        elif isinstance(receiver, set):
            self.receiver = receiver
        else:
            raise ValueError(
                "input argument `receiver` must be a string or a set of string"
            )

    def add_receiver(self, receiver: Union[Set[str], str]) -> None:
        if isinstance(receiver, str):
            self.receiver.add(receiver)
        elif isinstance(receiver, set):
            self.receiver = self.receiver.union(receiver)
        else:
            raise ValueError(
                "input argument `receiver` must be a string or a set of string"
            )

    def remove_receiver(self, receiver: Union[Set[str], str]) -> None:
        if isinstance(receiver, str):
            try:
                self.receiver.remove(receiver)
            except KeyError as e:
                logger.warn(f"Receiver {receiver} not found.")
        elif isinstance(receiver, set):
            self.receiver = self.receiver.difference(receiver)
        else:
            raise ValueError(
                "input argument `receiver` must be a string or a set of string"
            )
        
    def get_valid_receiver(self, target: str) -> set:
        all_agents_name = []
        for agent in self.environment.agents:
            all_agents_name.append(agent.name)

        if not (target in all_agents_name):
            return {"all"}
        else:
            return {target}
        
    def _post(self, content=None):
        if content is None:
            return ""
        else:
            # reaction_content = (
            #     f"{self.name} posts a tweet: '{content}'."
            # )
            reaction_content = content
        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content

    def _retweet(self, content=None, author=None, original_tweet_id=None, original_tweet=None):
        if author is None or original_tweet_id is None: return ""
        try:
            original_tweet = self.environment.tweet_db[original_tweet_id].content
        except:
            # raise Exception("Retweet. Not legal tweet id: {}".format(original_tweet_id))
            logger.warning("Retweet. Not legal tweet id: {}".format(original_tweet_id))
        # original_tweet = original_tweet_id
        if content is None:
            # reaction_content = (
            #         f"{self.name} retweets a tweet of [{author}]: '{original_tweet}'."
            #     )
            reaction_content = (
                    f"Retweets a tweet of [{author}]: '{original_tweet}'."
                )
        else:
            # reaction_content = (
            #     f"{self.name} retweets a tweet of [{author}]: '{original_tweet}' with additional statements: {content}."
            # )
            reaction_content = (
                f"Retweets a tweet of [{author}]: '{original_tweet}' with additional statements: {content}."
            )
        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content, original_tweet_id

    def _reply(self, content=None, author=None, original_tweet_id=None):
        if content is None or author is None or original_tweet_id is None: return ""
        try:
            original_tweet = self.environment.tweet_db[original_tweet_id].content
        except:
            # raise Exception("Comment. Not legal tweet id: {}".format(original_tweet_id))
            logger.warning("Reply. Not legal tweet id: {}".format(original_tweet_id))
        reaction_content = (
            f"{self.name} replies to [{author}]: {content}."
        )
        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content, author, original_tweet_id

    def _like(self, author=None, original_tweet_id=None):
        if author is None or original_tweet_id is None: return ""
        try:
            original_tweet = self.environment.tweet_db[original_tweet_id].content
        except:
            # raise Exception("Like. Not legal tweet id: {}".format(original_tweet_id))
            logger.warning("Like. Not legal tweet id: {}".format(original_tweet_id))
        reaction_content = f"{self.name} likes a tweet of [{author}]: '{original_tweet}'."

        # self.environment.broadcast_observations(self, target, reaction_content)
        return reaction_content, original_tweet_id