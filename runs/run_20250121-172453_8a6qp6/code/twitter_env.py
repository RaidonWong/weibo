import asyncio
from asyncio.log import logger
from typing import Any, Dict, List

from datetime import datetime as dt
import datetime
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel

from pydantic import Field,BaseModel
import json
import logging
import os
import openai
import copy
from typing import List, Optional, Tuple, Dict,Sequence,Union,Set
from string import Template
from pydantic import Field,BaseModel
import random   
from datetime import datetime as dt
from agentscope.message import Msg
from agentscope.memory import MemoryBase
from agentscope.agents import AgentBase
from agentscope.utils.token_utils import num_tokens_from_content,count_openai_token
from abc import ABC
from abc import abstractmethod
from typing import Any
from agentscope.msghub import msghub
import pickle
import mesa

class BaseEnvironment(BaseModel):
    """
    Base class for environment.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """






    agents: List[AgentBase]
    rule: BaseRule
    max_turns: int = 10
    cnt_turn: int = 0
    last_messages: List[Msg] = []
    rule_params: Dict = {}

    @abstractmethod
    async def step(self) -> List[Msg]:
        """Run one step of the environment"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the environment"""
        pass

 

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns

    def save_data_collector(self) -> None:
        pass



class BaseRule(BaseModel):
    pass


class BaseOrder(BaseModel):
    @abstractmethod
    def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
        """Return the index of the next agent to speak"""

    def reset(self) -> None:
        pass

class BaseUpdater(BaseModel):
    """
    The base class of updater class.
    """

    @abstractmethod
    def update_memory(self, environment: BaseEnvironment):
        pass

    def reset(self):
        pass

class BaseVisibility(BaseModel):
    @abstractmethod
    def update_visible_agents(self, environment: BaseEnvironment):
        """Update the set of visible agents for the agent"""

    def reset(self):
        pass

class BaseDescriber(BaseModel):
    @abstractmethod
    def get_env_description(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> List[str]:
        """Return the environment description for each agent"""
        pass

    def reset(self) -> None:
        pass

class BaseSelector(BaseModel):
    """
    Base class for all selecters
    """

    @abstractmethod
    def select_message(
        self, environment: BaseEnvironment, messages: List[Msg]
    ) -> List[Msg]:
        """Selects a set of valid messages from all messages"""
        pass

    def reset(self) -> None:
        pass


class Registry(BaseModel):
    """Registry for storing and building classes."""

    name: str
    entries: Dict = {}

    def register(self, key: str):
        def decorator(class_builder):
            self.entries[key] = class_builder
            return class_builder

        return decorator

    def build(self, type: str, **kwargs):
        if type not in self.entries:
            raise ValueError(
                f'{type} is not registered. Please register with the .register("{type}") method provided in {self.name} registry'
            )
        return self.entries[type](**kwargs)

    def get_all_entries(self):
        return self.entries


order_registry = Registry(name="OrderRegistry")
visibility_registry = Registry(name="VisibilityRegistry")
selector_registry = Registry(name="SelectorRegistry")
updater_registry = Registry(name="UpdaterRegistry")
describer_registry = Registry(name="DescriberRegistry")
class SimulationRule(BaseRule):
    """
    Rule for the environment. It controls the speaking order of the agents
    and maintain the set of visible agents for each agent.
    """

    order: BaseOrder
    visibility: BaseVisibility
    selector: BaseSelector
    updater: BaseUpdater
    describer: BaseDescriber

    def __init__(
        self,
        order_config,
        visibility_config,
        selector_config,
        updater_config,
        describer_config,
    ):
        order = order_registry.build(**order_config)
        visibility = visibility_registry.build(**visibility_config)
        selector = selector_registry.build(**selector_config)
        updater = updater_registry.build(**updater_config)
        describer = describer_registry.build(**describer_config)
        super().__init__(
            order=order,
            visibility=visibility,
            selector=selector,
            updater=updater,
            describer=describer,
        )

    def get_next_agent_idx(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> List[int]:
        """Return the index of the next agent to speak"""
        return self.order.get_next_agent_idx(environment, *args, **kwargs)

    def update_visible_agents(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> None:
        """Update the set of visible agents for the agent"""
        self.visibility.update_visible_agents(environment, *args, **kwargs)

    def select_message(
        self, environment: BaseEnvironment, messages: List[Msg], *args, **kwargs
    ) -> List[Msg]:
        """Select a set of valid messages from all the generated messages"""
        return self.selector.select_message(environment, messages, *args, **kwargs)

    def update_memory(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the memory of the agent who is able to see that message"""
        self.updater.update_memory(environment, *args, **kwargs)

    def get_env_description(
        self, environment: BaseEnvironment, *args, **kwargs
    ) -> List[str]:
        """Return the description of the environment for each agent"""
        return self.describer.get_env_description(environment, *args, **kwargs)

    def reset(self) -> None:
        self.order.reset()
        self.visibility.reset()
        self.selector.reset()
        self.updater.reset()
        self.describer.reset()

class TwitterMessage(Msg):
    post_time: dt
    msg_type: str='other'
    tweet_id: str=None
    parent_id: str=None
    num_rt: int=0
    num_cmt: int=0
    num_like: int=0
    receiver: Set[str] = Field(default=set())
    embedding:list=[]
    need_embedding: bool=True

class TwitterRule(SimulationRule):
    """
    Rule for the environment. It controls the speaking order of the agents
    and maintain the set of visible agents for each agent.
    """

    def update_tweet_page(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the tweet page of the agent who is able to see that message"""
        self.updater.update_tweet_page(environment, *args, **kwargs)

    def update_info_box(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the tweet page of the agent who is able to see that message"""
        self.updater.update_info_box(environment, *args, **kwargs)

    def update_memory(self, environment: BaseEnvironment, *args, **kwargs) -> None:
        """For each message, add it to the memory of the agent who is able to see that message"""
        self.updater.update_memory(environment, *args, **kwargs)

    def update_tweet_db(self, environment: BaseEnvironment, *args, **kwargs):
        messages = environment.last_messages
        for m in messages:
            if isinstance(m, TwitterMessage) and m.msg_type == 'post':
                idx = str(len(environment.tweet_db))
                environment.tweet_db[idx] = m
                m.tweet_id = str(idx)
            # update num_rt of original tweet
            elif isinstance(m, TwitterMessage) and m.msg_type == 'retweet':
                idx = str(len(environment.tweet_db))
                environment.tweet_db[idx] = m
                m.tweet_id = str(idx)                
                idx = m.parent_id
                if idx in environment.tweet_db:
                    environment.tweet_db[idx].num_rt+=1
            # update num_cmt of original tweet
            elif isinstance(m, TwitterMessage) and m.msg_type == 'comment':
                idx = m.parent_id
                environment.tweet_db[idx].num_cmt+=1
            # update num_like of original tweet
            elif isinstance(m, TwitterMessage) and m.msg_type == 'like':
                idx = m.parent_id
                environment.tweet_db[idx].num_like+=1

    def update_tweet_db_for_news(self, environment, author, content):
        idx = str(len(environment.tweet_db))
        m = TwitterMessage(
            content=content,
            sender=author, 
            receiver=set({"all"}),
            post_time=environment.current_time,
            msg_type='post',
            tweet_id=idx,
            parent_id=None,
            num_rt=0,
            num_cmt=0,
            num_like=0,         
        )
        environment.tweet_db[idx] = m    
        return [m]   

    def update_tweet_page_for_news(self, environment: BaseEnvironment,msg_lst) -> None:
        """For each message, add it to the tweet page of the agent who is able to see that message"""
        self.updater.update_tweet_page_for_news(environment, msg_lst)







class TwitterEnvironment(BaseEnvironment):
    
    """
    Environment used in Observation-Planning-Reflection agent architecture.

    Args:
        agents: List of agents
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
        current_time
        time_delta: time difference between steps
        trigger_news: Dict, time(turn index) and desc of emergent events
    """

    agents: List[AgentBase]
    rule: TwitterRule
    max_turns: int = 10
    cnt_turn: int = 0
    last_messages: List[Msg] = []
    rule_params: Dict = {}
    current_time: dt = dt.now()
    time_delta: int = 120
    trigger_news: Dict={}
    # tweet_db(firehose): store the tweets of all users; key: tweet_id, value: message
    tweet_db = {}
    output_path=""
    target="Metoo Movement"
    abm_model:mesa.Model = None
    class Config:
        arbitrary_types_allowed = True
    # @validator("time_delta")
    # def convert_str_to_timedelta(cls, string):
    #
    #     return datetime.timedelta(seconds=int(string))

    def __init__(self, rule, **kwargs):
        rule_config = rule
        order_config = rule_config.get("order", {"type": "sequential"})
        visibility_config = rule_config.get("visibility", {"type": "all"})
        selector_config = rule_config.get("selector", {"type": "basic"})
        updater_config = rule_config.get("updater", {"type": "basic"})
        describer_config = rule_config.get("describer", {"type": "basic"})
        rule = TwitterRule(
            order_config,
            visibility_config,
            selector_config,
            updater_config,
            describer_config,
        )

        super().__init__(rule=rule, **kwargs)
        self.rule.update_visible_agents(self)

    async def step(self) -> List[Msg]:
        """Run one step of the environment"""

        logger.info(f"Tick tock. Current time: {self.current_time}")

        # Get the next agent index
        agent_ids = self.rule.get_next_agent_idx(self)

        # Get the personal experience of each agent
        await asyncio.gather(
                    *[
                        self.agents[i].get_personal_experience()
                        for i in agent_ids
                    ]
        )   

        # Generate current environment description
        env_descriptions = self.rule.get_env_description(self)

        # check whether the news is a tweet; if so, add to the tweet_db
        self.check_tweet(env_descriptions)
        env_descriptions = self.rule.get_env_description(self)

        # Generate the next message
        messages = await asyncio.gather(
            *[
                self.agents[i].astep(self.current_time, env_descriptions[i])
                for i in agent_ids
            ]
        )

        # Some rules will select certain messages from all the messages
        selected_messages = self.rule.select_message(self, messages)
        self.last_messages = selected_messages
        self.print_messages(selected_messages)

        # Update opinion of mirror and other naive agents
        # update naive agents
        if self.abm_model is not None:
            self.abm_model.step()
            # then substitude the value of mirror using LLM results
            for i in agent_ids:
                self.abm_model.update_mirror(self.agents[i].name, self.agents[i].atts[-1])

        # Update the database of public tweets
        self.rule.update_tweet_db(self)
        print('Tweet Database Updated.')

        # Update the memory of the agents
        self.rule.update_memory(self)
        print('Agent Memory Updated.')

        # Update tweet page of agents
        self.rule.update_tweet_page(self)
        print('Tweet Pages Updated.')

        # TODO: Update the notifications(info box) of agents
        self.rule.update_info_box(self)
        print('Tweet Infobox Updated.')

        # Update the set of visible agents for each agent
        self.rule.update_visible_agents(self)
        print('Visible Agents Updated.')

        self.cnt_turn += 1

        # update current_time
        self.tick_tock()

        return selected_messages

    def print_messages(self, messages: List[Msg]) -> None:
        for message in messages:
            if message is not None:
                logger.info(f"{message.sender}: {message.content}")

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
        AgentBase.update_forward_refs()
        for agent in self.agents:
            agent.reset(environment=self)

    def is_done(self) -> bool:
        """Check if the environment is done"""
        return self.cnt_turn >= self.max_turns

    def tick_tock(self) -> None:
        """Increment the time"""
        self.current_time = self.current_time + datetime.timedelta(
            seconds=self.time_delta
        )

    def save_data_collector(self) -> None:
        """Output the data collector to the target file"""
        data = {}
        for agent in self.agents:
            data[agent.name] = agent.data_collector
        # naive agents in ABM model
        if self.abm_model is not None:
            opinion = {}
            for agent in self.abm_model.schedule.agents:
                opinion[agent.name] = agent.att[-1]
            data['opinion_results'] = opinion
        print('Output to {}'.format(self.output_path))
        with open(self.output_path,'wb') as f:
            pickle.dump(data, f)

    def check_tweet(self, env_descptions):
        cnt_turn = self.cnt_turn
        if 'posts a tweet' in env_descptions[0]:
            author = env_descptions[0][:env_descptions[0].index('posts a tweet')].strip()
            content = env_descptions[0]
            msg_lst = self.rule.update_tweet_db_for_news(self, author, content)
            self.rule.update_tweet_page_for_news(self, msg_lst)
            # del the trigger news
            self.trigger_news[cnt_turn]=""

    async def test(self, agent, context) -> List[Msg]:
        """Run one step of the environment"""
        """Test the system from micro-level"""

        # Generate the next message
        prompt, message, parsed_response = await agent.acontext_test(context)

        return prompt, message, parsed_response