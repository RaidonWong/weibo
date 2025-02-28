from __future__ import annotations
import asyncio
from loguru import logger

from datetime import datetime as dt
import datetime
from twitter_utils import TwitterMessage,Registry

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List
import json
import logging
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import openai
from openai import OpenAI
client = OpenAI()
import copy
from typing import List, Optional, Tuple, Dict,Sequence,Union,Set
from string import Template
from pydantic import Field,BaseModel
import random   
from mylogging import logger
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
    class Config:
        arbitrary_types_allowed = True
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






env_registry = Registry(name="EnvRegistry")
@env_registry.register("twitter")
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
    tweet_db:Dict= {}
    output_path:str=""
    target:str="HK Security"
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
                self.agents[i].reply(self.current_time, env_descriptions[i])
                for i in agent_ids
            ]
        )

        # Some rules will select certain messages from all the messages
        selected_messages = self.rule.select_message(self, messages)
        self.last_messages = selected_messages
        self.print_messages(selected_messages)

        # Update opinion of mirror and other naive agents
        # update naive agents
        

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
                logger.info(f"{message.name}: {message.content}")

    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0
        self.rule.reset()
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


describer_registry = Registry(name="DescriberRegistry")
@describer_registry.register("twitter")
class TwitterDescriber(BaseDescriber):
    
    def get_env_description(self, environment: BaseEnvironment) -> List[str]:
        """Return the environment description for each agent"""
        cnt_turn = environment.cnt_turn
        trigger_news = environment.trigger_news
        if cnt_turn in trigger_news:
            # broadcast the event news
            return [trigger_news[cnt_turn] for _ in range(len(environment.agents))]
        else:
            return ["" for _ in range(len(environment.agents))]

    def reset(self) -> None:
        pass


order_registry = Registry(name="OrderRegistry")

@order_registry.register("twitter")
class TwitterOrder(BaseOrder):
    """
    The agents speak concurrently in a random order
    """

    def get_next_agent_idx(self, environment: BaseEnvironment) -> List[int]:
        res = list(range(len(environment.agents)))
        random.shuffle(res)
        return res
    
selector_registry = Registry(name="SelectorRegistry")
@selector_registry.register("basic")
class BasicSelector(BaseSelector):
    """
    Base class for all selecters
    """

    def select_message(
        self, environment: BaseEnvironment, messages: List[Msg]
    ) -> List[Msg]:
        """Selects a set of valid messages from all messages"""
        return messages

    def reset(self) -> None:
        pass

updater_registry = Registry(name="UpdaterRegistry")
@updater_registry.register("twitter")
class TwitterUpdater(BaseUpdater):
    """
    The basic version of updater.
    The messages will be seen by all the receiver specified in the message.
    """

    def update_memory(self, environment: BaseEnvironment):
        added = False
        for message in environment.last_messages:
            
            if message.content == "":
                continue
            added |= self.add_message_to_all_agents(environment.agents, message)
        # If no one speaks in this turn. Add an empty message to all agents
        if not added:
            for agent in environment.agents:
                agent.add_message_to_memory([Msg(content="[Silence]")])

    def update_tweet_page_for_news(self, environment, msg_lst):
        # filter the message, only reserve post and retweet to be illustrated in the main page
        for message in msg_lst:
            self.add_tweet_to_all_agents(environment.agents, message)

    def update_tweet_page(self, environment: BaseEnvironment):
        # filter the message, only reserve post and retweet to be illustrated in the main page
        for message in environment.last_messages:
            if message.msg_type in ['post','retweet']:
                self.add_tweet_to_all_agents(environment.agents, message)

    def update_info_box(self, environment: BaseEnvironment):
        # filter the message, only reserve post and retweet to be illustrated in the main page
        for message in environment.last_messages:
            if message.msg_type in ['comment']:
                self.add_info_to_all_agents(environment.agents, message)

    def add_tweet_to_all_agents(
        self, agents: List[AgentBase], message: Msg
    ) -> bool:
        if "all" in message.receiver:
            # If receiver is all, then add the message to all agents
            for agent in agents:
                agent.add_message_to_tweet_page([message])
            return True
        else:
            # If receiver is not all, then add the message to the specified agents
            receiver_set = copy.deepcopy(message.receiver)
            for agent in agents:
                if agent.name in receiver_set:
                    agent.add_message_to_tweet_page([message])
                    receiver_set.remove(agent.name)
            if len(receiver_set) > 0:
                missing_receiver = ", ".join(list(receiver_set))
                # raise ValueError(
                #    "Receiver {} not found. Message discarded".format(missing_receiver)
                # )
                logger.warn(
                    "Receiver {} not found. Message discarded".format(missing_receiver)
                )
            return True

    def add_info_to_all_agents(
        self, agents: List[AgentBase], message: Msg
    ) -> bool:
        receiver_set = copy.deepcopy(message.receiver)
        for agent in agents:
            if agent.name in receiver_set:
                agent.add_message_to_info_box([message])
                receiver_set.remove(agent.name)
        if len(receiver_set) > 0:
            missing_receiver = ", ".join(list(receiver_set))
            logger.warn(
                "Receiver {} not found. Message discarded".format(missing_receiver)
            )
        return True
    
    def add_tool_response(
        self,
        name: str,
        agents: List[AgentBase],
        tool_response: List[str],
    ):
        for agent in agents:
            if agent.name != name:
                continue
            if agent.tool_memory is not None:
                agent.tool_memory.add_message(tool_response)
            break

    def get_embeddings(self,text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=model).data[0].embedding

        
    def add_message_to_all_agents(
        self, agents: List[AgentBase], message: Msg
    ) -> bool:
        if message.need_embedding:
            memory_embedding = self.get_embeddings(message.content)
            message.embedding = memory_embedding
        if "all" in message.receiver:
            # If receiver is all, then add the message to all agents
            for agent in agents:
                agent.add_message_to_memory([message])
            return True
        else:
            # If receiver is not all, then add the message to the specified agents
            receiver_set = copy.deepcopy(message.receiver)
            # print('# of receiver agents:', len(receiver_set))
            for agent in agents:
                if agent.name in receiver_set:
                    agent.add_message_to_memory([message])
                    receiver_set.remove(agent.name)
            if len(receiver_set) > 0:
                missing_receiver = ", ".join(list(receiver_set))
                # raise ValueError(
                #    "Receiver {} not found. Message discarded".format(missing_receiver)
                # )
                logger.warn(
                    "Receiver {} not found. Message discarded".format(missing_receiver)
                )
            return True
        
visibility_registry = Registry(name="VisibilityRegistry")
@visibility_registry.register("twitter")
class TwitterVisibility(BaseVisibility):
    """
    Visibility function for twitter: each agent can only see his or her following list

    Args:

        following_info:
            The follower list information. If it is a string, then it should be a
            path of json file storing the following info. If it is a
            dict of list of str, then it should be the following information of each agent.
    """

    follower_info: Union[str, Dict[str, List[str]]]
    current_turn: int = 0

    def update_visible_agents(self, environment: BaseEnvironment):
        self.update_receiver(environment)


    def update_receiver(self, environment: BaseEnvironment, reset=False):
        if self.follower_info is None:
            for agent in environment.agents:
                agent.set_receiver(set({agent.name})) # can only see itself
        else:
            if isinstance(self.follower_info, str):
                groups = json.load(open(self.follower_info, 'r'))
            else:
                groups = self.follower_info
            for agent in environment.agents:
                if agent.name in groups:
                    # add the agent itself
                    fl_list = groups[agent.name]+[agent.name]
                    agent.set_receiver(set(fl_list))
                else:
                    # can only see itself
                    agent.set_receiver(set({agent.name}))

    def reset(self):
        self.current_turn = 0
