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

class TwitterPage(MemoryBase):
    """
    messages: list of TwitterMessage
    tweet_num: illustrate most recent tweet_num tweets
    """
    messages: List[Msg] = Field(default=[])
    tweet_num: int=5
    
    def add_message(self, messages: List[Msg]) -> None:
        # only reserve the post/retweet action
        # pay attention to the storage size
        for message in messages:
            self.messages.insert(0, message)
        self.messages = self.messages[:self.tweet_num]

    def to_string(self) -> str:
        return "\n".join(
            [
                f'tweet id: {message.id} [{message.name}]: {message.content} --Post Time: {message.timestamp}'
                for message in self.messages
            ]
        )

    def reset(self) -> None:
        self.messages = []


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

class InfoBox(MemoryBase):
    """
    messages: list of TwitterMessage
    cmt_num: illustrate most recent tweet_num comments
    tweet_num: illustrate most recent tweet_num tweets
    """
    messages: Dict[str, Sequence[Msg]] = Field(default={})
    cmt_num: int=10
    tweet_num: int=3
    
    def add_message(self, messages: Union[Sequence[Msg],Msg]) -> None:
        # store comments into different groups (group by parent id)
        for message in messages:
            parent_id = message.parent_id
            if parent_id not in self.messages:
                self.messages[parent_id] = []
            self.messages[parent_id].insert(0, message)
            self.messages[parent_id] = self.messages[parent_id][:self.tweet_num]

    def to_string(self) -> str:
        return "\n".join(["original tweet id:"+parent_id+":\n"+"\n".join(
            [
                f'[{message.name}]: {message.content}'
                for message in self.messages[parent_id]
            ])
            for parent_id in self.messages
            ]
        )

    def reset(self) -> None:
        self.messages = {}

def extract_agent_configs(scene_id):
    agent_lib_dir = r"./configs/agent_lib.json"
    scenario_config_dir = r"./configs/scenario_configs.json"
    agent_config_dir = f"./configs/agent_configs_scene{scene_id}.json"
    with open(agent_lib_dir, 'r') as f:
        agents = json.load(f)
    with open(scenario_config_dir, 'r') as f:
        scenarios = json.load(f)
    scenario = [scene for scene in scenarios if scene["scenario_id"] == scene_id][0]

    # naive sampling strategy
    agent_info = scenario["agent_info"]
    agent_candidate = []
    for sub_agent_info in agent_info:
        for idx, agent in enumerate(agents):
            if agent['role'] == sub_agent_info["agent_role"]:
                agent_candidate.append(agent)
                del agents[idx]
                break
    if len(agent_candidate) != len(agent_info):
        raise Exception(f"[Customized Error] agent info in Scene {scene_id} can not be satisfied by current agent libs!")
    
    # replace scenario info with agent info
    characters = [char for char in agent_candidate if char['role'] == 'character']
    judges = [char for char in agent_candidate if char['role'] == 'judge']
    for i in range(len(characters)):
        scenario["description"] = Template(scenario["description"]).safe_substitute({f"character_{i+1}": characters[i]["attribute"]["name"]})
        scenario["agent_info"][i]["agent_name"] = characters[i]["attribute"]["name"]
        # optional
        # scenario["agent_info"][i]["agent_name"] = Template(scenario["agent_info"][i]["agent_name"]).safe_substitute({f"character_{i+1}": characters[i]["attribute"]["name"]})
        scenario["agent_info"][i]["profile"] = Template(scenario["agent_info"][i]["profile"]).safe_substitute({f"profile_{i+1}": characters[i]["attribute"]["profile"]})
        scenario["agent_info"][i]["model_config_name"] = Template(scenario["agent_info"][i]["model_config_name"]).safe_substitute({"model": characters[i]["model_config_name"]})
    for i in range(len(judges)):
        scenario["agent_info"][len(characters)+i]['agent_name'] = Template(scenario["agent_info"][len(characters)+i]["agent_name"]).safe_substitute({f"judge_{i+1}": judges[i]["attribute"]["name"]})
        scenario["agent_info"][len(characters)+i]['model_config_name'] = Template(scenario["agent_info"][len(characters)+i]["model_config_name"]).safe_substitute({"model": judges[i]["model_config_name"]})
    
    # replace prompt info with scenario info
    for i in range(len(characters)):
        input_args = {
            "agent_name": scenario["agent_info"][i]["agent_name"],
            "profile": scenario["agent_info"][i]["profile"],
            "background_info": scenario["background_info"],
            "description": scenario["description"],
            "social_goal": " ".join(scenario["agent_info"][i]["social_goal"]),
            "private_info": scenario["agent_info"][i]["private_info"] if scenario["agent_info"][i]["private_info"] else "N/A"
        }
        scenario["agent_info"][i]["prompt"] = Template(scenario["agent_info"][i]["prompt"]).safe_substitute(input_args)
    
    agent_configs = []
    for agent in scenario["agent_info"]:
        agent_parsered = {"class": "DialogAgent", "args": {"name": agent["agent_name"], "sys_prompt": agent["prompt"], "model_config_name": agent["model_config_name"], "use_memory": True}}
        agent_configs.append(agent_parsered)
    
    with open(agent_config_dir, 'a') as f:
        f.write(json.dumps(agent_configs, ensure_ascii=False, indent=4))

def manage_sequential(scene_id):  #这里的scene_id要+1
    with open('./configs/scenario_configs.json', 'r') as f:
        scenarios = json.load(f)
    scenario = scenarios[scene_id-1]
    sequential_info = scenario["sequential"]
    if len(sequential_info) == 1 and 'rand' in sequential_info[0]:
        max_round = int(sequential_info[0].split("@")[-1])
        characters = [char for char in scenario["agent_info"] if char['agent_role'] == 'character']
        sequential = []
        last_speaker = -1
        for round in range(max_round):
            while True:
                speaker = random.choice(range(len(characters)))
                if speaker != last_speaker:
                    sequential.append(speaker)
                    last_speaker = speaker
                    break
    else:
        characters = [char for char in scenario["agent_info"] if char['agent_role'] == 'character']
        sequential = [characters.index(i) for i in sequential_info]
    # print(sequential)
    return sequential


def parser_result(dir, output_dir):
    logs = []
    with open(dir, 'r') as f:
        for line in f:
            logs.append(json.loads(line[:-1]))
    dialog_results = []
    for log in logs:
        dialog_results.append({"name": log["name"], "content": log["content"]})
    with open(output_dir, 'a') as f:
        f.write(json.dumps(dialog_results, ensure_ascii=False, indent=4))

def set_parsers(
    agents: Union[AgentBase, list[AgentBase]],
    parser_name: str,
) -> None:
    """Add parser to agents"""
    if not isinstance(agents, list):
        agents = [agents]
    for agent in agents:
        agent.set_parser(parser_name)


from agentscope.agents import Operator
async def handle_operators(operators, x):
    tasks = [asyncio.to_thread(operator.reply, x) for operator in operators]
    responses = await asyncio.gather(*tasks)
    return responses

async def mysequentialpipeline(
    operators: Sequence[Operator],
    twitter_page:dict,
    like:int,
    retweet:int,
    x: Optional[dict] = None,
) -> dict:
    """Functional version of SequentialPipeline.

    Args:
        operators (`Sequence[Operator]`):
            Participating operators.
        x (`Optional[dict]`, defaults to `None`):
            The input dictionary.

    Returns:
        `dict`: the output dictionary.
    """
    if len(operators) == 0:
        raise ValueError("No operators provided.")

    responses=await handle_operators(operators, x)
    for msg in responses:
        like, retweet = update_page_num(msg.to_dict(), like, retweet)
        twitter_page[len(twitter_page)] = twitterpage_todict(msg.to_dict(), like, retweet)
        if x!=None:
            handle_retweet(msg,operators,find_agent_by_name(x.to_dict().get('name'),operators),x)
        
    return msg

def find_agent_by_name(agent_name,agents):
    for agent in agents:
        if agent.name == agent_name:
            return agent
    return None

def handle_retweet(msg,agents,operator,x):
    msg=msg.to_dict()
    metadata = msg.get('metadata', '').split(',')
    original_agent=find_agent_by_name(x.get('name'),agents)
    ori_msg_content=x.get('content', {}).get('content', '')
    msg_content=msg.get('content', {}).get('content', '')
    
    retweet_msg = Msg(
    name=operator.name,
    content=f"{operator.name} retweets a tweet of [{original_agent.name}]: '{ori_msg_content}' with additional statements: {msg_content}.",
    role='assistant'
)

    if 'retweet' in metadata:
        for audience in operator._audience:
            audience.observe(retweet_msg)
            operator.speak(retweet_msg)
            print("finish retweet")
        

import concurrent.futures
from typing import Sequence, Optional


import asyncio

async def myparallelpipeline(
    operators: Sequence[Operator],
    twitter_page: dict,
    like: int,
    retweet: int,
    x: Optional[dict] = None,
) -> dict:
    """Asynchronous version of SequentialPipeline with all operators executed concurrently, processed by completion order."""

    if len(operators) == 0:
        raise ValueError("No operators provided.")
    
    # 创建并行任务
    tasks = [operator(x) for operator in operators]  # 所有 operator 同时异步执行

    # 使用 as_completed 按照完成顺序处理任务
    results = {}
    for operator, result in zip(operators, asyncio.as_completed(tasks)):
        msg = await result  # 获取每个操作的结果
        like, retweet = update_page_num(msg.to_dict(), like, retweet)
        twitter_page[len(twitter_page)] = twitterpage_todict(msg.to_dict(), like, retweet)
        results[operator.name] = msg.to_dict()  # 使用 operator 的 name 作为键

    return results



def twitterpage_todict(msg,like,retweet):
    # 创建一个新的字典来存储提取的内容
    result = {}

    # 获取所需的字段
    name = msg.get('name')  # 获取'name'
    timestamp = msg.get('timestamp')  # 获取'timestamp'
    metadata = msg.get('metadata')  # 获取'metadata'

    # 获取content字段，并确保进一步提取'content'中的内容
    content = msg.get('content', {}).get('content', '')  # 进一步提取content字段中的内容

    # 将提取的数据存入新的字典
    result = {
        'name': name,
        'timestamp': timestamp,
        'metadata': metadata,
        'content': content,
        'like':like,
        'retweet':retweet
    }

    return result


def update_page_num(msg, like, retweet):
    # 获取metadata字段并将其分割成列表
    
    metadata = msg.get('metadata', '')
    if isinstance(metadata, str):
        metadata = metadata.split(',')
    else:
        
        metadata = []
        
    # 后续处理...


    # 如果metadata包含'like'，则增加like计数
    if 'like' in metadata:
        like += 1
    # 如果metadata包含'retweet'，则增加retweet计数
    if 'retweet' in metadata:
        retweet += 1
    
    return like, retweet

#def retweet(msg,agent):
    metadata = msg.get('metadata', '').split(',')
    if 'retweet' in metadata:
        with msghub(agent.follower,announcement=msg.get('content', {}).get('content', '')) as hub:
            pass

    



if __name__ == '__main__':
    #extract_agent_configs(0)
    #seq = manage_sequential(1)
    #parser_result("./runs/run_20241112-100514_3438x6/logging.chat", "./results/result_scene1.json")
    #raw_to_config('./raw_data/raw_config/')
    pass












