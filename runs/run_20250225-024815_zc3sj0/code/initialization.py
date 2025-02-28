from __future__ import annotations

import os
from typing import Dict, List, TYPE_CHECKING
from twitter_utils import Registry
import yaml
from mylogging import logger
from twitter_env import BaseEnvironment
from myllm import llm_registry
from reflection import memory_manipulator_registry
from twittermemory import memory_registry
from twitter_utils import page_registry,info_registry
from twitter_agent import agent_registry
from twitter_env import env_registry
from outputparser import output_parser_registry

import mesa

if TYPE_CHECKING:
    from agentscope.agents import AgentBase



def load_llm(llm_config: Dict):
    llm_type = llm_config.pop("llm_type", "gpt-3.5-turbo")
    return llm_registry.build(llm_type, **llm_config)


def load_memory(memory_config: Dict, llm=None):
    memory_type = memory_config.pop("memory_type", "twitter")
    if llm is not None:
        memory_config['llm'] = llm
    return memory_registry.build(memory_type, **memory_config)

def load_personal_history(memory_config: Dict):
    memory_type = memory_config.pop("memory_type", "personal_history")
    return memory_registry.build(memory_type, **memory_config)    

def load_memory_manipulator(memory_manipulator_config: Dict):
    memory_manipulator_type = memory_manipulator_config.pop(
        "memory_manipulator_type", "basic"
    )
    return memory_manipulator_registry.build(
        memory_manipulator_type, **memory_manipulator_config
    )

def load_page(page_config: Dict):
    if page_config is None: return None
    page_type = page_config.pop("page_type", "timeline")
    return page_registry.build(page_type, **page_config)

def load_info_box(info_box_config: Dict):
    if info_box_config is None: return None
    info_box_type = info_box_config.pop("info_box_type", "basic")
    return info_registry.build(info_box_type, **info_box_config)




def load_environment(env_config: Dict) -> BaseEnvironment:
    env_type = env_config.pop("env_type", "basic")
    return env_registry.build(env_type, **env_config)


def load_agent(agent_config: Dict) -> AgentBase:
    agent_type = agent_config.pop("agent_type", "twitter")
    agent = agent_registry.build(agent_type, **agent_config)
    return agent




def prepare_task_config(task, tasks_dir):
    """Read the yaml config of the given task in `tasks` directory."""
    all_task_dir = tasks_dir
    task_path = os.path.join(all_task_dir, task)
    config_path = os.path.join(task_path, "config.yaml")
    if not os.path.exists(task_path):
        all_tasks = []
        for task in os.listdir(all_task_dir):
            if (
                os.path.isdir(os.path.join(all_task_dir, task))
                and task != "__pycache__"
            ):
                all_tasks.append(task)
                for subtask in os.listdir(os.path.join(all_task_dir, task)):
                    if (
                        os.path.isdir(os.path.join(all_task_dir, task, subtask))
                        and subtask != "__pycache__"
                    ):
                        all_tasks.append(f"{task}/{subtask}")
        raise ValueError(f"Task {task} not found. Available tasks: {all_tasks}")
    if not os.path.exists(config_path):
        raise ValueError(
            "You should include the config.yaml file in the task directory"
        )
    task_config = yaml.safe_load(open(config_path))

    for i, agent_configs in enumerate(task_config["agents"]):
        llm = load_llm(agent_configs.get("llm", "gpt-3.5-turbo"))
        agent_configs["llm"] = llm
        agent_configs["memory"] = load_memory(agent_configs.get("memory", {}), llm)
        if agent_configs.get("personal_history"):
            agent_configs['personal_history'] = load_personal_history(agent_configs.get("personal_history", {}))
        if agent_configs.get("tool_memory", None) is not None:
            agent_configs["tool_memory"] = load_memory(agent_configs["tool_memory"])

        page = load_page(agent_configs.get("page", None))
        agent_configs["page"] = page
        info_box = load_info_box(agent_configs.get("info_box", None))
        agent_configs['info_box'] = info_box

        memory_manipulator = load_memory_manipulator(
            agent_configs.get("memory_manipulator", {})
        )
        agent_configs["memory_manipulator"] = memory_manipulator

        # Build the output parser
        output_parser_config = agent_configs.get("output_parser", {"type": "twitter"})
        if output_parser_config.get("type", None) == "role_assigner":
            output_parser_config["cnt_critic_agents"] = task_config.get(
                "cnt_critic_agents", 0
            )
        output_parser_name = output_parser_config.pop("type", task)
        agent_configs["output_parser"] = output_parser_registry.build(
            output_parser_name, **output_parser_config
        )

    return task_config
