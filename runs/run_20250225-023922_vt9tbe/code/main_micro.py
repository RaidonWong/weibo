import os
import logging
from argparse import ArgumentParser

from mylogging import logger
from micro_test import Simulation_Align
import agentscope
parser = ArgumentParser()
parser.add_argument("--task", type=str, default="test")
parser.add_argument(
    "--tasks_dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "..", "twitter","tasks"),
)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--ckpt_dir", type=str, default="/root/wangliang/AgentSpace/AgentSpace-main/twitter/ckpt")
args = parser.parse_args()

logger.set_level(logging.DEBUG if args.debug else logging.INFO)


def cli_main():
    a=agentscope.init(
        model_configs="/root/wangliang/AgentSpace/AgentSpace-main/twitter/configs/model_configs.json",
        project="MyAgentSense",
    )
    myagentscope = Simulation_Align.from_task(args.task, args.tasks_dir, args.ckpt_dir)
    
    myagentscope.run()


if __name__ == "__main__":
    cli_main()
