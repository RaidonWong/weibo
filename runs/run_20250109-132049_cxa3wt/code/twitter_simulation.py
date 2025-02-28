from twitter_utils import manage_sequential
from functools import partial
from agentscope.msghub import msghub
from agentscope.message import Msg
from agentscope.pipelines.functional import sequentialpipeline
import agentscope

# load OpenAI environment variants
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

def main(scene_id):
    "Social Simualtion under AgentSense settings"
    
    # Initialization
    agents = agentscope.init(
        model_configs="./configs/model_configs.json",
        agent_configs=f"./configs/agent_configs_scene{scene_id}.json",
        project="AgentSense",
    )
    
    sequential_idx = manage_sequential(scene_id)
    sequential = [agents[i] for i in sequential_idx]
    
    with msghub(agents) as hub:
        hub.broadcast(Msg(name=sequential[0].name, content="Hi there!", role='assistant'))
        x = sequentialpipeline(sequential[1:])
        # x = sequentialpipeline(sequential)


if __name__ == '__main__':
    main(1)
    