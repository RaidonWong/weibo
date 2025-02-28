from twitter_utils import manage_sequential,mysequentialpipeline,set_parsers,myparallelpipeline
from functools import partial
from agentscope.msghub import msghub
from agentscope.message import Msg
from agentscope.pipelines.functional import sequentialpipeline
import agentscope
from functools import partial
import asyncio
from Prompt import Prompts
import agentscope

# load OpenAI environment variants
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())




def main(scene_id):
    "Social Simualtion under AgentSense settings"
    TweetPage = partial(Msg, name="TweetPage", role="assistant", echo=True)
    twitter_page={}
    # Initialization
    agents = agentscope.init(
        model_configs="./configs/model_configs.json",
        agent_configs=f"./configs/agent_configs_scene{scene_id}.json",
        project="AgentSense",
    )

    MAX_TWEET_ROUND = 1
    like=0
    retweet=0
    #sequential_idx = manage_sequential(scene_id)#[0,1,0,1,0]
    sequential_idx = [0,1,2,3]
    sequential = [agents[i] for i in sequential_idx]
    for _ in range(1, MAX_TWEET_ROUND + 1):
        hint = TweetPage(content=Prompts.to_all.format(twitter_page))
        with msghub(agents,announcement=hint) as hub:
            set_parsers(agents, Prompts.action_parser)
            
            x = asyncio.run(mysequentialpipeline(sequential,twitter_page,like,retweet))
            # x = sequentialpipeline(sequential)

    
            


if __name__ == '__main__':
    main(0)
    