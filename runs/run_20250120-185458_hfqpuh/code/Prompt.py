from agentscope.parsers.json_object_parser import MarkdownJsonDictParser
class Prompts:
    """Prompts for werewolf game"""

    to_all = (
        "The content of Tweet Page are as follows:{}, "
        "Please do your action according to your own willings."
    )

    action_parser = MarkdownJsonDictParser(
        content_hint={
            "thought": "what you thought",
            "action": "what actions you take in this time.You can choose to ramark and like/retweet at the same time. There are several kinds of actions you can choose from. 1: remark:post a remark on others'remark or original post. 2:like:like the post. 3.retweet. 4.post:post a new tweet. 5.stay silence",
            "content":"if you choose to remark or retweet, this is the content of remark or retweet",
        },
        required_keys=["thought", "action"],
        keys_to_memory=["action","content"],
        keys_to_metadata="action",
    )
