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
            "remark": "what you remark",
            "finish_tweet": "whether decision on continue to "
            "tweet or not (true/false)",
        },
        required_keys=["thought", "remark", "finish_discussion"],
        keys_to_memory="remark",
        keys_to_content="remark",
        keys_to_metadata=["finish_tweet"],
    )
