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
            
        },
        required_keys=["thought", "remark"],
        keys_to_memory="remark",
        keys_to_content="remark",
    )
