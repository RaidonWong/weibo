import json


def is_valid_json(string):
    try:
        json.loads(string)  # 尝试解析字符串为 JSON
        return True, None  # 如果成功解析，则返回 True，且没有错误
    except json.JSONDecodeError as e:
        # 返回 False 和错误信息，包括位置和错误类型
        return False, f"Error at line {e.lineno}, column {e.colno}: {e.msg}"

# 示例用法
test_string = '{"id": 1, "name": "John Doe"}'  # 示例 JSON 字符串
test_string_invalid = '{"id": 1, "name": "John Doe"'  # 不完整的 JSON 字符串


my_string='{"id": 153967884435691, "id_str": "153967884435691****", "date": "2022-06-22 18:36:57+00:00", "user": {"id": 128752, "id_str": "128752****", "username": "a***l", "rawDescription": "Dropped the terrible and got in touch with my benevolent side... Mostly. Without freedom, justice, and equality for all, there is none. Climate Crisis is now.", "created": "2013-03-22 02:26:34+00:00", "followersCount": 5535, "friendsCount": 5523, "statusesCount": 346581, "favouritesCount": 551424, "listedCount": 233, "mediaCount": 6475, "location": "not in Kansas anymore", "protected": null, "verified": false, "blue": false, "blueType": null, "descriptionLinks": [], "_type": "snscrape.modules.twitter.User"}, "lang": "en", "rawContent": "@megantastic @ebruenig Not to mention the times and ways their story has changed.Link to article for previous screenshot:https://t.co/4DKVPvU31o", "replyCount": 0, "retweetCount": 0, "likeCount": 0, "quoteCount": 0, "conversationId": 1539425644857245697, "hashtags": [], "cashtags": [], "mentionedUsers": [{"id": 22343674, "username": "megantastic", "displayname": "i got spurs that jingle jangle jingle", "_type": "snscrape.modules.twitter.UserRef"}, {"id": 1471542956, "username": "ebruenig", "displayname": "Elizabeth Bruenig", "_type": "snscrape.modules.twitter.UserRef"}], "links": [{"url": "https://www.politico.com/news/2022/05/27/texas-police-wrong-decision-wait-breaching-uvalde-classroom-00035760", "text": "politico.com/news/2022/05/2\u2026", "tcourl": "https://t.co/4DKVPvU31o"}], "viewCount": null, "retweetedTweet": null, "quotedTweet": null, "place": null, "coordinates": null, "inReplyToTweetId": 1539678513178898432, "inReplyToUser": {"id": 1287526376, "username": "a_standal", "displayname": "Amie the Great", "_type": "snscrape.modules.twitter.UserRef"}, "source": "<a href=http://twitter.com/download/iphone rel=nofollow>Twitter for iPhone</a>", "sourceUrl": "http://twitter.com/download/iphone", "sourceLabel": "Twitter for iPhone", "media": {"photos": [], "videos": [], "animated": []}, "_type": "snscrape.modules.twitter.Tweet"}'
valid, error = is_valid_json(my_string)
print(f"Is valid JSON? {valid}")  # 输出: True
print(f"Error: {error}")  # 输出: None



mycolumn_23 = my_string[1800:1813]  # 获取第23个字符
print(f"Column 23 character: '{mycolumn_23}'")