
from typing import Dict
import json
import httpx
import jmespath

client = httpx.Client(
    headers={
        "x-ig-app-id": "936619743392459",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "*/*",
    }
)

def scrape_user(username: str):
    """Scrape Instagram user's data"""
    result = client.get(
        f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}",
    )
    data = json.loads(result.content)
    return data["data"]["user"]


def parse_user(data: Dict) -> Dict:
    
    results = jmespath.search(
        """{
        "edge_followed_by": edge_followed_by.count,
        "edge_follow": edge_follow.count, 
        "is_private": is_private,
        "is_business_account": is_business_account
        "is_joined_recently": is_joined_recently
        "has_external_url": has_external_url
    }""",
        data,
    )
    full_name_length = len(data["full_name"])
    username_length = len(data["username"])
    full_name_has_number = any(char.isdigit() for char in data["full_name"])
    username_has_number = any(char.isdigit() for char in data["username"])
    

    ordered_results = {
        "edge_followed_by": results["edge_followed_by"],
        "edge_follow": results["edge_follow"],
        "username_length": username_length,
        "username_has_number": username_has_number,
        "full_name_has_number": full_name_has_number,
        "full_name_length": full_name_length,
        "is_private": results["is_private"],
        "is_joined_recently": results["is_joined_recently"],
        "is_business_account": results["is_business_account"],
        "has_external_url": results["has_external_url"],
        
    }

    for result in ordered_results:
        if ordered_results[result] == False or ordered_results[result] == None:
            ordered_results[result] = 0
        elif ordered_results[result] == True:
            ordered_results[result] = 1

    return ordered_results

username = input("Enter an instagram username: ")
userInfo = parse_user(scrape_user(username))
