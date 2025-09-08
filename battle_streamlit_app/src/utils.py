def parse_json_response(response):
    import json
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return None

def format_player_info(player):
    return f"Name: {player.name}, HP: {player.hp}, Attack: {player.attack}, Defense: {player.defense}, Evade: {player.evade}, Skill: {player.skill}"

def log_battle_event(event):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp} - {event}"