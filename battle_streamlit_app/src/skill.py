def skill(attacker, defender, llm):
    format_instructions = """
    반드시 아래 JSON 하나만 출력하세요:
    {
      "attacker_delta": {"hp": int, "attack": int, "defense": int, "evade": int},
      "defender_delta": {"hp": int, "attack": int, "defense": int, "evade": int},
      "log": [string, ...]
    }
    규칙:
    - delta는 변화량(가감)입니다. 예: hp:-20은 피해 20, hp:+15는 회복 15.
    - 숫자는 정수만. 없는 항목은 0으로.
    - 반사는 defender 스킬에 '반사'/'되돌리'가 있을 때 10~50% 내에서 합리적으로.
    - 근거 없는 과도한 버프/디버프는 피하세요.
    """

    prompt = f"""
    당신은 턴제 RPG의 특수스킬 해석기입니다.
    두 캐릭터의 현재 능력치와 스킬 설명을 보고, 이번 턴에만 적용될 변화량(delta)을 산출해 주세요.

    attacker(공격자) 현재 상태:
    {{
      "name": "{attacker.name}",
      "hp": {attacker.hp}, "attack": {attacker.attack}, "defense": {attacker.defense}, "evade": {attacker.evade},
      "skill": "{attacker.skill}"
    }}

    defender(방어자) 현재 상태:
    {{
      "name": "{defender.name}",
      "hp": {defender.hp}, "attack": {defender.attack}, "defense": {defender.defense}, "evade": {defender.evade},
      "skill": "{defender.skill}"
    }}

    해석 지침:
    - 스킬 문구에 근거해서 회복/버프/디버프/반사/회피 등을 판단하세요.
    - 수치가 명시되지 않으면 상식적인 범위로 보수적으로 가정하세요.
    - 출력은 delta만 반환하고, 최종 수치는 코드가 적용합니다.
    - attacker의 스킬 적용 후 defender의 스킬이 발동할 수 있습니다. 양쪽 스킬을 모두 고려하세요.
    - 만약 조건을 충족해야만 발동하는 스킬이라면, 현재 상태로 발동하지 않는다면 모두 0으로 출력하세요.

    예시:
    input:
    attacker(공격자) 현재 상태:
            "name" : "쿵푸판다",
            "hp": 90,
            "공격력": 50,
            "방어력": 90,
            "회피율": 70,
            "스킬": "본인의 몸을 단단하게 만들어 자신의 방어력을 50 증가시킨 후, 상대에게 해당 방어력의 10%만큼 피해를 입힌다."

    defender(방어자) 현재 상태:
            "name": "버서커",
            "hp": 20,
            "공격력": 180,
            "방어력": 10,
            "회피율": 90,
            "스킬": "피가 5 이상으로 떨어지면 회피율이 50% 증가한다."

    output:
    attacker_delta: {"hp": 0, "attack": 0, "defense": 50, "evade": 0}
    defender_delta: {"hp": -5, "attack": 0, "defense": 0, "evade": 50}

    {format_instructions}
    """.strip()

    ai = llm.invoke(prompt)
    try:
        data = json.loads(ai.content)
    except Exception:
        data = {
            "attacker_delta": {"hp": 0, "attack": 0, "defense": 0, "evade": 0},
            "defender_delta": {"hp": 0, "attack": 0, "defense": 0, "evade": 0},
            "log": ["스킬 해석 실패: 기본값 적용"]
        }

    def get_delta(side, key):
        try:
            return int(data.get(f"{side}_delta", {}).get(key, 0))
        except Exception:
            return 0

    a_hp = get_delta("attacker", "hp")
    d_hp = get_delta("defender", "hp")
    a_atk = get_delta("attacker", "attack")
    a_def = get_delta("attacker", "defense")
    a_eva = get_delta("attacker", "evade")
    d_atk = get_delta("defender", "attack")
    d_def = get_delta("defender", "defense")
    d_eva = get_delta("defender", "evade")

    attacker.hp = max(0, attacker.hp + a_hp)
    defender.hp = max(0, defender.hp + d_hp)

    attacker.attack = max(0, attacker.attack + a_atk)
    attacker.defense = max(0, attacker.defense + a_def)
    attacker.evade = int(max(0, min(100, attacker.evade + a_eva)))

    defender.attack = max(0, defender.attack + d_atk)
    defender.defense = max(0, defender.defense + d_def)
    defender.evade = int(max(0, min(100, defender.evade + d_eva)))

    logs = data.get("log", [])
    if not isinstance(logs, list):
        logs = [str(logs)]
    logs.append(
        f"적용 결과 → {attacker.name} HP:{attacker.hp}, ATK:{attacker.attack}, DEF:{attacker.defense}, EVA:{attacker.evade} | "
        f"{defender.name} HP:{defender.hp}, ATK:{defender.attack}, DEF:{defender.defense}, EVA:{defender.evade}"
    )
    return "\n".join(logs)