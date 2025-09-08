import os
import json
import random
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Tuple, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from src.player import Player


# 환경변수 로드 (.env 지원)
load_dotenv()


def create_llm(model: str, temperature: float = 0.7) -> ChatOpenAI:
    # langchain_openai ChatOpenAI는 model 파라미터 사용
    return ChatOpenAI(model=model, temperature=temperature)


def build_character_prompt():
    # 노트북의 프롬프트와 예시를 재구성
    examples = [
        {
            "input": "이 캐릭터는 판다야, 겉모습은 매우 귀엽지만 쿵푸를 매우 잘하고 어마무시한 방어력을 가지고 있지",
            "name": "쿵푸판다",
            "hp": 250,
            "공격력": 20,
            "방어력": 20,
            "회피율": 10,
            "스킬": "본인의 몸을 단단하게 만들어 자신의 방어력을 50 증가시킨 후, 상대에게 해당 방어력의 100%만큼 피해를 입힌다.",
        },
        {
            "input": "이 캐릭터는 광전사야, 광전사는 매우 강력한 공격력을 가지고 있어",
            "name": "버서커",
            "hp": 90,
            "공격력": 40,
            "방어력": 10,
            "회피율": 30,
            "스킬": "피가 60 이상으로 떨어지면 회피율이 50% 증가한다.",
        },
        {
            "input": "이 캐릭터는 괴수야, 괴수는 매우 강한 체력과 방어력을 가지고 있어",
            "name": "타이탄",
            "hp": 130,
            "공격력": 10,
            "방어력": 30,
            "회피율": 10,
            "스킬": "자신의 HP를 현재 HP의 30%만큼 회복하고, 상대에게 회복한 피해의 100%만큼 피해를 입힌다.",
        },
    ]

    json_parser = JsonOutputParser()
    format_instructions = json_parser.get_format_instructions()

    example_prompt = PromptTemplate(
        template=(
            "input : {input}\n"
            "name : {name}, hp : {hp}, 공격력 : {공격력}, 방어력 : {방어력}, 스킬 : {스킬}, 회피율 : {회피율} \n"
        ),
        input_variables=["input", "name", "hp", "공격력", "방어력", "스킬", "회피율"],
    )

    prefix = (
        "당신은 캐릭터 생성기입니다. 사용자가 입력한 프롬프트에 맞는 캐릭터를 생성하고, 캐릭터에 맞는 능력치를 부여하세요. "
        "능력치는 각각 hp, 공격력, 방어력,회피율과 특수공격을 위한 스킬이 주어집니다. "
        "특수공격을 제외한 능력치는 다음과 같이 부여하세요. HP는 기본적으로 100~300이어야 하며 공격력은 20~40, 방어력은 10~30, 회피율은 5~18 사이로 부여하세요. "
        "단, 특수공격의 성능이 좋다면 , 기본 능력치의 범위를 벗어날 만큼 약해도 괜찮습니다. "
        "특수공격 스킬은 캐릭터에 맞게 1개 부여하세요. 특수공격은 반드시 능력치에 관한 것이어야 합니다. 또한 스킬은 반드시 전투에 영향을 미치는 것이어야 합니다.\n"
        f"형식지정 : {{format_instructions}}\n"
    )

    suffix = "이제 새로운 캐릭터를 생성해봅시다. ###input : {input}###에 맞는 캐릭터를 생성하세요."

    fewshot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input"],
        partial_variables={"format_instructions": format_instructions},
    )

    return fewshot_prompt, json_parser


def generate_character(user_prompt: str, llm: ChatOpenAI) -> dict | None:
    fewshot_prompt, json_parser = build_character_prompt()
    prompt = fewshot_prompt.format(input=user_prompt)
    ai_message = llm.invoke(prompt)
    try:
        result = json_parser.parse(ai_message.content)
        return result
    except Exception as e:
        st.error(f"캐릭터 생성 파싱 실패: {e}")
        st.code(ai_message.content)
        return None


def skill(attacker: Player, defender: Player, llm: ChatOpenAI) -> List[str]:
    """노트북의 스킬 해석 로직을 포팅. delta JSON을 받아 적용하고 로그 반환."""
    import json as _json

    format_instructions = (
        "반드시 아래 JSON 하나만 출력하세요:\n"
        "{\n"
        "  \"attacker_delta\": {\"hp\": int, \"attack\": int, \"defense\": int, \"evade\": int},\n"
        "  \"defender_delta\": {\"hp\": int, \"attack\": int, \"defense\": int, \"evade\": int},\n"
        "  \"log\": [string, ...]\n"
        "}\n"
        "규칙:\n"
        "- delta는 변화량(가감)입니다. 예: hp:-20은 피해 20, hp:+15는 회복 15.\n"
        "- 숫자는 정수만. 없는 항목은 0으로.\n"
        "- 반사는 defender 스킬에 '반사'/'되돌리'가 있을 때 10~50% 내에서 합리적으로.\n"
        "- 근거 없는 과도한 버프/디버프는 피하세요.\n"
    )

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

{format_instructions}
""".strip()

    ai = llm.invoke(prompt)
    try:
        data = _json.loads(ai.content)
    except Exception:
        data = {
            "attacker_delta": {"hp": 0, "attack": 0, "defense": 0, "evade": 0},
            "defender_delta": {"hp": 0, "attack": 0, "defense": 0, "evade": 0},
            "log": ["스킬 해석 실패: 기본값 적용"],
        }

    def _get_delta(side: str, key: str) -> int:
        try:
            return int(data.get(f"{side}_delta", {}).get(key, 0))
        except Exception:
            return 0

    a_hp = _get_delta("attacker", "hp")
    d_hp = _get_delta("defender", "hp")
    a_atk = _get_delta("attacker", "attack")
    a_def = _get_delta("attacker", "defense")
    a_eva = _get_delta("attacker", "evade")
    d_atk = _get_delta("defender", "attack")
    d_def = _get_delta("defender", "defense")
    d_eva = _get_delta("defender", "evade")

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
    return logs


def calc_damage(attacker: Player, defender: Player) -> int:
    base = max(1, int(attacker.attack - defender.defense / 3))
    luck = random.randint(0, 5)
    return max(1, base + luck)


def check_evasion(defender: Player) -> Tuple[bool, str]:
    raw_ev = getattr(defender, "evade", 0)
    try:
        raw_ev = float(raw_ev)
    except Exception:
        raw_ev = 0.0
    chance = max(0.0, min(0.95, raw_ev / 100.0))
    if random.random() < chance:
        percent = int(round(chance * 100))
        return True, f"{defender.name}이(가) 회피율 {percent}%로 회피!"
    return False, ""


def simulate_battle(p1: Player, p2: Player, llm: Optional[ChatOpenAI] = None, seed: Optional[int] = None) -> List[str]:
    if seed is not None:
        random.seed(seed)
    logs: List[str] = []
    turn = 1
    remained_skill_count = {"p1": 1, "p2": 1}
    cumulative_damage = {"p1": 0, "p2": 0}

    while p1.hp > 0 and p2.hp > 0:
        logs.append(f"--- {turn}턴 시작 ---")
        for attacker, defender, key in [(p1, p2, "p1"), (p2, p1, "p2")]:
            if p1.hp <= 0 or p2.hp <= 0:
                break
            logs.append(f"{attacker.name}의 턴")
            can_skill = (remained_skill_count[key] >= 1) and (cumulative_damage[key] >= 70)
            if can_skill and llm is not None:
                logs.append(f"{attacker.name}이(가) 스킬을 사용합니다: {attacker.skill}")
                try:
                    s_logs = skill(attacker, defender, llm)
                    logs.extend(s_logs)
                except Exception as e:
                    logs.append(f"스킬 처리 오류: {e}")
                remained_skill_count[key] -= 1
            else:
                evaded, evasion_log = check_evasion(defender)
                if evaded:
                    logs.append(evasion_log)
                    logs.append(f"{defender.name}이(가) 공격을 회피했습니다!")
                else:
                    damage = calc_damage(attacker, defender)
                    defender.hp = max(0, defender.hp - damage)
                    cumulative_damage[key] += damage
                    logs.append(
                        f"{attacker.name}이(가) {defender.name}에게 {damage}의 피해! ({defender.name} HP: {defender.hp})"
                    )
                    logs.append(f"누적 피해 업데이트: {attacker.name} = {cumulative_damage[key]}")
        turn += 1

    logs.append("--- 전투 종료 ---")
    if p1.hp <= 0 and p2.hp <= 0:
        logs.append("무승부!")
    elif p1.hp <= 0:
        logs.append(f"승자: {p2.name}")
    else:
        logs.append(f"승자: {p1.name}")
    return logs


def render_player(player: Player, title: str):
    st.subheader(title)
    st.write(
        f"""
        이름: {player.name}
        - HP: {player.hp}
        - 공격력: {player.attack}
        - 방어력: {player.defense}
        - 회피율: {player.evade}
        - 스킬: {player.skill}
        """
    )


def main():
    st.set_page_config(page_title="Battle Game", page_icon="⚔️", layout="centered")
    st.title("⚔️ Battle Game")

    # 사이드바: OpenAI 설정
    st.sidebar.header("LLM 설정")
    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    model_name = st.sidebar.text_input("OpenAI 모델", value=default_model)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    have_key = bool(os.getenv("OPENAI_API_KEY"))
    st.sidebar.caption("API Key는 환경변수 OPENAI_API_KEY 를 사용합니다 (.env 지원)")
    if not have_key:
        st.sidebar.warning("OPENAI_API_KEY 가 설정되어 있지 않습니다. 캐릭터 생성/스킬 사용이 비활성화됩니다.")

    # 세션 상태 초기화
    if "player1" not in st.session_state:
        st.session_state.player1 = None
    if "player2" not in st.session_state:
        st.session_state.player2 = None
    if "logs" not in st.session_state:
        st.session_state.logs = ""

    st.header("캐릭터 생성")
    col1, col2 = st.columns(2)
    with col1:
        player1_prompt = st.text_input("플레이어 1 캐릭터를 입력하세요:", key="p1_prompt")
    with col2:
        player2_prompt = st.text_input("플레이어 2 캐릭터를 입력하세요:", key="p2_prompt")

    btn_cols = st.columns([1, 1, 2])
    with btn_cols[0]:
        if st.button("캐릭터 생성", disabled=not have_key):
            if player1_prompt and player2_prompt:
                llm = create_llm(model_name, temperature)
                data1 = generate_character(player1_prompt, llm)
                data2 = generate_character(player2_prompt, llm)
                if data1 and data2:
                    try:
                        st.session_state.player1 = Player(
                            data1["name"], int(data1["hp"]), int(data1["공격력"]), int(data1["방어력"]), data1["스킬"], int(data1["회피율"]) 
                        )
                        st.session_state.player2 = Player(
                            data2["name"], int(data2["hp"]), int(data2["공격력"]), int(data2["방어력"]), data2["스킬"], int(data2["회피율"]) 
                        )
                        st.success("캐릭터가 생성되었습니다.")
                    except Exception as e:
                        st.error(f"캐릭터 데이터 형식 오류: {e}")
                else:
                    st.warning("캐릭터 생성에 실패했습니다.")
            else:
                st.warning("두 플레이어의 캐릭터를 입력해야 합니다.")

        

    # 캐릭터 표시 및 전투 실행
    if st.session_state.player1 and st.session_state.player2:
        c1, c2 = st.columns(2)
        with c1:
            render_player(st.session_state.player1, "플레이어 1")
        with c2:
            render_player(st.session_state.player2, "플레이어 2")

        st.divider()
        st.subheader("전투 시작")
        seed = st.number_input("랜덤 시드(선택)", min_value=0, max_value=10_000, value=42, step=1)
        run_cols = st.columns([1, 3])
        with run_cols[0]:
            run_btn = st.button("전투 시뮬레이션 실행")
        if run_btn:
            llm = create_llm(model_name, temperature) if have_key else None
            # 원본 객체 변형 방지: 얕은 복사로 새 객체 생성
            p1 = Player(**st.session_state.player1.__dict__)
            p2 = Player(**st.session_state.player2.__dict__)
            logs = simulate_battle(p1, p2, llm=llm, seed=int(seed))
            st.session_state.logs = "\n".join(logs)

    # 로그 영역
    if st.session_state.logs:
        st.divider()
        st.subheader("전투 로그")
        st.code(st.session_state.logs)


if __name__ == "__main__":
    main()