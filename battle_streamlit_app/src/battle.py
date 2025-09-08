import random
from typing import Optional, Tuple
from src.player import Player


class Battle:
    def __init__(self, p1: Player, p2: Player, llm=None, seed: Optional[int] = None):
        self.p1 = p1
        self.p2 = p2
        self.turn = 1
        self.llm = llm
        self.remained_skill_count = {"p1": 1, "p2": 1}
        self.cumulative_damage = {"p1": 0, "p2": 0}
        if seed is not None:
            random.seed(seed)

    def calc_damage(self, attacker: Player, defender: Player) -> int:
        base = max(1, int(attacker.attack - defender.defense / 3))
        luck = random.randint(0, 5)
        return max(1, base + luck)

    def check_evasion(self, defender: Player) -> Tuple[bool, str]:
        raw_ev = getattr(defender, 'evade', 0)
        try:
            raw_ev = float(raw_ev)
        except Exception:
            raw_ev = 0.0
        chance = max(0.0, min(0.95, raw_ev / 100.0))
        if random.random() < chance:
            percent = int(round(chance * 100))
            return True, f"{defender.name}이(가) 회피율 {percent}%로 회피!"
        return False, ""

    def is_over(self) -> bool:
        return self.p1.hp <= 0 or self.p2.hp <= 0

    def winner(self) -> Optional[Player]:
        if self.p1.hp <= 0 and self.p2.hp <= 0:
            return None
        if self.p1.hp <= 0:
            return self.p2
        if self.p2.hp <= 0:
            return self.p1
        return None

    def simulate(self):
        while not self.is_over():
            for attacker, defender, key in [
                (self.p1, self.p2, "p1"),
                (self.p2, self.p1, "p2"),
            ]:
                if self.is_over():
                    break
                damage = self.calc_damage(attacker, defender)
                evaded, evasion_log = self.check_evasion(defender)
                if evaded:
                    print(evasion_log)
                else:
                    defender.hp = max(0, defender.hp - damage)
                    # 누적 피해는 플레이어 식별 키로 집계
                    self.cumulative_damage[key] += damage

        print("--- 전투 종료 ---")
        w = self.winner()
        if w is None:
            print("무승부!")
        else:
            print(f"승자: {w.name}")