class Player:
    def __init__(self, name, hp, attack, defense, skill, evade):
        self.name = name
        self.hp = hp
        self.attack = attack
        self.defense = defense
        self.skill = skill
        self.evade = evade

    def __str__(self):
        return f"캐릭터 이름: {self.name}\n체력: {self.hp}\n공격력: {self.attack}\n방어력: {self.defense}\n회피율: {self.evade}\n특수공격 스킬: {self.skill}"