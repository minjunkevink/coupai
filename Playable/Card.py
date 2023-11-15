class Card:
    def __init__(self, name):
        self.name = name

    def reveal(self):
        # Return the name or other identifying information about the card
        return self.name

class Duke(Card):
    def __init__(self):
        super().__init__("Duke")

    # Duke specific actions and methods
    def tax(self):
        # Implement tax action specific to Duke
        pass

class Assassin(Card):
    def __init__(self):
        super().__init__("Assassin")

    # Assassin specific actions and methods
    def assassinate(self, target):
        # Implement assassinate action specific to Assassin
        pass

# Similarly for Captain, Ambassador, Contessa