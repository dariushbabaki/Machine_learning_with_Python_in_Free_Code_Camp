import random

def player(prev_play, opponent_history=[]):
    # If it's the first move (no previous play exists)
    if prev_play == "":
        return "R"
    
    # Update the opponent's move history
    opponent_history.append(prev_play)
    
    # With a 10% probability, choose a random move to introduce variation
    if random.random() < 0.1:
        return random.choice(["R", "P", "S"])
    
    # Analyze the frequency of opponent moves
    total = len(opponent_history)
    r_count = opponent_history.count("R")
    p_count = opponent_history.count("P")
    s_count = opponent_history.count("S")
    
    # If any move constitutes more than 50% of the opponent's moves, choose the counter move
    if r_count > total * 0.5:
        return "P"  # Paper beats Rock
    elif p_count > total * 0.5:
        return "S"  # Scissors beat Paper
    elif s_count > total * 0.5:
        return "R"  # Rock beats Scissors
    
    # If the opponent repeated the same move in the last two plays,
    # assume they are likely to repeat it, so choose the counter move
    if total >= 2 and opponent_history[-1] == opponent_history[-2]:
        if opponent_history[-1] == "R":
            return "P"
        elif opponent_history[-1] == "P":
            return "S"
        elif opponent_history[-1] == "S":
            return "R"
    
    # Default strategy: counter the opponent's last move
    last = opponent_history[-1]
    if last == "R":
        return "P"
    elif last == "P":
        return "S"
    elif last == "S":
        return "R"
    
    # In the unforeseen case, return a random move
    return random.choice(["R", "P", "S"])
