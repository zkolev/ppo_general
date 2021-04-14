


def get_valid_actions(s, n_actions=45):
    """
    Obtain the indices of valid actions based on the state of the environment.
    NB: This function is valid only for GeneralGame with fu

    # state vector encodes the following information:
    # s[0] - remaining rolls for this turn
    # s[1:7] - encode dice rolls 1: n(1s), 2: n(2s) .. etc
    # s[7:] - positions status - booleans

    """
    s_len = len(s)
    n_positions = len(s[7:])

    actions = [i for i in range(n_positions) if s[(i + 7)] != 1]

    # If there are remaining rolls all dice
    # rolls combinations are valid actions

    if s[0] > 0:
        actions += list(range(n_positions, n_actions))

    return actions
