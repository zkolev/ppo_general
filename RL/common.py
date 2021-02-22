"""
Contains utility functions

"""
from itertools import combinations
import numpy as np
import torch


def set_discrete_action_space(positions_count):
    """
    :return: List of tuples
             The list contains the complete
             action space of the environment
    """

    # Checkout position actions:
    actions = [(0, p) for p in range(positions_count)]

    # Roll dice actions
    for i in range(5):
        actions.extend([(1, list(k))
                        for k in combinations(range(5), (i + 1))])

    return actions


def stable_softmax(logits):
    scaled_logits = logits - np.max(logits, axis=1, keepdims=True)
    num = np.exp(scaled_logits)
    return num / np.sum(num)

# TODO: To rename the function
def get_distribution_from_logits(policy_logits, action_mask=None):
    """Get masked distribution, accounting for impossible actions """

    if action_mask is not None:
        mask = action_mask
    else:
        mask = torch.zeros_like(policy_logits, dtype=torch.bool)

    distr = torch.distributions.Categorical(
        logits=torch.where(mask, torch.full_like(policy_logits, -float("Inf")), policy_logits)
    )
    return distr

def eval_policy(env, policy, iters, positions_count):

    discrete_positions = set_discrete_action_space(positions_count=positions_count)
    game = env(print_state=False)

    for i in range(iters):
        s, act, r, m = game.start_game()
        scores = []
        running_score = 0

        while not m:
            #Sample action
            logits = policy(torch.tensor(s, dtype=torch.float32))
            distr = get_distribution_from_logits(logits)

            act = discrete_positions[torch.argmax(distr.probs)]
            s, act, r, m = game.step(act)
            running_score += r

        scores.append(running_score)

    return torch.tensor(r, dtype=torch.float32)