import torch
from GeneralGame.common import get_distribution_from_logits

class ActCritNetwork(torch.nn.Module):
    def __init__(self, input_size, num_actions, eval_only=True):
        super(ActCritNetwork, self).__init__()
        self.in_layer = torch.nn.Linear(input_size, 10)
        self.l_2 = torch.nn.Linear(10, 5)

        self.pl = torch.nn.Linear(5, num_actions)
        self.out = torch.nn.Linear(5, 1)

        # Disable gradient calculation
        if eval_only:
            self.eval()

    def forward(self, x):
        input_data = torch.from_numpy(x).float().detach()
        x = self.in_layer(input_data)
        x = self.l_2(x)
        return self.out(x), self.pl(x)

    def sample_action(self, x, action_mask=None):
        """action_mask: boolean tensor of the same shape as x """

        _, policy_logits = self(x)
        distr = get_distribution_from_logits(policy_logits=policy_logits,
                                             action_mask=action_mask)

        return distr.sample()

    def set_weights(self, w):
        pass

    def get_weights(self):
        pass



class PPO(object):
    def __init__(self,
                 network,
                 input_size,
                 num_actions,
                 eval_only=False,
                 update_epochs=5,
                 mb_size=128,
                 clip = 0.2,
                 w_policy=1,
                 w_vf=1,
                 w_entropy=1):

        self.w_policy = w_policy
        self.w_entropy = w_entropy
        self.w_vf = w_vf
        self.clip = clip

    def update(traj):
        """Input: trajectories / outputs new weights + update_done flag """

        # Convert input data to tensors

        # Construct the dataset

        # For K epochs iterate over each mini batch

        # Calculate loss and propagate gradient

        pass

    def __ppo_loss(self, s, a, g, old_pol_log_pi, gaes, action_masks=None):
        # The input will be tensors

        # We get logits estimate
        p_hat, v_hat = self.network(s)
        distr = get_distribution_from_logits(p_hat, action_masks)

        log_p_hat = torch.nn.functional.log_softmax(p_hat)

        # Value loss
        value_loss = self.w_vf * torch.mean((v_hat - g) ** 2)

        # Entropy loss
        entropy_loss = self.w_entropy * (torch.mean(distr.entropy()))

        # Policy Loss
        action_log_pi = p_hat.gather(1, a)
        ratio = action_log_pi - old_pol_log_pi
        surr_1 = ratio * gaes
        surr_2 = torch.clamp(ratio, min=1.0 - self.clip, max=1 + self.clip) * gaes

        policy_loss = - self.w_policy * torch.min(surr_1, surr_2).mean()

        return policy_loss + value_loss + entropy_loss

