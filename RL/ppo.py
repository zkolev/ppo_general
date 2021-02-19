import torch
from RL.common import get_distribution_from_logits


class GeneralGameTrajectories(torch.utils.data.Dataset):
    def __init__(self, rollouts):
        super(GeneralGameTrajectories).__init__()

        traj_keys = ["states", "actions", "action_masks", "returns",
                     "advantages", "old_pi_logits" ]

        self.data = {}
        for k in traj_keys:
            self.data[k]=torch.cat([d[k] for d in rollouts])

        # Get the size of the data
        self.n_samples = self.data['states'].size()[0]


    def __getitem__(self, index):
        return {k: self.data[k][index] for k in self.data}

    def __len__(self):
        return self.n_samples


class ActCritNetwork(torch.nn.Module):
    def __init__(self, input_size, num_actions, eval_only=True):
        super(ActCritNetwork, self).__init__()
        self.in_layer = torch.nn.Linear(input_size, 20)
        self.l_1 = torch.nn.Linear(20, 10)
        self.l_2 = torch.nn.Linear(10, 5)

        self.pl = torch.nn.Linear(5, num_actions)
        self.out = torch.nn.Linear(5, 1)

        # Disable gradient calculation
        if eval_only:
            self.eval()

    def forward(self, input_data):
        # input_data = torch.from_numpy(x).float().detach()
        x = self.in_layer(input_data)
        x = self.l_1(x)
        x = self.l_2(x)
        return self.out(x), self.pl(x)

    def sample_action(self, x, action_mask=None):
        """action_mask: boolean tensor of the same shape as x """

        _, policy_logits = self(x)
        distr = get_distribution_from_logits(policy_logits=policy_logits,
                                             action_mask=action_mask)

        return distr.sample()


class PPO(object):
    def __init__(self,
                 input_size,
                 num_actions,
                 lr = 1e-3,
                 minibatch_size=128,
                 clip=0.2,
                 w_policy=1,
                 w_vf=1,
                 w_entropy=1):

        self.network = ActCritNetwork(input_size=input_size,
                                      num_actions=num_actions,
                                      eval_only=False)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)


        self.w_policy = w_policy
        self.w_entropy = w_entropy
        self.w_vf = w_vf
        self.clip = clip
        self.minibatch_size =minibatch_size


    def update(self, traj, epochs=5):
        """
        Input: trajectories / outputs new weights + update_done flag
        Expected Trajectory:
        dict of tensors with key/type/shape as follows:

        states float32 (N X D)
        actions int32 (N X 1)
        action_masks boolean (N X # Actions)
        returns float32 (N X 1)
        advantages float32 (N X 1)
        old_pi_logits float32 (N X 1)
        aux_data: dict of tensors {}

        NB: aux_data is a placeholder for meta data

        """
        # Construct the dataset

        # For K epochs iterate over each mini batch

        ds = GeneralGameTrajectories(traj)
        dl = torch.utils.data.DataLoader(ds,
                                         batch_size=self.minibatch_size,
                                         shuffle=True,
                                         num_workers=2)

        for epoch in range(epochs):
            for i, b in enumerate(dl):

                self.optimizer.zero_grad()
                loss = self.__ppo_loss(s=b['states'],
                                       a=b['actions'],
                                       g=b['returns'],
                                       old_pol_log_pi=b['old_pi_logits'],
                                       advantages=b['advantages'],
                                       action_masks=b['action_masks']
                                       )
                print(f"Epoch {epoch}, batch {i}, loss: {loss}" )
                loss.backward()
                self.optimizer.step()

        return self.network.state_dict()

    def __ppo_loss(self, s, a, g, old_pol_log_pi, advantages, action_masks=None):
        # The input will be tensors

        # We get logits estimate
        v_hat, p_hat = self.network(s)
        distr = get_distribution_from_logits(p_hat, action_masks)

        # Value loss
        value_loss = self.w_vf * torch.mean((v_hat - g) ** 2)

        # Entropy loss
        entropy_loss = self.w_entropy * (torch.mean(distr.entropy()))

        # Policy Loss
        action_log_pi = p_hat.gather(1, a)
        ratio = action_log_pi - old_pol_log_pi
        surr_1 = ratio * advantages
        surr_2 = torch.clamp(ratio, min=1.0 - self.clip, max=1 + self.clip) * advantages

        policy_loss = - self.w_policy * torch.min(surr_1, surr_2).mean()

        return policy_loss + value_loss + entropy_loss

