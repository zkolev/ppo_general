import torch
import torch.nn.functional as fn
from RL.common import get_distribution_from_logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class GeneralGameTrajectories(torch.utils.data.Dataset):
    def __init__(self, rollouts, normalize_advantages=True):
        super(GeneralGameTrajectories).__init__()

        traj_keys = ["states", "actions", "action_masks", "returns",
                     "advantages", "old_pi_logits"]

        self.data = {}
        for k in traj_keys:
            self.data[k] = torch.cat([d[k] for d in rollouts]).to(device)

        # Normalize advantages:
        if normalize_advantages:
            self.data['advantages'] = (self.data['advantages'] - self.data['advantages'].mean()) \
                                      / (self.data['advantages'].std() + 1e-5)

        # Get the size of the data
        self.n_samples = self.data['states'].size()[0]

    def __getitem__(self, index):
        return {k: self.data[k][index] for k in self.data}

    def __len__(self):
        return self.n_samples


class ActCritNetwork(torch.nn.Module):
    def __init__(self, input_size, num_actions, eval_only=True):
        super(ActCritNetwork, self).__init__()
        self.in_layer = torch.nn.Linear(input_size, 25)
        self.l_1 = torch.nn.Linear(25, 15)
        # self.l_2 = torch.nn.Linear(10, 5)

        self.pl_1 = torch.nn.Linear(15, 10)
        self.pl_out = torch.nn.Linear(10, num_actions)

        self.v_1 = torch.nn.Linear(15, 5)
        self.v_out = torch.nn.Linear(5, 1)

        # Disable gradient calculation
        if eval_only:
            self.eval()

    def forward(self, input_data):
        # input_data = torch.from_numpy(x).float().detach()
        x = fn.relu(self.in_layer(input_data))
        x = fn.relu(self.l_1(x))

        pl = fn.relu(self.pl_1(x))
        vlr = fn.relu(self.v_1(x))

        return self.v_out(vlr), self.pl_out(pl)

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
                 lr=1e-3,
                 minibatch_size=128,
                 clip=0.2,
                 w_policy=1,
                 w_vf=0.5,
                 w_entropy=1,
                 epoch_writer=None):

        self.network = ActCritNetwork(input_size=input_size,
                                      num_actions=num_actions,
                                      eval_only=False).to(device)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        self.w_policy = w_policy
        self.w_entropy = w_entropy
        self.w_vf = w_vf
        self.clip = clip
        self.minibatch_size = minibatch_size

        self.global_step = 0
        self.updates = 0
        self.writer = epoch_writer


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
                                         num_workers=1)

        for epoch in range(epochs):

            p_loss, v_loss, e_loss, c_loss = [], [], [], []

            for i, b in enumerate(dl):

                self.optimizer.zero_grad()
                policy_loss, value_loss,  entropy_loss = self.__ppo_loss(**b)
                loss = - self.w_policy * policy_loss + \
                       self.w_vf * value_loss + \
                       - self.w_entropy * entropy_loss

                # add loss to list
                p_loss.append(policy_loss.item())
                v_loss.append(value_loss.item())
                e_loss.append(entropy_loss.item())
                c_loss.append(loss.item())

                # Update
                loss.backward()
                self.optimizer.step()


        if self.writer:
            for l_label, l_value in zip(['loss', 'policy', 'value', 'entropy'],[c_loss, p_loss, v_loss, e_loss]):
                self.writer.add_scalar(f'Loss/{l_label}', torch.tensor(l_value).mean(), self.updates)

            self.updates += 1

        print(f"Loss after update {self.updates} is {torch.tensor(c_loss).mean().item()}")
        return self.network.state_dict()

    def __ppo_loss(self,
                   states,
                   actions,
                   returns,
                   old_pi_logits,
                   advantages,
                   action_masks=None,
                   **kwargs):
        # The input will be tensors

        # We get logits estimate
        v_hat, p_hat = self.network(states)
        distr = get_distribution_from_logits(p_hat, action_masks)

        # Value loss
        value_loss = torch.mean((v_hat - returns) ** 2)

        # Entropy loss
        entropy_loss = torch.mean(distr.entropy())

        # Policy Loss
        action_log_pi = p_hat.gather(1, actions)
        ratio = action_log_pi - old_pi_logits
        surr_1 = ratio * advantages
        surr_2 = torch.clamp(ratio, min=1.0 - self.clip, max=1 + self.clip) * advantages

        policy_loss = torch.min(surr_1, surr_2).mean()

        return policy_loss, value_loss,  entropy_loss
