import math
import random
import re
import time
import warnings
import selectors
import sys
import cv2
import os

from gymnasium.spaces import Box
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
from torch.autograd import Variable
from typing import List, Callable
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import hdbscan
from sklearn.ensemble import IsolationForest

from robobase.envs.env import Demo, DemoEnv
from robobase.replay_buffer.replay_buffer import ReplayBuffer


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def check_for_kill_input(timeout: int = 0.0001):
    sel = selectors.DefaultSelector()
    try:
        # pytest will throw value error on this line
        sel.register(sys.stdin, selectors.EVENT_READ)
    except Exception:
        return False
    events = sel.select(timeout)
    if events:
        key, _ = events[0]
        return key.fileobj.readline().rstrip("\n").lower() == "q"
    else:
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau, update_second_net=True):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        param_to_update = target_param if update_second_net else param
        param_to_update.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.weight.data.fill_(1.0)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def uniform_weight_init(given_scale):
    def f(m):
        if isinstance(m, nn.Linear):
            in_num = m.in_features
            out_num = m.out_features
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            space = m.kernel_size[0] * m.kernel_size[1]
            in_num = space * m.in_channels
            out_num = space * m.out_channels
            denoms = (in_num + out_num) / 2.0
            scale = given_scale / denoms
            limit = np.sqrt(3 * scale)
            nn.init.uniform_(m.weight.data, a=-limit, b=limit)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.LayerNorm):
            m.weight.data.fill_(1.0)
            if hasattr(m.bias, "data"):
                m.bias.data.fill_(0.0)

    return f

def put_text(img, text, font_size=1, thickness=2, resize=False,position="top"):
    img = img.copy()
    if resize:
        img = cv2.resize(np.uint8(img), (256, 256))
    if position == "top":
        p = (10, 30)
    elif position == "bottom":
        p = (300, img.shape[0] - 20)
    # put the frame number in the top left corner
    img = cv2.putText(
        img,
        text,
        p,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (0, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return img


from sklearn.ensemble import IsolationForest
import numpy as np

from sklearn.ensemble import IsolationForest
import numpy as np

def remove_outliers_isolation_forest(data, contamination=0.1):
    model = IsolationForest(contamination=contamination)
    predictions = model.fit_predict(data.reshape(-1, 1))
    
    data = data.copy() 

    if predictions[0] == -1: 
        next_idx = 1
        while next_idx < len(data) and predictions[next_idx] == -1:
            next_idx += 1
        if next_idx < len(data):  
            data[0] = data[next_idx]

    
    if predictions[-1] == -1:  
        prev_idx = len(data) - 2
        while prev_idx >= 0 and predictions[prev_idx] == -1:
            prev_idx -= 1
        if prev_idx >= 0:  
            data[-1] = data[prev_idx]

    
    for i in range(1, len(data) - 1):
        if predictions[i] == -1:  
            prev_idx = i - 1
            while prev_idx >= 0 and predictions[prev_idx] == -1:
                prev_idx -= 1 

            next_idx = i + 1
            while next_idx < len(data) and predictions[next_idx] == -1:
                next_idx += 1  

       
            if prev_idx >= 0 and next_idx < len(data):
                data[i] = (data[prev_idx] + data[next_idx]) / 2
            elif prev_idx >= 0:
                data[i] = data[prev_idx]
            elif next_idx < len(data):
                data[i] = data[next_idx]
                
    return data



def hdbscan_with_custom_merge(entropy, dir, rollout_id, plot=True):
    
    entropy = np.array(entropy)
    entropy_norm = (entropy-np.mean(entropy))/np.std(entropy)
    entropy_norm = remove_outliers_isolation_forest(entropy_norm)
    entropy_norm = (entropy_norm-np.mean(entropy_norm))/np.std(entropy_norm)
    indices = np.arange(len(entropy_norm))
    indices = (indices-np.mean(indices))/np.std(indices)
    X = np.stack((indices,entropy_norm),axis=-1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    clusterer.fit(X)

    initial_labels = clusterer.labels_

    def split_large_clusters(labels, data, max_size=25):
        unique_labels = np.unique(labels)
        new_label = max(labels) + 1  

        for label in unique_labels:
            if label == -1: 
                continue

            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > max_size:
                cluster_points = data[cluster_indices]
                
                num_splits = len(cluster_indices) // max_size + (len(cluster_indices) % max_size > 0)
                
                for i in range(num_splits):
                    split_indices = cluster_indices[i * max_size:(i + 1) * max_size]
                    labels[split_indices] = new_label
                    new_label += 1  

        return labels
    
    initial_labels = split_large_clusters(initial_labels, X)
    
    unique_labels = np.unique(initial_labels[initial_labels >= 0])  
    
    refined_labels = np.full_like(initial_labels, -1)  

    for label in unique_labels:
        cluster_points = X[initial_labels == label]

        if  np.mean(cluster_points[:, 1] < 1):
            refined_labels[initial_labels == label] = 0  
        else:
            refined_labels[initial_labels == label] = -1  

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(entropy_norm)), entropy_norm, marker='o', markersize=5)  
        plt.title('1D Data Plot')
        plt.xlabel('Timestep')
        plt.ylabel('Entropy')
        plt.grid(True) 
        os.makedirs(os.path.join(dir, "plot"), exist_ok=True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-entropy-curve.png"))
        plt.close()

    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], X[:, 1], c=initial_labels, cmap='viridis', marker='o')
        plt.title('HDBSCAN Initial Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster Label')
        os.makedirs(os.path.join(dir, "plot"), exist_ok=True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-hdbscan-raw.png"))
        plt.close()

    if plot:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=refined_labels, cmap='viridis', marker='o')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Refined Cluster Label', rotation=270, labelpad=15)
        plt.title('HDBSCAN + Custom Merge Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.savefig(os.path.join(dir, f"plot/rollout{rollout_id}-hdbscan-refine.png"))
        plt.close()
    return np.abs(refined_labels)

def gaussian_kernel(x, bandwidth):
    """
    Calculate Gaussian Kernel
    """
    batch_size, num_samples, dim = x.size()
    
    x_i = x.unsqueeze(2)  # (batch_size, num_samples, 1, dim)
    x_j = x.unsqueeze(1)  # (batch_size, 1, num_samples, dim)
    
    distances = torch.sum((x_i - x_j) ** 2, dim=-1)  # (batch_size, num_samples, num_samples)
    
    kernel_values = torch.exp(-distances / (2 * bandwidth ** 2))
    
    return kernel_values

class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None or self._every == 0:
            return False
        every = self._every // self._action_repeat
        if step % every == 0 and step != 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, low=-1.0, high=1.0):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def _clamp(self, x):
        return torch.clamp(x, self.low, self.high)

    def sample(self, clip=None):
        return self._clamp(super().sample())


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)

class KDE():
    def __init__(self, kde_flag=True, marginal_flag=True):
        self.flag = kde_flag
        self.marginal_flag = marginal_flag
    
    def kde_entropy(self,x,k=1):
        """
        Calculate entropy by kernel density estimation
        """
        batch_size, num_samples, dim = x.size()
        if self.flag:
            bandwidth = self.estimate_bandwidth(x[0])
            self.flag = False
        bandwidth = 1 
        
        kernel_values = gaussian_kernel(x, bandwidth)  # (batch_size, num_samples, num_samples)
    
        density = kernel_values.sum(dim=2) / num_samples  # (batch_size, num_samples)
        
        max_indices = torch.argmax(density, dim=1)  # (batch_size,)

        batch_indices = torch.arange(batch_size)  #  [0, 1, ..., batch_size-1]
        max_density_points = x[batch_indices, max_indices, :]  # (batch_size, dim)

        log_density = torch.log(density + 1e-8)  
        
        entropy = -log_density.mean(dim=1, keepdim=True)  # (batch_size, 1)
        
        return entropy.squeeze(), max_density_points.squeeze()

    def estimate_bandwidth(self,x, rule='scott'):
    
        num_samples, dim = x.size()
    
        std = x.std(dim=0).mean().item() 
        if rule == 'silverman':
            bandwidth = 1.06 * std * num_samples**(-1/5)
        elif rule == 'scott':
            bandwidth = std * num_samples**(-1/(dim + 4))
        else:
            raise ValueError("Unsupported rule. Choose 'silverman' or 'scott'.")
    
        return bandwidth

def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        requires_grad=False,
    )
    # chooses between best and random actions using epsilon greedy
    return torch.stack(
        [
            argmax_acs[i] if r > eps else rand_acs[i]
            for i, r in enumerate(torch.rand(logits.shape[0]))
        ]
    )


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(logits, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = Variable(torch.zeros_like(logits).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits, tens_type=type(logits.data))
    return F.softmax(y / temperature, dim=-1)


# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y
    return y


class GumbelSoftmax(pyd.Categorical):
    @property
    def mean(self):
        init_shape = self.logits.shape
        logits_2d = self.logits.reshape(-1, self._num_events)
        return onehot_from_logits(logits_2d).view(init_shape)

    def log_prob(self, value):
        return (self.logits * value).max(-1)[0]

    def sample(self, temp=1.0, hard=True):
        init_shape = self.logits.shape
        logits_2d = self.logits.reshape(-1, self._num_events)
        return gumbel_softmax(logits_2d, temp, hard=hard).view(init_shape)


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


def torch_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def soft_argmax(onehot_actions, low, high, bins, alpha=10.0):
    soft_max = torch.softmax(onehot_actions * alpha, dim=-1)
    indices_kernel = torch_linspace(low, high, bins)
    values = (soft_max * indices_kernel).sum(-1)
    return values


class DemoStep(dict):
    """A step of a demo which holds state along with joint and gripper positions."""

    def __init__(
        self,
        joint_positions: np.ndarray,
        gripper_open: float,
        state: dict,
        gripper_matrix: np.array = None,
        misc: dict = {},
    ):
        """Init.

        Args:
            joint_positions (np.ndarray): joint positions excluding the gripper.
            gripper_open (float): value between 0.0 and 1.0 representing open and
                closed respectively.
            state (dict): state observations expected as inputs to the model.
        """
        super().__init__(**state)
        self.joint_positions = joint_positions
        self.gripper_open = gripper_open
        self.gripper_matrix = gripper_matrix
        self.misc = misc


def observations_to_action_with_onehot_gripper(
    current_observation: DemoStep,
    next_observation: DemoStep,
    action_space: Box,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """
    action = np.concatenate(
        [
            (
                next_observation.misc["joint_position_action"][:-1]
                - current_observation.joint_positions
                if "joint_position_action" in next_observation.misc
                else next_observation.joint_positions
                - current_observation.joint_positions
            ),
            [1.0 if next_observation.gripper_open == 1 else 0.0],
        ]
    ).astype(np.float32)
    if np.any(action[:-1] > action_space.high[:-1]) or np.any(
        action[:-1] < action_space.low[:-1]
    ):
        return None
    return action


def observations_to_action_with_onehot_gripper_nbp(
    current_observation: DemoStep,
    next_observation: DemoStep,
    action_space: Box,
):
    """Calculates the action linking two sequential observations.

    Args:
        current_observation (DemoStep): the observation made before the action.
        next_observation (DemoStep): the observation made after the action.
        action_space (Box): the action space of the unwrapped env.

    Returns:
        np.ndarray: action taken at current observation. Returns None if action
            outside action_space.
    """

    action_trans = next_observation.gripper_matrix[:3, 3]

    rot = R.from_matrix(next_observation.gripper_matrix[:3, :3])
    action_orien = rot.as_quat(
        canonical=True
    )  # Enforces w component always positive and unit vector

    action_gripper = [1.0 if next_observation.gripper_open == 1 else 0.0]
    action = np.concatenate(
        [
            action_trans,
            action_orien,
            action_gripper,
        ]
    )

    if np.any(action[:-1] > action_space.high[:-1]) or np.any(
        action[:-1] < action_space.low[:-1]
    ):
        warnings.warn(
            "Action outside action space.",
            UserWarning,
        )
        return None
    return action


def observations_to_timesteps(
    demo: List[DemoStep],
    action_space: Box,
    skipping: bool = True,
    obs_to_act_func: Callable[
        [DemoStep, DemoStep, Box], np.ndarray
    ] = observations_to_action_with_onehot_gripper,
):
    """Converts demo steps into timesteps.

    Args:
        demo (List[DemoStep]): an episode.
        action_space (Box): the actions space of the unwrapped env.
        skipping (bool): option to augment demonstration data through observations.
        obs_to_act_func: function to call for determining action.

    Returns:
        List[List[Tuple]]: a list of timestep demonstrations. Each demonstration
            ends with the following format where a_t is stored in info_{t+1}:

                [(s_0), (s_1, r_1, term_1, trunc_1, info_1), ...]
    """
    loaded_demos = []
    skip = 1
    first_step = demo[0]
    # enter loop until skipping more observations goes outside action_space
    while True:
        info = {"demo": 1}
        # add first observation to demo_timesteps following format defined above
        demo_timesteps = [(first_step, info)]
        i = 0
        while i < len(demo[:-1]):
            demo_step = demo[i]
            r = 0.0
            done = False
            # find the next observation
            for j in range(1, 1 + skip):
                next_demo_step = demo[i + j]
                if (i + j) >= (len(demo) - 1):
                    r = 1.0
                    done = True
                    break
                if next_demo_step.gripper_open != demo_step.gripper_open:
                    break
            i += j
            # calculate action
            action = obs_to_act_func(demo_step, next_demo_step, action_space)
            # wipe demo_timesteps if action outside action_space
            if action is None:
                demo_timesteps = []
                break
            # add action into info to be extracted later
            info = {"demo_action": action, "demo": 1}
            demo_timesteps.append(
                (
                    next_demo_step,
                    r,
                    done,
                    False,
                    info,
                )
            )
        if len(demo_timesteps) == 0:
            break
        loaded_demos.append(Demo(demo_timesteps))
        if skipping:
            skip += 1
        else:
            break
    return loaded_demos


def rescale_demo_actions(rescale_fn: Callable, demos: List[Demo], cfg: DictConfig):
    """Rescale actions in demonstrations to [-1, 1] Tanh space.
    This is because RoboBase assumes everything to be in [-1, 1] space.

    Args:
        rescale_fn: callable that takes info containing demo action and cfg and
            outputs the rescaled action
        demos: list of demo episodes whose actions are raw, i.e., not scaled
        cfg: Configs

    Returns:
        List[Demo]: list of demo episodes whose actions are rescaled
    """
    for demo in demos:
        for step in demo:
            *_, info = step
            if "demo_action" in info:
                # Rescale demo actions
                info["demo_action"] = rescale_fn(info, cfg)
    return demos


def add_demo_to_replay_buffer(wrapped_env: DemoEnv, replay_buffer: ReplayBuffer):
    """Loads demos into replay buffer by passing observations through wrappers.

    CYCLING THROUGH DEMOS IS HANDLED BY WRAPPED ENV.

    Args:
        wrapped_env: the fully wrapped environment.
        replay_buffer: replay buffer to be loaded.
    """
    is_sequential = replay_buffer.sequential
    ep = []

    # Extract demonstration episode in replay buffer transitions
    obs, info = wrapped_env.reset()
    fake_action = wrapped_env.action_space.sample()
    term, trunc = False, False
    while not (term or trunc):
        next_obs, rew, term, trunc, next_info = wrapped_env.step(fake_action)
        action = next_info.pop("demo_action")
        assert np.all(action <= 1.0)
        assert np.all(action >= -1.0)
        ep.append([obs, action, rew, term, trunc, info, next_info])
        obs = next_obs
        info = next_info
    final_obs, _ = obs, info
    for obs, action, rew, term, trunc, info, _ in ep:
        replay_buffer.add(obs, action, rew, term, trunc, demo=info["demo"])

    if not is_sequential:
        replay_buffer.add_final(final_obs)


def merge_replay_demo_iter(replay_iter, demo_replay_iter):
    return iter(DemoMergedIterator(replay_iter, demo_replay_iter))


class DemoMergedIterator:
    def __init__(self, replay_iter, demo_replay_iter):
        self.replay_iter = replay_iter
        self.demo_replay_iter = demo_replay_iter
        self._is_safe = False

    def __iter__(self):
        return self

    def _check_keys(self, batch, demo_batch):
        assert set(batch.keys()) == set(
            demo_batch.keys()
        ), f"Keys in demo batch are different: {batch.keys()}, {demo_batch.keys()}"

    def __next__(self):
        batch = next(self.replay_iter)
        demo_batch = next(self.demo_replay_iter)
        if not self._is_safe:
            self._check_keys(batch, demo_batch)
            self._is_safe = True
        # Override demo to be 1 for demo_batch
        demo_batch["demo"] = torch.ones_like(demo_batch["demo"])
        return {k: torch.cat([batch[k], demo_batch[k]], 0) for k in batch.keys()}
