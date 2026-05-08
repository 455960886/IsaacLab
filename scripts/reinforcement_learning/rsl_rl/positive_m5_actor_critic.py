# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic import ActorCritic, resolve_nn_activation


class _PositiveM5ActorWrapper(nn.Module):
    """Wrap the base actor so that the M5 output is always in [0, pi]."""

    def __init__(self, base_actor: nn.Module, m5_action_index: int = 2, m5_max: float = math.pi / 2):
        super().__init__()
        # base_actor 是真正负责从 observation 预测 action 均值的网络。
        # 这个 wrapper 不改变其它动作，只单独处理 M5 这一维。
        self.base_actor = base_actor

        # m5_action_index 表示 M5 对应动作在 action 向量里的下标。
        # 当前任务里默认第 2 维是 M5。
        self.m5_action_index = m5_action_index

        # m5_max 是 M5 允许输出的最大角度。
        # 下面 forward 里会用 sigmoid 把这一维限制到 [0, m5_max]。
        self.m5_max = m5_max

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 先让原始 actor 输出所有动作。
        actions = self.base_actor(observations)

        # clone 一份，避免原地修改 autograd 计算图中的张量。
        actions = actions.clone()

        # 只处理 M5 这一维：
        # sigmoid 输出范围是 [0, 1]，再乘以 m5_max，就得到 [0, m5_max]。
        # 这样可以避免 M5 输出负角度。
        actions[..., self.m5_action_index] = torch.sigmoid(actions[..., self.m5_action_index]) * self.m5_max
        return actions


class PositiveM5ActorCritic(ActorCritic):
    """ActorCritic variant whose M5 action is exported and inferred as an absolute angle in [0, pi]."""

    def __init__(self, *args, m5_action_index: int = 2, m5_max: float = math.pi, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = _PositiveM5ActorWrapper(self.actor, m5_action_index=m5_action_index, m5_max=m5_max)


def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation_name: str) -> nn.Sequential:
    """构建普通 MLP，用在 actor head 和 critic 上。"""

    # RSL-RL 里 activation 是字符串，比如 "elu"、"relu"。
    # resolve_nn_activation 会把字符串转换成真正的 torch activation module。
    activation = resolve_nn_activation(activation_name)
    layers: list[nn.Module] = []
    prev_dim = input_dim

    # 依次堆叠 Linear + activation。
    # 例如 hidden_dims=[512, 256] 时，会得到 input->512->256。
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim

    # 最后一层只做 Linear，不再加 activation。
    # actor 的 output_dim 是 num_actions，critic 的 output_dim 是 1。
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class _TransformerFusionActor(nn.Module):
    """用 Transformer 融合夹爪 RGB、前向 ToF 点云和关节状态特征。

    注意：这里不是直接处理原始图片或原始点云。
    原始图片和点云已经在 observations.py 里分别被 ResNet18 和 PointNet2 提成特征。
    这个 actor 收到的是一个拼接后的 flat observation：

        [夹爪 RGB 特征 512维, 前向 ToF 点云特征 1024维, 关节状态 5维]

    这里做的事情是：
        1. 把 flat observation 重新切成三个模态；
        2. 每个模态先通过自己的 adapter 变成同样长度的 token；
        3. 把这些 token 送进 Transformer，让不同模态互相注意；
        4. 用融合后的 token 输出动作。
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int],
        activation: str,
        image_feature_dim: int = 512,
        pointcloud_feature_dim: int = 1024,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 当前策略 observation 总长度是 1541：
        #   image_feature_dim      = 512   夹爪 RGB 经过 ResNet18 后的特征
        #   pointcloud_feature_dim = 1024  前向 ToF 点云经过 PointNet2 后的特征
        #   state_dim              = 5     关节状态
        #
        # 这里不把 state_dim 写死为 5，而是用总长度自动算出来。
        # 这样如果以后 joint_pos 维度变化，只要前两个特征维度不变，这里仍能工作。
        state_dim = num_actor_obs - image_feature_dim - pointcloud_feature_dim
        if state_dim <= 0:
            raise ValueError(
                "TransformerFusionActor expects actor obs layout "
                f"[image:{image_feature_dim}, pointcloud:{pointcloud_feature_dim}, state:remaining], "
                f"but got num_actor_obs={num_actor_obs}."
            )
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}.")

        self.image_feature_dim = image_feature_dim
        self.pointcloud_feature_dim = pointcloud_feature_dim
        self.state_dim = state_dim

        # adapter 的作用：
        # 不同模态原始特征维度不同，数值分布也不同。
        # 例如图像是 512 维，点云是 1024 维，关节只有 5 维。
        #
        # Transformer 要求每个 token 的维度相同，所以先把三种输入都投影到 embed_dim=128。
        # LayerNorm 放在前面，用于每个模态单独归一化，避免某个模态因为数值尺度大而压过其它模态。
        self.image_adapter = nn.Sequential(nn.LayerNorm(image_feature_dim), nn.Linear(image_feature_dim, embed_dim))
        self.pointcloud_adapter = nn.Sequential(
            nn.LayerNorm(pointcloud_feature_dim), nn.Linear(pointcloud_feature_dim, embed_dim)
        )
        self.state_adapter = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, embed_dim))

        # CLS token 是一个可学习的“汇总 token”。
        # 它本身不对应任何一个传感器，而是通过 attention 从 image/pointcloud/state 三个 token 里收集信息。
        # Transformer 输出后，我们取 CLS token 作为融合后的全局表示。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # modality_embedding 用来告诉 Transformer 每个 token 的来源。
        # 4 个 token 分别是：
        #   0: CLS token
        #   1: image token
        #   2: pointcloud token
        #   3: state token
        # 如果不加这个 embedding，Transformer 只能看到 4 个 128 维向量，但不知道谁来自哪个模态。
        self.modality_embedding = nn.Parameter(torch.zeros(1, 4, embed_dim))

        # TransformerEncoderLayer 是标准 Transformer 编码层。
        # d_model=128 表示每个 token 是 128 维。
        # nhead=4 表示 4 个 attention head，会从不同子空间学习模态之间的关系。
        # dim_feedforward=256 是每层 Transformer 内部 MLP 的隐藏维度。
        # norm_first=True 表示使用 pre-norm，一般训练更稳定。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 这里堆叠 num_layers=2 层 Transformer。
        # enable_nested_tensor=False 是为了避免 PyTorch 在 norm_first=True 时打印无关 warning。
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False
        )

        # Transformer 输出的 CLS token 是 128 维。
        # 我们还额外拼接 state_token，保留一条关节状态的直接通路。
        # 所以 actor head 的输入是 128 + 128 = 256 维。
        self.actor_head = _build_mlp(embed_dim * 2, actor_hidden_dims, num_actions, activation)

        # 用较小随机数初始化可学习 token，避免训练开始时数值过大。
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations 的形状是 [num_envs, 1541]。
        # 这里根据约定好的 layout，把 flat observation 切回三个部分。
        img_end = self.image_feature_dim
        pc_end = img_end + self.pointcloud_feature_dim

        # image_features:      [B, 512]
        # pointcloud_features: [B, 1024]
        # state_features:      [B, 5]
        # B 表示并行环境数量，比如训练时可能是几百或几千。
        image_features = observations[..., :img_end]
        pointcloud_features = observations[..., img_end:pc_end]
        state_features = observations[..., pc_end:]

        # 三个 adapter 分别把不同模态变成同样维度的 token：
        # image_token:      [B, 128]
        # pointcloud_token: [B, 128]
        # state_token:      [B, 128]
        image_token = self.image_adapter(image_features)
        pointcloud_token = self.pointcloud_adapter(pointcloud_features)
        state_token = self.state_adapter(state_features)

        # cls_token 参数本身形状是 [1, 1, 128]。
        # expand 后复制到每个环境，变成 [B, 1, 128]。
        cls_token = self.cls_token.expand(observations.shape[0], -1, -1)

        # torch.stack 后：
        #   [image_token, pointcloud_token, state_token] -> [B, 3, 128]
        tokens = torch.stack((image_token, pointcloud_token, state_token), dim=1)

        # 拼上 CLS token 后：
        #   [CLS, image, pointcloud, state] -> [B, 4, 128]
        tokens = torch.cat((cls_token, tokens), dim=1)

        # 加上 modality embedding，让 Transformer 知道每个 token 的身份。
        tokens = tokens + self.modality_embedding

        # Transformer 在 4 个 token 之间做 self-attention。
        # 这一步会学习：
        #   图像 token 应该关注点云 token 的哪些信息；
        #   点云 token 应该关注关节 token 的哪些信息；
        #   CLS token 应该从所有模态里汇总哪些信息。
        fused_tokens = self.transformer(tokens)

        # 取第 0 个 token，也就是 CLS token，作为多模态融合后的全局表示。
        # fused_token: [B, 128]
        fused_token = fused_tokens[:, 0]

        # 这里保留一条 state_token 直连 actor head 的路径。
        # 原因是关节状态是低维且非常重要的本体感觉信息，不应该完全依赖 Transformer 间接传递。
        # actor_input: [B, 256]
        actor_input = torch.cat((fused_token, state_token), dim=-1)

        # 输出动作均值，形状是 [B, num_actions]。
        # 后续 RSL-RL 会根据这个均值和 std 构造高斯分布采样动作。
        return self.actor_head(actor_input)


class TransformerFusionActorCritic(ActorCritic):
    """带 Transformer 融合 actor 的 RSL-RL ActorCritic。

    RSL-RL 要求 policy 类提供和 ActorCritic 类似的接口：
        - self.actor
        - self.critic
        - self.std 或 self.log_std
        - act / act_inference / evaluate 等方法

    这些方法已经在父类 ActorCritic 里实现。
    这里继承 ActorCritic，但不直接调用 ActorCritic.__init__，
    因为父类会先构造一个普通 MLP actor，而我们想用 Transformer actor 替换它。
    所以这里手动初始化 nn.Module，再手动创建 actor、critic 和动作噪声参数。
    """

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        m5_action_index: int = 2,
        m5_max: float = math.pi,
        **kwargs,
    ):
        if kwargs:
            print(
                "TransformerFusionActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        # 不调用 ActorCritic.__init__，只初始化 PyTorch Module 的基础状态。
        # 这样可以避免先创建一个无用的普通 MLP actor。
        nn.Module.__init__(self)
        print("num_actor_obs:", num_actor_obs)
        print("num_critic_obs:", num_critic_obs)

        # 创建新的 Transformer actor。
        # 它输入 actor observation，也就是 policy obs 的 1541 维。
        fusion_actor = _TransformerFusionActor(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
        )

        # 外面再套一层 M5 wrapper，保证 M5 动作范围仍然是 [0, pi]。
        self.actor = _PositiveM5ActorWrapper(fusion_actor, m5_action_index=m5_action_index, m5_max=m5_max)

        # critic 没有使用 Transformer。
        # 当前任务里 critic observation 是低维 privileged observation，例如 joint_pos + object_yaw。
        # 所以普通 MLP 就足够。
        self.critic = _build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)

        # PPO actor 输出的是动作均值 mean。
        # RSL-RL 还需要一个动作标准差 std，用 mean 和 std 构造高斯策略分布。
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            # scalar 表示每个动作维度一个可学习标准差。
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            # log 表示学习 log_std，使用时再 exp 成标准差。
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # distribution 会在 ActorCritic.update_distribution() 里被创建。
        # 训练时 RSL-RL 会用它采样动作、计算 log_prob 和 entropy。
        self.distribution = None

        # 关闭 torch.distributions.Normal 的参数检查，提高训练时的执行速度。
        # 这是沿用 RSL-RL 原始 ActorCritic 的做法。
        Normal.set_default_validate_args(False)

        print(f"Transformer Fusion Actor: {self.actor}")
        print(f"Critic MLP: {self.critic}")
