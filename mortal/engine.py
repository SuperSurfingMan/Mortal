import json
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from typing import *
from libriichi.consts import oracle_obs_shape
import s6c_features

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        defense_guard = None,
        extra_features = '',
        name = 'NoName',
        boltzmann_epsilon = 0,
        boltzmann_temp = 1,
        top_p = 1,
    ):
        """extra_features：obs 尾部特征族（'s6c'；默认 '' 行为不变）。

        必须与 ckpt 的 ``[control].extra_features`` 一致（用 ``s6c_features.resolve``
        解析），否则 brain 的首层通道数与输入不匹配。防守守卫始终吃未拼接的 v4 obs。
        """
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        assert extra_features in ('', s6c_features.SPEC), f'未知 extra_features: {extra_features}'
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.defense_guard = defense_guard
        self.extra_features = extra_features
        self.name = name

        self.boltzmann_epsilon = boltzmann_epsilon
        self.boltzmann_temp = boltzmann_temp
        self.top_p = top_p

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        obs_np = np.stack(obs, axis=0)
        obs = torch.as_tensor(obs_np, device=self.device)
        # S6c：brain 吃 1014 行；守卫仍吃原始 1012 行（其行号常量不受影响）
        obs_in = (obs if not self.extra_features
                  else torch.as_tensor(s6c_features.attach(obs_np), device=self.device))
        masks = torch.as_tensor(np.stack(masks, axis=0), device=self.device)
        batch_size = obs.shape[0]
        if invisible_obs is not None:
            invisible_obs = torch.as_tensor(np.stack(invisible_obs, axis=0), device=self.device)
        elif self.is_oracle:
            # Oracle Guiding 推理口径：α=0，隐藏信息通道全零（不喂任何暗牌）
            invisible_obs = torch.zeros(
                (batch_size, *oracle_obs_shape(self.version)),
                device=self.device, dtype=obs_in.dtype)

        match self.version:
            case 1:
                mu, logsig = self.brain(obs_in, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks)
            case 2 | 3 | 4:
                phi = self.brain(obs_in, invisible_obs) if self.is_oracle else self.brain(obs_in)
                q_out = self.dqn(phi, masks)

        if self.defense_guard is not None:
            # 推理期防守守卫：Q'(t) = Q(t) - λ·risk(t)（只改打牌动作 0..33）
            q_out = self.defense_guard(q_out, obs, masks)

        if self.boltzmann_epsilon > 0:
            is_greedy = torch.full((batch_size,), 1-self.boltzmann_epsilon, device=self.device).bernoulli().to(torch.bool)
            logits = (q_out / self.boltzmann_temp).masked_fill(~masks, -torch.inf)
            sampled = sample_top_p(logits, self.top_p)
            actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

def sample_top_p(logits, p):
    if p >= 1:
        return Categorical(logits=logits).sample()
    if p <= 0:
        return logits.argmax(-1)
    probs = logits.softmax(-1)
    probs_sort, probs_idx = probs.sort(-1, descending=True)
    probs_sum = probs_sort.cumsum(-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.
    sampled = probs_idx.gather(-1, probs_sort.multinomial(1)).squeeze(-1)
    return sampled

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass
