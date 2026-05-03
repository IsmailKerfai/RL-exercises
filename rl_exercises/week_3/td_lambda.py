from __future__ import annotations

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent


class RandomWalkPredictionEnv(gym.Env):
    """Random walk from the paper."""

    def __init__(
        self,
        n_states=7,
        start_state=None,
        seed=None,
    ):
        assert n_states >= 3

        self.n_states = n_states
        self.left_terminal = 0
        self.right_terminal = n_states - 1
        if start_state is None:
            self.start_state = n_states // 2
        else:
            self.start_state = start_state

        self.rng = np.random.default_rng(seed)

        self.observation_space = gym.spaces.Discrete(n_states)
        self.action_space = gym.spaces.Discrete(2)
        self.state = self.start_state

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start_state
        return self.state, {}

    def step(self, action):
        # action 0 moves left, action 1 moves right
        self.state += -1 if action == 0 else 1

        terminated = False
        if self.state == self.left_terminal or self.state == self.right_terminal:
            terminated = True

        reward = 1.0 if self.state == self.right_terminal else 0.0
        return self.state, reward, terminated, False, {}

    def true_values(self):
        values = np.zeros(self.n_states, dtype=float)
        for state in range(1, self.right_terminal):
            values[state] = state / self.right_terminal
        return values

    def render(self, mode="human"):
        print("state:", self.state)


class TDLambdaAgent(AbstractAgent):
    """TD(lambda) value prediction."""

    def __init__(
        self,
        env,
        alpha=0.1,
        gamma=1.0,
        lambda_=0.8,
        seed=None,
    ):
        assert alpha > 0
        assert 0 <= gamma <= 1
        assert 0 <= lambda_ <= 1

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.rng = np.random.default_rng(seed)
        self.V = np.zeros(env.observation_space.n, dtype=float)
        self.eligibility = np.zeros_like(self.V)

    def predict_action(self, state, info={}, evaluate=False):  # type: ignore # noqa
        return int(self.rng.integers(self.env.action_space.n)), info or {}

    def reset_episode(self):
        self.eligibility.fill(0.0)

    def update_agent(self, batch) -> float:  # type: ignore
        state, _, reward, next_state, done, _ = batch[0]
        return self.TD_lambda(state, reward, next_state, done)

    def TD_lambda(self, state, reward, next_state, done):
        next_value = 0.0 if done else self.V[next_state]
        delta = reward + self.gamma * next_value - self.V[state]

        self.eligibility[state] += 1.0
        self.V += self.alpha * delta * self.eligibility
        self.eligibility *= self.gamma * self.lambda_
        return float(self.V[state])

    def save(self, path):  # type: ignore
        np.save(path, self.V)

    def load(self, path):  # type: ignore
        self.V = np.load(path)
        self.eligibility = np.zeros_like(self.V)


def run_td_lambda_experiment(
    lambdas=(0.0, 0.3, 0.8, 1.0),
    episodes=100,
    alpha=0.1,
    gamma=1.0,
    seed=0,
):
    """Compare a few lambda values on the random walk."""
    rows = []

    for lambda_ in lambdas:
        env = RandomWalkPredictionEnv(seed=seed)
        agent = TDLambdaAgent(
            env=env,
            alpha=alpha,
            gamma=gamma,
            lambda_=lambda_,
            seed=seed,
        )

        for episode in range(episodes):
            state, info = env.reset(seed=seed + episode)
            agent.reset_episode()
            done = False

            while not done:
                action, _ = agent.predict_action(state, info)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.update_agent([(state, action, reward, next_state, done, info)])
                state = next_state

        true_values = env.true_values()
        nonterminal = slice(1, env.right_terminal)
        value_errors = agent.V[nonterminal] - true_values[nonterminal]
        rmse = float(np.sqrt(np.mean(value_errors**2)))

        row = {"lambda": float(lambda_), "rmse": rmse}
        for state in range(1, env.right_terminal):
            row[f"V_{state}"] = float(agent.V[state])
            row[f"true_V_{state}"] = float(true_values[state])
        rows.append(row)

    return rows
