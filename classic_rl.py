
import warnings
import pickle

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


###############################################################################
###############################################################################
# Value Iteration Algorithm

def perform_value_iteration(env):
    gamma = 1.0
    theta = 1e-10

    V = np.zeros(env.observation_space.n, dtype=np.float64)

    while True:
        Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)

        for state in range(len(env.P)):
            for action in range(len(env.P[state])):
                for prob, next_state, reward, terminated in env.P[state][action]:
                    Q[state][action] += prob * (reward + gamma * V[next_state] * (not terminated))

        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break

        V = np.max(Q, axis=1)

    Pi = {state: action for state, action in enumerate(np.argmax(Q, axis=1))}

    return V, Pi

# Value Iteration Algorithm
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Policy Iteration Algorithm

def perform_policy_iteration(env):

    def policy_eval(env, Pi):
        gamma = 1.0
        theta = 1e-10

        prev_V = np.zeros(env.observation_space.n, dtype=np.float64)

        while True:
            V = np.zeros(env.observation_space.n, dtype=np.float64)

            for state in range(len(env.P)):
                for prob, next_state, reward, terminated in env.P[state][Pi[state]]:
                    V[state] += prob * (reward + gamma * prev_V[next_state] * (not terminated))

            if np.max(np.abs(prev_V - V)) < theta:
                break

            prev_V = V.copy()

        return V


    def policy_impr(env, V):
        gamma = 1.0

        Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)

        for state in range(len(env.P)):
            for action in range(len(env.P[state])):
                for prob, next_state, reward, terminated in env.P[state][action]:
                    Q[state][action] += prob * (reward + gamma * V[next_state] * (not terminated))

        Pi = {state: action for state, action in enumerate(np.argmax(Q, axis=1))}

        return Pi

    gamma = 1.0
    theta = 1e-10

    random_actions = np.random.choice(tuple(env.P[0].keys()), env.observation_space.n)
    Pi = {state: action for state, action in enumerate(random_actions)}

    while True:
        prev_Pi = {state: Pi[state] for state in range(env.observation_space.n)}

        V = policy_eval(env, Pi)
        Pi = policy_impr(env, V)

        if prev_Pi == {state: Pi[state] for state in range(env.observation_space.n)}:
            break

    return V, Pi

# Policy Iteration Algorithm
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# SARSA Algorithm

def perform_sarsa(env):

    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
        return values

    gamma = 1.0
    init_alpha = 0.5
    min_alpha = 0.01
    alpha_decay_ratio = 0.5
    init_epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay_ratio = 0.9
    n_episodes = 3000

    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    Q_track = np.zeros((n_episodes, env.observation_space.n, env.action_space.n), dtype=np.float64)
    Pi_track = []

    action_selection = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    for episode in range(n_episodes):
        state, info = env.reset()

        terminated = False

        action = action_selection(state, Q, epsilons[episode])

        while not terminated:
            next_state, reward, terminated, _, info = env.step(action)
            next_action = action_selection(next_state, Q, epsilons[episode])

            td_target = reward + gamma * Q[next_state][next_action] * (not terminated)
            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alphas[episode] * td_error

            state, action = next_state, next_action

        Q_track[episode] = Q
        Pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    Pi = {state: action for state, action in enumerate(np.argmax(Q, axis=1))}

    return Q, V, Pi, Q_track, Pi_track

# SARSA Algorithm
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Expected SARSA Algorithm

def perform_expected_sarsa(env):

    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
        return values

    gamma = 1.0
    init_alpha = 0.5
    min_alpha = 0.01
    alpha_decay_ratio = 0.5
    init_epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay_ratio = 0.9
    n_episodes = 3000

    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    Q_track = np.zeros((n_episodes, env.observation_space.n, env.action_space.n), dtype=np.float64)
    Pi_track = []

    action_selection = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    for episode in range(n_episodes):
        state, info = env.reset()

        terminated = False

        while not terminated:
            action = action_selection(state, Q, epsilons[episode])
            next_state, reward, terminated, _, info = env.step(action)

            expected_value = np.sum([Q[next_state][a] * (1 - epsilons[episode] + (epsilons[episode] / env.action_space.n)) if a == np.argmax(Q[next_state])
                                     else Q[next_state][a] * (epsilons[episode] / env.action_space.n)
                                     for a in range(env.action_space.n)])

            td_target = reward + gamma * expected_value * (not terminated)
            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alphas[episode] * td_error

            state = next_state

        Q_track[episode] = Q
        Pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    Pi = {state: action for state, action in enumerate(np.argmax(Q, axis=1))}

    return Q, V, Pi, Q_track, Pi_track

# Expected SARSA Algorithm
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Q-learning Algorithm

def perform_qlearning(env):

    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
        return values

    gamma = 1.0
    init_alpha = 0.5
    min_alpha = 0.01
    alpha_decay_ratio = 0.5
    init_epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay_ratio = 0.9
    n_episodes = 3000

    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    Q_track = np.zeros((n_episodes, env.observation_space.n, env.action_space.n), dtype=np.float64)
    Pi_track = []

    action_selection = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    for episode in range(n_episodes):
        state, info = env.reset()

        terminated = False

        while not terminated:
            action = action_selection(state, Q, epsilons[episode])
            next_state, reward, terminated, _, info = env.step(action)

            td_target = reward + gamma * Q[next_state].max() * (not terminated)
            td_error = td_target - Q[state][action]

            Q[state][action] = Q[state][action] + alphas[episode] * td_error

            state = next_state

        Q_track[episode] = Q
        Pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)        
    Pi = {state: action for state, action in enumerate(np.argmax(Q, axis=1))}

    return Q, V, Pi, Q_track, Pi_track

# Q-learning Algorithm
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Double Q-learning Algorithm

def perform_double_qlearning(env):

    def decay_schedule(init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10):
        decay_steps = int(max_steps * decay_ratio)
        rem_steps = max_steps - decay_steps
        values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
        values = (values - values.min()) / (values.max() - values.min())
        values = (init_value - min_value) * values + min_value
        values = np.pad(values, (0, rem_steps), "edge")
        return values

    gamma = 1.0
    init_alpha = 0.5
    min_alpha = 0.01
    alpha_decay_ratio = 0.5
    init_epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay_ratio = 0.9
    n_episodes = 3000

    Q1 = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    Q2 = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    Q1_track = np.zeros((n_episodes, env.observation_space.n, env.action_space.n), dtype=np.float64)
    Q2_track = np.zeros((n_episodes, env.observation_space.n, env.action_space.n), dtype=np.float64)
    Pi_track = []
    
    action_selection = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    
    for episode in range(n_episodes):
        state, info = env.reset()

        terminated = False

        while not terminated:
            action = action_selection(state, (Q1 + Q2) / 2, epsilons[episode])
            next_state, reward, terminated, _, info = env.step(action)

            if np.random.randint(2):
                argmax_Q1 = np.argmax(Q1[next_state])
                td_target = reward + gamma * Q2[next_state][argmax_Q1] * (not terminated)
                td_error = td_target - Q1[state][action]
                Q1[state][action] = Q1[state][action] + alphas[episode] * td_error
            else:
                argmax_Q2 = np.argmax(Q2[next_state])
                td_target = reward + gamma * Q1[next_state][argmax_Q2] * (not terminated)
                td_error = td_target - Q2[state][action]
                Q2[state][action] = Q2[state][action] + alphas[episode] * td_error

            state = next_state

        Q1_track[episode] = Q1
        Q2_track[episode] = Q2        
        Pi_track.append(np.argmax((Q1 + Q2) / 2, axis=1))

    Q = (Q1 + Q2) / 2.
    V = np.max(Q, axis=1)    
    Pi = {state: action for state, action in enumerate(np.argmax(Q, axis=1))}

    return Q, V, Pi, (Q1_track + Q2_track) / 2., Pi_track

# Double Q-learning Algorithm
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# First-visit Monte Carlo Control Algorithm

def perform_fv_mc_control(env):
    pass

# First-visit Monte Carlo Control Algorithm
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Mountain Car Experiments

def mountain_car_env_experiments():
    """
    this environment is solved using Q-learning algorithm.
    """

    env = gym.make("MountainCar-v0")

    episodes = 5000
    is_training = True

    # divide position and velocity into segments:
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20) # between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20) # between -0.07 and 0.07

    if is_training:
        Q = np.zeros((len(pos_space), len(vel_space), env.action_space.n)) # init a 20x20x3 array
    else:
        with open("mountain_car.pkl", "rb") as file:
            Q = pickle.load(file)

    alpha = 0.9 # learning rate
    gamma = 0.9 # discount factor.

    epsilon = 1
    epsilon_decay_rate = 2 / episodes
    rng = np.random.default_rng() # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state, _ = env.reset() # starting position, starting velocity always 0
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)

        terminated = False

        rewards = 0

        while not terminated and rewards > -1000:
            if is_training and rng.random() < epsilon:
                # choose random action (0=drive left, 1=stay neutral, 2=drive right):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_p, state_v, :])

            next_state, reward, terminated, _, _ = env.step(action)
            next_state_p = np.digitize(next_state[0], pos_space)
            next_state_v = np.digitize(next_state[1], vel_space)

            if is_training:
                Q[state_p, state_v, action] = Q[state_p, state_v, action] + alpha * (reward + gamma * np.max(Q[next_state_p, next_state_v,:]) - Q[state_p, state_v, action])

            state = next_state
            state_p = next_state_p
            state_v = next_state_v

            rewards += reward

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episode[i] = rewards

    env.close()

    # save Q-table:
    if is_training:
        with open("mountain_car.pkl", "wb") as file:
            pickle.dump(Q, file)

    mean_rewards = np.zeros(episodes)

    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(mean_rewards)
    #plt.savefig("mountain_car.png")
    plt.show()

# Mountain Car Experiments
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Cartpole Experiments

def cartpole_env_experiments():
    """
    this environment is solved using Q-learning algorithm.
    """

    env = gym.make("CartPole-v1")

    # divide position, velocity, pole angle, and pole angular velocity into segments:
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    is_training = True

    if is_training:
        Q = np.zeros((len(pos_space) + 1, len(vel_space) + 1, len(ang_space) + 1, len(ang_vel_space) + 1, env.action_space.n)) # init a 11x11x11x11x2 array
    else:
        with open("cartpole.pkl", "rb") as file:
            Q = pickle.load(file)

    alpha = 0.1 # learning rate
    gamma = 0.99 # discount factor.

    epsilon = 1
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()

    rewards_per_episode = []

    i = 0

    while True:
        state, _ = env.reset()
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False

        rewards = 0

        while not terminated and rewards < 10000:

            if is_training and rng.random() < epsilon:
                # choose random action (0=go left, 1=go right):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state_p, state_v, state_a, state_av, :])

            next_state, reward, terminated, _, _ = env.step(action)
            next_state_p = np.digitize(next_state[0], pos_space)
            next_state_v = np.digitize(next_state[1], vel_space)
            next_state_a = np.digitize(next_state[2], ang_space)
            next_state_av= np.digitize(next_state[3], ang_vel_space)

            if is_training:
                Q[state_p, state_v, state_a, state_av, action] = Q[state_p, state_v, state_a, state_av, action] + alpha * (reward + gamma * np.max(Q[next_state_p, next_state_v, next_state_a, next_state_av,:]) - Q[state_p, state_v, state_a, state_av, action])

            state = next_state
            state_p = next_state_p
            state_v = next_state_v
            state_a = next_state_a
            state_av= next_state_av

            rewards += reward

            if not is_training and rewards % 100 == 0:
                print(f"Episode: {i}  Rewards: {rewards}")

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[len(rewards_per_episode) - 100:])

        if is_training and i % 100 == 0:
            print(f"Episode: {i} {rewards}  Epsilon: {epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}")

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i += 1

    env.close()

    # save Q-table:
    if is_training:
        with open("cartpole.pkl", "wb") as file:
            pickle.dump(Q, file)

    mean_rewards = []

    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]))

    plt.plot(mean_rewards)
    #plt.savefig(f'cartpole.png')
    plt.show()

# Cartpole Experiments
###############################################################################
###############################################################################


###############################################################################
###############################################################################
# Taxi Experiments

def taxi_env_experiments():
    env = gym.make("Taxi-v3")

    is_training = True
    episodes = 15000

    if is_training:
        Q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 500 x 6 array
    else:
        with open("taxi.pkl", "rb") as file:
            Q = pickle.load(file)

    alpha = 0.9 # learning rate
    gamma = 0.9 # discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1
    epsilon_decay_rate = 0.0001

    rng = np.random.default_rng() # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False # True when fall in hole or reached goal
        truncated = False # True when actions > 200

        rewards = 0
        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(Q[state, :])

            next_state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward

            if is_training:
                Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

            state = next_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            alpha = 0.0001

        rewards_per_episode[i] = rewards

    env.close()

    sum_rewards = np.zeros(episodes)

    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])

    plt.plot(sum_rewards)
    #plt.savefig('taxi.png')
    plt.show()

    # save Q-table:
    if is_training:
        with open("taxi.pkl", "wb") as file:
            pickle.dump(Q, file)

# Taxi Experiments
###############################################################################
###############################################################################


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    #env = gym.make("CliffWalking-v0")
    #env = gym.make("Taxi-v3")

    state, info = env.reset()
    """
    # Value Iteration Algorithm:
    V, Pi = perform_value_iteration(env)
    print("\nValue Iteration:")
    print(f"{V = }")
    print(f"{Pi = }")
    print()

    # Policy Iteration Algorithm:
    V, Pi = perform_policy_iteration(env)
    print("Policy Iteration:")
    print(f"{V = }")
    print(f"{Pi = }")
    print()
    sdf
    """
    # SARSA Algorithm:
    _, V, Pi, _, _ = perform_sarsa(env)
    print("SARSA:")
    print(f"{V = }")
    print(f"{Pi = }")
    print()

    # Expected SARSA Algorithm:
    _, V, Pi, _, _ = perform_expected_sarsa(env)
    print("Expected SARSA:")
    print(f"{V = }")
    print(f"{Pi = }")
    print()

    # Q-learning Algorithm:
    _, V, Pi, _, _ = perform_qlearning(env)
    print("Q-learning:")
    print(f"{V = }")
    print(f"{Pi = }")
    print()

    # Double Q-learning Algorithm:
    _, V, Pi, _, _ = perform_double_qlearning(env)
    print("Double Q-learning:")
    print(f"{V = }")
    print(f"{Pi = }")
    print()

    # First-visit Monte Carlo Algorithm:
    _, V, Pi, _, _ = perform_fv_mc_control(env)
    print("First-visit Monte Carlo Control:")
    print(f"{V = }")
    print(f"{Pi = }")
    print()

    ###########################################################################
    #mountain_car_env_experiments()
    #cartpole_env_experiments()
    #taxi_env_experiments()
    ###########################################################################
