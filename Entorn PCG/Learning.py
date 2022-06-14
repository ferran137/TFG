import numpy as np
import time
import threading
from Environment import Environment


RIGHT = 0
UP = 1
LEFT = 2

def translate_action(action):
    """
    Specific to the public civility game environmment, translates what each action number means

    :param action: int number identifying the action
    :return: string with the name of the action
    """

    part_1 = ""
    part_2 = ""

    if action < 3:
        part_1 = "MOVE "
    else:
        part_1 = "PUSH GARBAGE "

    if action % 3 == 0:
        part_2 = "RIGHT"
    elif action % 3 == 1:
        part_2 = "FORWARD"
    else:
        part_2 = "LEFT"

    action_name = part_1 + part_2
    return action_name


def scalarisation_function(values, w):
    """
    Scalarises the value of a state using a linear scalarisation function

    :param values: the different components V_0(s), ..., V_n(s) of the value of the state
    :param w:  the weight vector of the scalarisation function
    :return:  V(s), the scalarised value of the state
    """

    f = 0
    for objective in range(len(values)):
        f += w[objective]*values[objective]

    return f


def scalarised_Qs(env, Q_state, w):
    """
    Scalarises the value of each Q(s,a) for a given state using a linear scalarisation function

    :param Q_state: the different Q(s,a) for the state s, each with several components
    :param w: the weight vector of the scalarisation function
    :return: the scalarised value of each Q(s,a)
    """

    scalarised_Q = np.zeros(len(env.all_actions))
    for action in range(len(Q_state)):
        scalarised_Q[action] = scalarisation_function(Q_state[action], w)

    return scalarised_Q


def Q_function_calculator(env, state, V, discount_factor):
    """

    Calculates the value of applying each action to a given state. Heavily adapted to the public civility game

    :param env: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: the value obtained for each action
    """

    Q_state = np.zeros([len(env.all_actions), len(V[0,0,0])])
    for action in env.all_actions:
        state_translated = env.translate_state(state[0], state[1], state[2])
        env.hard_reset(state_translated[0], state_translated[1], state_translated[2])
        next_state, rewards, _ = env.step([action])
        for objective in range(len(rewards)):
            Q_state[action, objective] = rewards[objective] + discount_factor * V[next_state[0], next_state[1], next_state[2], objective]
    return Q_state


def deterministic_optimal_policy_calculator(Q, env, weights):
    """
    Create a deterministic policy using the optimal Q-value function

    :param Q: optimal Q-function that has the optimal Q-value for each state-action pair (s,a)
    :param env: the environment, needed to know the number of states (adapted to the public civility game)
    :param weights: weight vector, to know how to scalarise the Q-values in order to select the optimals
    :return: a policy that for each state returns an optimal action
    """
    
    policy = np.zeros([12, 12, 12])
    V = np.zeros([12, 12, 12, 2])

    for cell_L in range(12):
        for cell_R in range(12):
            for cell_G in range(12):
                if cell_L != cell_R:

                    best_action = np.argmax(scalarised_Qs(env, Q[cell_L,cell_R,cell_G], weights))
                                        
                    policy[cell_L, cell_R, cell_G] = best_action
                    V[cell_L, cell_R, cell_G] = Q[cell_L, cell_R, cell_G, best_action]
    return policy, V


def choose_action(st, eps, q_table, env, weights):
    """

    :param st: the current state in the environment
    :param eps: the epsilon value
    :param q_table:  q_table or q_function the algorithm is following
    :return:  the most optimal action for the current state or a random action
    """

    NB_ACTIONS = 6
    
    if np.random.random() <= eps:
        return np.random.randint(NB_ACTIONS)
    
    else:
        return np.argmax(scalarised_Qs(env, q_table[st[0], st[1], st[2]], weights))
    
    




def update_q_table(q_table, env, weights, alpha, gamma, action, state, new_state, reward):

    best_action = np.argmax(scalarised_Qs(env, q_table[new_state[0], new_state[1], new_state[2]], weights))
    
    q_table[state[0], state[1], state[2], action] += alpha * ( np.array(reward) + gamma*( q_table[new_state[0], new_state[1], new_state[2],best_action]) - q_table[state[0], state[1], state[2], action])      
                                                            
        
        
        
            
    
    


def q_learning(env, weights, alpha=0.8, gamma=0.7, max_episodes=5000):
    """
    Q-Learning Algorithm as defined in Sutton and Barto's 'Reinforcement Learning: An Introduction' Section 6.5,
    (1998).

    It has been adapted to the particularities of the public civility game, a deterministic environment, and also
     adapted to a MOMDP environment, having a reward function with several components (but assuming the linear scalarisation
    function is known).

    :param env: the environment encoding the (MO)MDP
    :param weights: the weight vector of the known linear scalarisation function
    :param alpha: the learning rate of the algorithm, can be set at discretion
    :param gamma: discount factor of the (MO)MPD, can be set at discretion (notice that this will change the Q-values)
    :return: the learnt policy and its associated state-value (V) and state-action-value (Q) functions
    """

    ### Settings related to the public civility game itself
    n_objectives = 2
    n_actions = 6
    n_cells = 12
    
    ### Settings of Q-learning
    max_steps = 20
    epsilon = 0.999
    eps_reduc = epsilon/max_episodes
    alpha_reduc = 0.5*alpha/max_episodes
    
    print('eps_reduc= ',eps_reduc)
    print('alpha_reduc = ',alpha_reduc )
    print('gamma', gamma)

    Q = np.zeros([n_cells, n_cells, n_cells, n_actions, n_objectives])

    ### Settings for visualising the results
    for_graphics = list()


    ### Algorithm starts here
    for episode in range(1, max_episodes + 1):
        done = False
        env.hard_reset()
        state = env.get_state()
        initial_state = state.copy() # initial_state = [8, 9, 6]

        if episode % 100 == 0:
            print("Episode : ", episode)

        step_count = 0

        alpha -= alpha_reduc
        epsilon -= eps_reduc

        while not done and step_count < max_steps:

            step_count += 1
            actions = list()

            actions.append(choose_action(state, epsilon, Q, env, weights)) ## the action of the learning agent
            actions.append(1) ## the action of the other agent, who has a fixed policy
            
            
            new_state, rewards, dones = env.step(actions)
            reward = rewards[0]
            update_q_table(Q, env, weights, alpha, gamma, actions[0], state, new_state, reward)

            state = new_state
            done = dones[0]


        ### For visualising results later
        q = Q[initial_state[0], initial_state[1], initial_state[2]].copy()
        sq = scalarised_Qs(env, q, weights)
        a = np.argmax(sq)
        #print(q[a])
        for_graphics.append(q[a])


    # Output a deterministic optimal policy and its associated Value table
    policy, V = deterministic_optimal_policy_calculator(Q, env, weights)

    np_graphics = np.array(for_graphics)
    np.save('graphics.npy', np_graphics)

    return policy, V, Q


def example_execution(env, policy, render=False):
    """

    Simulation of the environment without learning. The L Agent moves according to the parameter policy provided.

    :param env: the environment encoding the (MO)MDP
    :param policy: a function S -> A assigning to each state the corresponding recommended action
    :return:
    """

    max_timesteps = 20

    for i in range(1):
        timesteps = 0
        env.hard_reset()

        state = env.get_state()
        done = False

        env.set_stats(i % 2, 99, 99, 99, 99)
        if render:
            if not env.drawing_paused():
                time.sleep(0.5)
                env.update_window()

        while (timesteps < max_timesteps) and (not done):
            timesteps += 1

            actions = list()
            actions.append(policy[state[0], state[1], state[2]])

            state, rewards, dones = env.step(actions)


            done = dones[0]  # R Agent does not interfere

            if render:
                if not env.drawing_paused():
                    time.sleep(0.5)
                    env.update_window()


class QLearner:
    """
    A Wrapper for the Q-learning method, which uses multithreading
    in order to handle the game rendering.
    """

    def __init__(self, environment, policy, drawing=False):

        threading.Thread(target=example_execution, args=(environment, policy, drawing,)).start()
        if drawing:
            environment.render('Evaluating')

"""
max_episodes = 5000
env = Environment(is_deterministic=True)
w_E = 7.1
weights = [1.0, w_E]

policy, v, q = q_learning(env, weights, max_episodes=max_episodes)




if __name__ == "__main__":

    env = Environment(is_deterministic=True)
    w_E = 7.1
    max_episodes = 5000

    print("-------------------")
    print("L(earning) Agent will learn now using Q-Learning in the Public Civility Game.")
    print("The Ethical Weight of the Scalarisation Function is set to W_E: " + str(w_E))
    print("-------------------")
    print("Learning Process started. Will finish when Episode: " + str(max_episodes))
    weights = [1.0, w_E]

    policy, v, q = q_learning(env, weights, max_episodes=max_episodes)
    

    print("-------------------")
    print("The Learnt Policy has the following Value:")
    policy_value = v[8,9,6]
    print("Individual Value V_0 = " + str(round(policy_value[0],2)))
    print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))

    env = Environment(is_deterministic=True)
    QLearner(env, policy, drawing=True)

"""
    



