from Sokoban import Environment
import numpy as np
import random




def scalarisation_function(values, weights):

    f = 0
    
    for objective in range(len(values)):
        
        f += weights[objective]*values[objective]

    return f


    
def scalarised_Qs(env, Q_state, weights):

    scalarised_Q = np.zeros(len(env.all_actions))
    for action in range(len(Q_state)):
        
        scalarised_Q[action] = scalarisation_function(Q_state[action], weights)

    return scalarised_Q



    
def choose_action_epsilon_greedy(st, eps, q_table, env, weights):
    
    random_value = random.uniform(0,1)
    
    if random_value < eps:
        return np.random.randint(env.NUM_ACTIONS)
    
    else:
        return np.argmax(scalarised_Qs(env, q_table[st[0], st[1]], weights))
    


def update_q_table(q_table, env, weights, alpha, gamma, action, state, new_state, reward):
    

    best_action = np.argmax(scalarised_Qs(env, q_table[new_state[0], new_state[1]], weights))
    
    q_table[state[0], state[1], action] += alpha * ( reward + gamma*( q_table[new_state[0], new_state[1], best_action]) - q_table[state[0], state[1], action]) 
         
    return q_table[state[0], state[1], action]

        
def deterministic_optimal_policy_calculator(Q, env, weights):
    policy = np.zeros([env.NUM_CELLS, env.NUM_CELLS])
    V = np.zeros([env.NUM_CELLS, env.NUM_CELLS, env.NUM_OBJECTIVES])

    for cell_L in range(env.NUM_CELLS):
        for cell_R in range(env.NUM_CELLS):
            if cell_L != cell_R:

                best_action = np.argmax(scalarised_Qs(env, Q[cell_L,cell_R], weights))
                                        
                policy[cell_L, cell_R] = best_action
                V[cell_L, cell_R] = Q[cell_L, cell_R, best_action]
    return policy, V


def is_terminal_state(env,state):
    if state[0] == env.AGENT_GOAL:
        return True
    else:
        return False


def numero_a_paraula(action_number):
    if action_number == 0:
        return 'up'
    if action_number == 1:
        return 'right'
    if action_number == 2:
        return 'down'
    if action_number == 3:
        return 'left'


######################################################
######################################################
######################################################



def q_learning(weights, alpha = 1.0, gamma = 1.0, epsilon = 0.9, max_episodes = 10000):
    
    env = Environment()
    
    
    ### Settings related to Sokovan
    n_objectives = env.NUM_OBJECTIVES
    n_cells = env.NUM_CELLS
    n_actions = env.NUM_ACTIONS
    
    
    ### Settings for Q-learning
    max_steps = 30
    Q = np.zeros([n_cells, n_cells, n_actions, n_objectives])
    
    ### Algorithm starts here    
    for episode in range(1,max_episodes+1):
        env.env_clean_up()
        env.env_init()
        state = env.env_start()
        
        step_count = 0
        
        while not is_terminal_state(env,state) and step_count < max_steps:
            
            step_count += 1
            
            action_done_numero = choose_action_epsilon_greedy(state, epsilon, Q, env, weights)
            
            action_done_paraula = numero_a_paraula(action_done_numero)
            
            rewards, new_state = env.env_step(action_done_paraula)
            
            
            vec_rewards = np.array([0,0])    
            vec_rewards[0] = rewards['GOAL_REWARD']
            vec_rewards[1] = rewards['IMPACT_REWARD']
            
            
            new_q_value = update_q_table(Q, env, weights, alpha, gamma, action_done_numero, state, new_state, vec_rewards)
            
            
            if is_terminal_state(env, state):
                Q[state[0],state[1],action_done_numero] = vec_rewards
            
            else:
                Q[state[0],state[1],action_done_numero] = new_q_value
            
            
            state = new_state
        
        #print('Episodi ', episode,' finalitzat')
    
    # Output a deterministic optimal policy and its associated Value table
    policy, V = deterministic_optimal_policy_calculator(Q, env, weights)
            
    return policy, V, Q


    