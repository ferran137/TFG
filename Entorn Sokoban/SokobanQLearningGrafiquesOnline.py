from Sokoban import Environment
import numpy as np
import random
import matplotlib.pyplot as plt




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

max_episodes = 5000

def q_learning(weights, alpha = 1.0, gamma = 1.0, epsilon = 0.9, max_episodes = max_episodes):
    
    env = Environment()
    
    
    ### Settings related to Sokovan
    n_objectives = env.NUM_OBJECTIVES
    n_cells = env.NUM_CELLS
    n_actions = env.NUM_ACTIONS
    
    
    ### Settings for Q-learning
    max_steps = 30
    Q = np.zeros([n_cells, n_cells, n_actions, n_objectives])
    
    
    ################################
    ## Settings per les gràfiques ##
    
    for_graphics = list()
    
    ### Algorithm starts here
    
    
    for episode in range(1,max_episodes+1):
        env.env_clean_up()
        env.env_init()
        state = env.env_start()
        initial_state = state
        
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
        
        #####################
        ## DADES GRÀFIQUES ##
        #####################
        
        q = Q[initial_state[0],initial_state[1]].copy()
        sq = scalarised_Qs(env, q, weights)
        a = np.argmax(sq)
        for_graphics.append(q[a])

        ###################
        
    # Output a deterministic optimal policy and its associated Value table
    policy, V = deterministic_optimal_policy_calculator(Q, env, weights)
    
    np_graphics = np.array(for_graphics)
    
    return policy, V, Q, np_graphics


weights = [1,6.1]

w_E = weights[1]

policy, V, Q, np_graphics = q_learning(weights)


print("-------------------")
print("The Learnt Policy has the following Value:")
policy_value = V[1,3]
print("Individual Value V_0 = " + str(round(policy_value[0],2)))
print("Ethical Value (V_N + V_E) = " + str(round(policy_value[1],2)))

## GENEREM LES GRÀFIQUES ##

x_0 = np_graphics[:,0]
x_E = np_graphics[:,1]


def reward_online(llista):
    suma = 0
    for i in range(len(llista)):
        suma += llista[i]
    mitjana = suma/len(llista)
    
    return mitjana

reward_online_objectiu = reward_online(x_0)
print('Reward Online Objectiu = ',reward_online_objectiu)

reward_online_etic = reward_online(x_E)
print('Reward Online Ètic = ',reward_online_etic)


plt.title(label = 'Environment: Sokoban ($w_E = $ '+ str(w_E) + ')')
plt.axhline(y=40, color = 'tomato', label = 'Reward individual política ètica')
plt.axhline(y=0, color = 'palegreen', label = 'Reward ètic política ètica')
plt.plot(range(max_episodes), x_0, linewidth=2, markersize=10, marker='s', markevery=500, label="Objectiu individual $V_0$", color='red')
plt.plot(range(max_episodes), x_E, linewidth=2, markersize=10, marker='^', markevery=500, label="Objectiu ètic $V_N + V_E$", color='green')
plt.ylabel("Suma de rewards al final de l'episodi")
plt.xlabel('Episodi')
plt.legend(loc='center right',fontsize = 'medium')
plt.show()


"""

if __name__ == '__main__':

    env = Environment()
    
    env.env_clean_up()
    
    env.env_init()
    
    env.visualise_environment()

    state = env.env_start()
    print('Observation', state)
    
    while not is_terminal_state(env, state):
        
        action_number = policy[state[0],state[1]]
        
        action_word = numero_a_paraula(action_number)
        
        rewards, state = env.env_step(action_word)
    
        print('Agent location', env.agent_location)

        env.visualise_environment()
        
        print('Action:', action_word)
        print('Rewards:', rewards)
        print('Observation', state)
    


    print("\nIs terminal?", env.is_terminal())

"""


