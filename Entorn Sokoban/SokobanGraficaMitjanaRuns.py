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


def reward_online(llista):  # Fa el càclul de la mitjana
    suma = 0
    for i in range(len(llista)):
        suma += llista[i]
    mitjana = suma/len(llista)
    
    return mitjana

weights = [1,6.1]

w_E = weights[1]

n_runs = 20

llista_runs_0 = []
llista_runs_E = []


for run in range(n_runs):
    _, _, _, np_graphics = q_learning(weights)
    llista_runs_0.append(np_graphics[:,0])
    llista_runs_E.append(np_graphics[:,1])

mitjana_runs_0 = []
mitjana_runs_E = []

for episode in range(max_episodes):
    suma_0 = 0
    suma_E = 0
    for run in range(n_runs):
        suma_0 += llista_runs_0[run][episode]
        suma_E += llista_runs_E[run][episode]
    mitjana_runs_0.append(suma_0/n_runs)
    mitjana_runs_E.append(suma_E/n_runs)


mitjana_total_0 = reward_online(mitjana_runs_0)
mitjana_total_E = reward_online(mitjana_runs_E)

print('Mitjana total rw0 = ', mitjana_total_0)
print('Mitjana total rwE = ', mitjana_total_E)


def variance(llista):
    suma = 0
    m = reward_online(llista)
    for i in range(len(llista)):
        suma += (llista[i]-m)*(llista[i]-m)
    var = suma/len(llista)
    return var

variance_0 = variance(mitjana_runs_0)
variance_E = variance(mitjana_runs_E)

print('Variança_0 = ', variance_0)
print('Variança_E = ', variance_E)

    
## GENEREM LES GRÀFIQUES ##



plt.title(label = 'Environment: Sokoban ($w_E = $ '+ str(w_E) + ')')
plt.axhline(y=40, color = 'tomato', label = 'Reward individual política ètica')
plt.axhline(y=0, color = 'palegreen', label = 'Reward ètic política ètica')
plt.plot(range(max_episodes), mitjana_runs_0, linewidth=2, markersize=10, marker='s', markevery=500, label="Mitjana 20 runs objectiu individual $V_0$", color='red')
plt.plot(range(max_episodes), mitjana_runs_E, linewidth=2, markersize=10, marker='^', markevery=500, label="Mitjana 20 runs objectiu ètic $V_N + V_E$", color='green')
plt.ylabel("Suma de rewards al final de l'episodi")
plt.xlabel('Episodi')
plt.legend(loc='center right',fontsize = 'medium')
plt.show()




