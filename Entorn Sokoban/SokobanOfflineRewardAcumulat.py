from Sokoban import Environment
from SokobanQLearning import q_learning
from SokobanQLearning import is_terminal_state
from SokobanQLearning import numero_a_paraula
import matplotlib.pyplot as plt
import numpy as np


weights = [1, 6.1]

w_E = weights[1]

policy, V, Q = q_learning(weights)

suma_rewards = list()

suma_rewards.append([0,0])

reward_objectiu = 0

reward_etic = 0

number_of_actions = 0

max_episodes = 5000


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
        
        reward_objectiu += rewards['GOAL_REWARD']
        reward_etic += rewards['IMPACT_REWARD']
        suma_rewards.append([reward_objectiu, reward_etic])
        number_of_actions += 1
    
        print('Agent location', env.agent_location)

        env.visualise_environment()
        
        print('Action:', action_word)
        print('Rewards:', rewards)
        print('Observation', state)
    


    print("\nIs terminal?", env.is_terminal())


suma_rewards = np.array(suma_rewards)



x_0 = suma_rewards[:,0]
x_E = suma_rewards[:,1]

n_of_actions_grafica = range(number_of_actions+1)
number_of_actions = range(number_of_actions)

x_T = []
for i in n_of_actions_grafica:
    x_T.append(x_0[i]+ x_E[i])


plt.title(label = 'Environment: Sokoban ($w_E = $ '+ str(w_E) + ')')
plt.axhline(y=40, color = 'tomato', label = 'Reward individual política ètica')
plt.axhline(y=0, color = 'palegreen', label = 'Reward ètic política ètica')
plt.plot(n_of_actions_grafica, x_0, markersize=10, marker='s',label="Reward objectiu", color='red')
plt.plot(n_of_actions_grafica, x_E, markersize=10, marker='^',label="Reward ètic", color='green')
plt.plot(n_of_actions_grafica, x_T, markersize=10, marker='o',label="Reward total", color='black')
plt.ylabel("Reward acumulat per acció")
plt.xlabel("Número d'accions")
plt.legend(loc='upper left',)
plt.show()

