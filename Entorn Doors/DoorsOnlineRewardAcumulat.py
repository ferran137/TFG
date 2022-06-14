import numpy as np
import matplotlib.pyplot as plt
from DoorsVectorial import Environment
from DoorsVectorial import q_learning
from DoorsVectorial import is_terminal_state
from DoorsVectorial import state_to_row
from DoorsVectorial import num_a_paraula


weights = [1,2.1]

w_E = weights[1]

policy, V, q_table, np_graphics = q_learning(weights)

suma_rewards = list()

suma_rewards.append([0,0])

reward_objectiu = 0

reward_etic = 0

number_of_actions = 0



if __name__ == '__main__':
    
    e = Environment()

    observation = e.env_start()

    while not is_terminal_state(e, e.agent_location):
        
        
        action_number = policy[state_to_row(observation)]
        
        action_word = num_a_paraula(action_number)
    
        rewards, observation = e.env_step(action_word)
        
        reward_objectiu += rewards['GOAL_REWARD']
        reward_etic += rewards['IMPACT_REWARD']
        suma_rewards.append([reward_objectiu, reward_etic])
        number_of_actions += 1
        
 
    print("\nIs terminal?", e.is_terminal())
   

suma_rewards = np.array(suma_rewards)


x_0 = suma_rewards[:,0]
x_E = suma_rewards[:,1]

n_of_actions_grafica = range(number_of_actions+1)
number_of_actions = range(number_of_actions)

x_T = []

for i in n_of_actions_grafica:
    x_T.append(x_0[i]+ x_E[i])


plt.title(label = 'Environment: Doors ($w_E = $ '+ str(w_E) + ')')
plt.axhline(y=43, color = 'tomato', label = 'Reward individual política ètica')
plt.axhline(y=0, color = 'palegreen', label = 'Reward ètic política ètica')
plt.plot(n_of_actions_grafica, x_0, linewidth=2, markersize=10, marker='s',label="Reward objectiu", color='red')
plt.plot(n_of_actions_grafica, x_E, linewidth=2, markersize=10, marker='^',label="Reward ètic", color='green')
plt.plot(n_of_actions_grafica, x_T, markersize=10, marker='o',label="Reward total", color='black')
plt.ylabel("Reward acumulat per acció")
plt.xlabel("Número d'accions")
plt.legend(loc='upper left')
plt.show()

