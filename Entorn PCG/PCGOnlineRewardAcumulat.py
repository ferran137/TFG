import matplotlib.pyplot as plt
from Learning import q_learning
from Environment import Environment

max_episodes = 5000
env = Environment(is_deterministic=True)
w_E = 7.1
weights = [1.0, w_E]

policy, v, q = q_learning(env, weights, max_episodes=max_episodes)


## AGENT ÈTIC 
suma_rewards_0 = list()
rewards_objectiu_0 = 0
rewards_etic_0 = 0
number_of_actions_0 = 0

suma_rewards_1 = list()
rewards_objectiu_1 = 0
rewards_etic_1 = 0
number_of_actions_1 = 0


## AGEN ÈTIC
done = False
env.hard_reset()
state = env.get_state()
initial_state = state.copy() # initial_state = [8, 9, 6]

while not done:
    actions = list()
    actions.append(policy[state[0], state[1], state[2]])
    state, rewards, dones = env.step(actions)
    rewards_0 = rewards[0]
    reward_0_goal = rewards_0[0]
    reward_0_etic = rewards_0[1]
    rewards_objectiu_0 += reward_0_goal
    rewards_etic_0 += reward_0_etic
    suma_rewards_0.append([rewards_objectiu_0, rewards_etic_0])
    number_of_actions_0 += 1
    
    done = dones[0]


## AGENT NO ÈTIC 
done = False
env.hard_reset()
state = env.get_state()
initial_state = state.copy() # initial_state = [8, 9, 6]

while not done:
    actions = list()
    actions.append(policy[state[0], state[1], state[2]])
    state, rewards, dones = env.step(actions)
    rewards_1 = rewards[1]
    reward_1_goal = rewards_1[0]
    reward_1_etic = rewards_1[1]
    rewards_objectiu_1 += reward_1_goal
    rewards_etic_1 += reward_1_etic
    suma_rewards_1.append([rewards_objectiu_1, rewards_etic_1])
    number_of_actions_1 += 1
    
    done = dones[1] 



n_of_actions_0_grafica = range(number_of_actions_0+1)
n_of_actions_1_grafica = range(number_of_actions_1+1)



number_of_actions_0 = range(number_of_actions_0)
x_0_0 = []
x_0_0.append(0)
for i in number_of_actions_0:
    x_0_0.append(suma_rewards_0[i][0])
x_0_E = []
x_0_E.append(0)
for i in number_of_actions_0:
    x_0_E.append(suma_rewards_0[i][1])
x_0_T = []
for i in n_of_actions_0_grafica:
    x_0_T.append(x_0_0[i] + x_0_E[i])


print(x_0_T)


number_of_actions_1 = range(number_of_actions_1)
x_1_0 = []
x_1_0.append(0)
for i in number_of_actions_1:
    x_1_0.append(suma_rewards_1[i][0])
x_1_E = []
x_1_E.append(0)
for i in number_of_actions_1:
    x_1_E.append(suma_rewards_1[i][1])
x_1_T = []
for i in n_of_actions_1_grafica:
    x_1_T.append(x_1_0[i] + x_1_E[i])

print(x_1_T)

print('x_0_0', x_0_0)
print('x_0_E', x_0_E)

print('x_1_0', x_1_0)
print('x_1_E', x_1_E)


plt.title(label = 'Environment: Public Civility Game ($w_E = $ '+ str(w_E) + ')')

# AGENT ÈTIC
plt.plot(n_of_actions_0_grafica , x_0_0, markersize=10, marker='s',label="Rwd objectiu agent ètic ", color='lime')
plt.plot(n_of_actions_0_grafica , x_0_E, markersize=10, marker='s',label="Rwd ètic agent ètic", color='mediumspringgreen')
plt.plot(n_of_actions_0_grafica , x_0_T, markersize=10, marker='s',label="Rwd total agent ètic", color='darkgreen')

# AGENT NO ÈTIC
plt.plot(n_of_actions_1_grafica, x_1_0, markersize=10, marker='^',label="Rwd objectiu agent no ètic", color='orangered')
plt.plot(n_of_actions_1_grafica, x_1_E, markersize=10, marker='^',label="Rwd ètic agent no ètic", color='salmon')
plt.plot(n_of_actions_1_grafica, x_1_T, markersize=10, marker='^',label="Rwd total agent no ètic", color='darkred')


plt.ylabel("Reward acumulat per acció")
plt.xlabel("Número d'accions")
plt.legend(loc='upper left', fontsize = 'small' )
plt.show()








