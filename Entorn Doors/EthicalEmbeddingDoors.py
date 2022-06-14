from DoorsVectorial import Environment
from DoorsVectorial import q_learning


def state_to_row(observation):
    row_f=0
    for i in range(0,observation[0]):
        row_f +=4
        
    if observation[1] == False and observation[2] == False:
        row_f += 0
    if observation[1] == True and observation[2] == False:
        row_f += 1
    if observation[1] == True and observation[2] == True:
        row_f += 2
    if observation[1] == False and observation[2] == True:
        row_f += 3
    return row_f

number_of_possible_states = 4*14

env = Environment()



def is_best_ethical(possible_v, env, algorithm):
    """
    Method to assert that the ethical policy that we have computed is indeed the best-ethical one.
    To do it we check if the ethical value of "possible_v" is the same as the ethical value of the
    ethical value function computed with weights [0, 1] (i.e., only giving importance to the ethical dimension).
    :param possible_v: the value function of the policy that we hope that corresponds with the best-ethical one
    :param env: the Multi-Objective MDP
    :param algorithm: any single-objective RL algorithm (for example, q-learning)
    :return: True if the policy is indeed best-ethical and False otherwise.
    """
    if algorithm == 'q_learning':
        _, ethical_value_function, _, _ = q_learning([0,1])

    for state in range(number_of_possible_states):
        max_ethical = ethical_value_function[state][1]
        
        if possible_v[state][1] < max_ethical:
            return False
        
    return True




def SolveSOMDP(env, weight_vector, algorithm):
    """
    Generic method to call an algorithm from another Python file that solves an MDP.
    The method expects that the algorithm returns the optimal value function.
    :param env: a Multi-Objective MDP
    :param weight_vector: the weight vector to scalarise the Multi-Objective MDP into a Single-Objective MDP
    :param algorithm: the algorithm to be applied to solve the MDP (for example, q-learning)
    :return: the optimal value function of the scalarised MDP
    """

    #TODO: Currently only implemented for q-learning, you can modify it to use any algorithm you want.
    if algorithm == 'q_learning':
        _, value_function, _, _= q_learning(weight_vector)
    else:
        return "Fatal error, algorithm not found."

    return value_function

def ethical_weight_finder_per_state(hull):
    """
    Ethical weight computation for a single state. Considers the points in the hull of a given state and returns
    the ethical weight that guarantees optimality for the ethical point of the hull

    :param hull: set of two 2-D points, coded as a numpy array
    :return: the etical weight w, a positive real number
    """
    
    w = 0.0
    second_best_ethical = hull[0]
    best_ethical = hull[1]
    
    
    individual_delta = second_best_ethical[0] - best_ethical[0]
    ethical_delta = best_ethical[1] - second_best_ethical[1]

    if ethical_delta > 0.0001:
        w = individual_delta/ethical_delta

    return w
# No cal tocar res


def ethical_weight_finder(hull, epsilon, only_initial_states=True):
    """
    Computes for each (initial) state the ethical weight that guarantees
    that the best-ethical policy has more scalarised value than the unethical one..

    :param hull: the convex-hull-value function storing a partial convex hull for each state. The states are adapted
    to the public civility game.
    :param epsilon: the epsilon positive number considered in order to guarantee ethical optimality (it does not matter
    its value as long as it is greater than 0).
    :return: the desired ethical weight
    """
    ""

    w = 0.0

    if only_initial_states:
        initial_state = [0, False, False]
        hull_unethical = hull[0][state_to_row(initial_state)]
        hull_ethical = hull[1][state_to_row(initial_state)]
        print('hull_unethical =', hull_unethical)
        print('hull_ethical =', hull_ethical)
        
        w = max(w, ethical_weight_finder_per_state([hull_unethical, hull_ethical]))
    
    else:
        for state in range(number_of_possible_states):
            hull_unethical = hull[0][state]
            hull_ethical = hull[1][state]
            w = max(w, ethical_weight_finder_per_state([hull_unethical, hull_ethical])) 
     
    return w + epsilon



def new_Ethical_Embedding(env, epsilon):
    """
    Computes the ethical weight using Optimistic Linear Support.
    :param env: the MOMDP
    :return: the ethical weight
    """

    delta = 0.00000000000000000000001
    v_ethical = SolveSOMDP(env, weight_vector=[delta, 1-delta], algorithm="q_learning")
    
    
    v_ethical_0 = SolveSOMDP(env, weight_vector=[0, 1], algorithm="q_learning")
    
    
    print('v_ethical_0 = ', v_ethical_0[0])
    print('v_ethical amb delta = ',v_ethical[0])
    
    print("Best ethical policy computed.")
    assert delta > 0.0, "The variable delta needs to be strictly positive."
    assert is_best_ethical(v_ethical, env, algorithm="q_learning"), "If this gives an error you need to set a smaller value for delta."

    v_unethical = SolveSOMDP(env, weight_vector=[1.0, 0.0], algorithm="q_learning")

    print("Unethical policy computed.")
    hull = [v_unethical, v_ethical]
    
    
    new_ethical_weight = ethical_weight_finder(hull, epsilon=epsilon)


    while new_ethical_weight > epsilon:

        ethical_weight = new_ethical_weight
        
        print()
        print("So far the ethical weight is: ", ethical_weight)
        print("The values of the different policies are:", v_unethical[0], v_ethical[0])
        print("---Process will finish when the two values are identical----")
        print()

        v_unethical = SolveSOMDP(env, weight_vector=[1.0, ethical_weight], algorithm="q_learning")
        hull[0] = v_unethical
        new_ethical_weight = ethical_weight_finder(hull, epsilon=epsilon)
        print(ethical_weight)

    return ethical_weight



if __name__ == "__main__":

    env = Environment()
    epsilon = 0.1

    w_E = new_Ethical_Embedding(env, epsilon)

    print("The found ethical weight is: ", w_E)
    
    
    
    
 