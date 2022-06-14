The files implement in Python3 the Public Civility Game.

Required libraries:

* SciPy 1.4 or a higher version
* NumPy 1.14 or a higher version
* PyGame 2.0 or a higher version (only necessary for animated visualisation of the learnt policies)

A brief summary of the files provided:

- Environment.py has the logic of the Public Civility Game environment.
- ItemAndAgent.py has the logic of the agents and the piece of garbage.
- ValuesNorms.py saves the ethical knowledge that we want the agents to learn.


- Learning.py implements the q-learning algorithm. This is the file that you need to complete.
- Graphics.py provides visual results of the policy learnt by the agent.

If the q-learning algorithm was correctly written, you should see that the q-values for the individual and
the ethical dimensions converge to their expected values (marked with blue horizontal lines in 0.59 and 0.24).