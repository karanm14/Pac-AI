# Pac-AI
Created a neural network evolved using genetic algorithm to play the game of Pac-Man

# Download the zip file and run python pacman.py to train.

# Slides
https://docs.google.com/presentation/d/1BzKfS02BqEoEVkfG0_SRdUYeMOTwYg3I2t5-0--q6sY/edit?usp=sharing


Base game of Pac Man
# By David Reilly, Modified by Andy Sommerville


# Our team created an AI based player. Rather than seeking inputs from user, it plays on its own

# cs4701final
PacAI

### Articles about Genetic Algorithm
<a href="https://medium.com/analytics-vidhya/understanding-genetic-algorithms-in-the-artificial-intelligence-spectrum-7021b7cc25e7">Understanding Genetic Algorithms in the Artificial Intelligence Spectrum</a>

### Inputs
- Immediate walls (R L U D)
- Distance from ghost
- Ghost going towards you?
[ WALL_R WALL_L WALL_U WALL_D | GDIST_1 GDIST_2 GDIST_3 GDIST_4 | GMODE_1 GMODE_2 GMODE_3 GMODE_4 | CLOSESTFOOD_X CLOSESTFOOD_Y | CLOSESTENG_X CLOSESTENG_Y ]

### Fitness Function 
We need to minimize the fitness function = 1/(elapsedTime+score). 


