import random 
from math import pow

class wing:
    def __init__(self, ABCD = [random.uniform(0, 63) for i in range(4)]):
        self.ABCD = ABCD

    def get_fitness(self):
        return pow(self.ABCD[0] - self.ABCD[1], 2) + pow(self.ABCD[2] - self.ABCD[3], 2) - pow(self.ABCD[0] - 30, 3) - pow(self.ABCD[2] - 40, 3)

    def __lt__(self, o):
        return self.ABCD < o.ABCD

def evolve( population, retain =0.2 ,random_select =0.05 , mutate =0.1):
    """
    Function for evolving a population , that is , creating
    offspring(next generation population ) from combining
   ( crossover ) the fittest individuals of the current
    population
    : param population : the current population
    : param target : the value that we are aiming for
    : param retain : the portion of the population that we
    allow to spawn offspring
    : param random_select : the portion of individuals that
    are selected at random , not based on their score
    : param mutate : the amount of random change we apply to
    new offspring
    : return : next generation population
    """
    graded = [( x.get_fitness(), x) for x in population ]
    #print(graded)
    graded = [ x[1] for x in sorted( graded, reverse=True ) ]
    retain_length = int(len(graded) * retain )
    parents = graded [:retain_length]

    # randomly add other individuals to promote genetic
    # diversity
    for individual in graded [retain_length:]:
        if random_select > random.random():
            parents.append( individual )
    
    # crossover parents to create offspring
    desired_length = len( population ) - len( parents )
    children = []
    while len( children ) < desired_length :
        male = random.randint(0, len( parents) - 1)
        female = random.randint(0, len( parents) - 1)
        if male != female :
            male = parents[male]
            female = parents[female]
            half = int(len(male.ABCD) / 2)
            child = wing(male.ABCD[:half] + female.ABCD[half:])
            children.append( child )

    for individual in children :
        if mutate > random.random():
            pos_to_mutate = random.randint(0, len(individual.ABCD) - 1)
            # this mutation is not ideal , because it
            # restricts the range of possible values ,
            # but the function is unaware of the min/max
            # values used to create the individuals
            individual.ABCD[ pos_to_mutate ] = random.uniform(min( individual.ABCD ), max( individual.ABCD ))
    
    parents.extend( children )
    return parents

popu = [wing() for i in range(100000)]

count = 0
last_fitness = 0


while (1):
    popu = evolve(popu)
    if popu[0].get_fitness() == last_fitness:
        count += 1
        if (count == 10):
            print('local maximum fitnes reached: ', last_fitness)
            exit()
    else:
        count = 0
    last_fitness = popu[0].get_fitness()
