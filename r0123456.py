import math
import random
import numpy as np
import time

import Reporter

print("Running")


# Modify the class name to match your student number.
class r0123456:
    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.age = 10

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()
        # Your code here.
        popSize = 1000
        population = self.initPopulation(popSize, range(len(distanceMatrix[0])))
        start = time.time()

        nbIter = 0
        while nbIter < 100 and len(population) > 1:
            selectedPop = self.selection(population, distanceMatrix)
            recombinatedPop = self.recombination(selectedPop)
            mutatedPop = self.mutation(recombinatedPop)
            finalPop = self.eliminationFitness(mutatedPop, distanceMatrix)

            allObjectives = [self.getObjective(route[0], distanceMatrix) for route in finalPop]

            bestObjective = min(allObjectives)
            meanObjective = np.mean(allObjectives)
            bestSolution = np.array(finalPop[allObjectives.index(bestObjective)])

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)

            if timeLeft < 0:
                break

            nbIter += 1
            print("Iteration: " , nbIter,
                  "\t Best objective: ", bestObjective,
                  "\t Length of population: " , len(population))

            population = finalPop
            # Update age variable for all routes
            for index in range(len(population)):
                population[index][1] += 1
        # Your code here.
        end = time.time()

        print('Best Solution: ' + str(bestSolution[0]))
        print('Objective function output: ' + str(bestObjective))
        print('Total time elapsed: ' + str(end - start) + ' seconds')
        return 0

    def mutation(self, offspring):
        # Swap mutation. Pick two genes from the population and randomly pick
        # a node in one of the two genes to create a new route.
        # ii = 0
        # newOffspring = [[]]
        # mutated = []
        # while ii < len(offspring):
        #     both_lists = [offspring[ii][0], offspring[random.randint(0, len(offspring) - 1)][0]]
        #     for item in range(len(offspring[ii][0])):
        #         selected_list = random.choice(both_lists)
        #         selected_item = random.choice(selected_list)
        #         both_lists = [[ele for ele in sub if ele != selected_item] for sub in both_lists]
        #         mutated.append(selected_item)
        #     ii += 1
        #     newOffspring.append([mutated, 0])
        #     mutated = []
        # newOffspring = [x for x in newOffspring if x]
        # return newOffspring

        # todo(loris): add inverion mutation
        newOffspring = [[]]
        for i in range(len(offspring)):
            partition1 = random.randint(1, len(offspring[i][0]) - 1)
            partition2 = random.randint(1, len(offspring[i][0]) - 1)
            lowerpartition = min(partition1, partition2)
            higherpartition = max(partition1, partition2)

            toMutate = offspring[i][0]

            mutated = toMutate[:lowerpartition] + \
                      toMutate[lowerpartition:higherpartition][::-1] + \
                      toMutate[higherpartition:]

            newOffspring.append([mutated, 0])

        newOffspring = [x for x in newOffspring if x]
        return newOffspring

    def eliminationFitness(self, population, distanceMatrix):
        # fitness-based elimination
        pop_fitness = np.zeros(len(population))
        for index in range(len(pop_fitness)):
            pop_fitness = self.getObjective(population[index][0], distanceMatrix)

        mean_fitness = np.mean(pop_fitness)

        # Select size of subset population
        percentage = 0.01
        subset_size = int(len(population) * percentage)

        for _ in range(subset_size):
            index = random.randint(0, len(population) - 1)
            fitness = self.getObjective(population[index][0], distanceMatrix)

            # Eliminate elements with below average fitness
            if fitness < mean_fitness:
                del population[index]

        return population

    def eliminationAge(self, population):
        for route in population:
            if route[1] > self.age:
                population.remove(route)

        return population

    # returns list of children after recombination of parent pairs
    def recombination(self, population):
        # create pairs of parents, is the amount of genes in the population is uneven, the last gene is not used
        gene = None
        if len(population) % 2 != 0:
            population = population[:len(population) - 1]
            gene = population[len(population) - 1:]

        population = [[population[i], population[i + 1]] for i in range(len(population)) if i % 2 == 0]

        # for each pair OX and POS operators are executed, childeren are added to newPop
        newPop = []
        for pair in population:
            newPop.append(self.OX(pair))
            newPop.append(self.POS(pair))

        # if the population was uneven, the not used gene is added again
        if gene is not None:
            newPop += gene

        return newPop

    # recombination operators
    # Position based crossover https://www.researchgate.net/figure/Position-based-crossover-POS_fig4_226665831
    # Positions of 1 parent are kept and the rest is filled with the other parent
    def POS(self, parents):

        # only the 'route part' of the genes are used
        parent1 = parents[0][0]
        parent2 = parents[1][0]
        positions = []

        # child is initialized as a list with -1 values
        child = list(-1 for _ in range(len(parent1)))

        # a random amount of positions is generated 
        for _ in range(random.randint(0, len(parent1) - 1)):
            position = random.randint(0, len(parent1) - 1)
            if position not in positions:
                positions.append(position)

        # the random generated positions of the cities from parent1 are used
        for i in positions:
            child[i] = parent1[i]

        # the the rest of the child is filled with cities from parent2
        j = 0
        for i in range(len(parent2)):
            if child[i] == -1:
                while parent2[j] in child:
                    j += 1
                child[i] = parent2[j]

        # new child is returned with age 0
        return [child, 0]

    # Ordered crossover https://www.researchgate.net/figure/OX-Crossover-Operator-Default-Implementation_fig3_299533271
    # Parents are divided into 3 partitions
    # the middle partition of 1 parent is kept
    # the rest is filled with the same order as the other parent, starting at the third partition
    def OX(self, parents):

        # only the 'route part' of the genes are used
        parent1 = parents[0][0]
        parent2 = parents[1][0]

        # child is initialized as a list with -1 values
        child = list(-1 for _ in range(len(parent1)))

        # random positions of the partitions are chosen then the highes and lowest are determined as first and last partition
        partition1 = random.randint(1, len(parent1) - 1)
        partition2 = random.randint(1, len(parent1) - 1)
        lowerpartition = min(partition1, partition2)
        higherpartition = max(partition1, partition2)

        # middle part of parent1 is copied into child
        child[lowerpartition: higherpartition] = parent1[lowerpartition: higherpartition]

        # parent2 is shifted so the starting point is the third partition
        parent2 = parent2[higherpartition:] + parent2[:higherpartition]

        # child is filled with the positions of parent 2 starting at the third partition
        j = higherpartition
        for i in parent2:
            if i not in child:
                child[j] = i
                j += 1
                if j >= len(child):
                    j = 0

        # new child is returned with age 0
        return [child, 0]

    def selection(self, population, distanceMatrix):
        """
        Takes the population of the new iteration,
        applying Ranking selection (fitness-based),
        with linear decay.
        s: selection pressure parameter. It is virtually fixed to 1 in this implementation.
        """
        allObjectives = []
        routes = []

        # store all the current solutions (routes) and the respective Objective values
        for route in population:
            routes.append(route)
            allObjectives.append(self.getObjective(route[0], distanceMatrix))

        integerList = [i for i in range(1, len(allObjectives) + 1)]

        # sort the routes based on the ascending objective value
        sortedIndices = sorted(range(len(allObjectives)), key=lambda k: allObjectives[k])
        orderedRoutes = [routes[i] for i in sortedIndices]

        # Ranking: linear decay implementation (we reverse the array to make the shortest routes more likely to be selected)
        indices_sum = sum(integerList)
        probabilities = [i / indices_sum for i in integerList]
        probabilities.reverse()

        # ATTENTION: you should change the variable "population" if you want only select a subset and not the whole input
        newPopulation = [orderedRoutes[i] for i in np.random.choice(range(len(orderedRoutes)), len(population), replace=1, p=probabilities)]

        return newPopulation

    def generateRoute(self, cityList):
        route = random.sample(cityList, len(cityList))
        return route

    def initPopulation(self, size, cityList):
        population = list()

        for index in range(size):
            population.append([self.generateRoute(cityList), index])
        return population

    def getObjective(self, route, distanceMatrix):
        totalLength = 0
        for index in range(len(route)):
            city1 = route[index]
            try:
                city2 = route[index + 1]
            except IndexError:
                city2 = route[0]

            totalLength += self.getLength(city1, city2, distanceMatrix)

        return totalLength

    def getLength(self, city1, city2, distanceMatrix):
        return distanceMatrix[city1][city2]


if __name__ == '__main__':
    test = r0123456()
    test.optimize('tour29.csv')
