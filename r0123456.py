import math
import random
import numpy as np

import Reporter


# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        popSize = 40
        population = self.initPopulation(popSize, range(len(distanceMatrix[0])))

        nbIter = 0
        while (nbIter < 10000):

            selectedPop = self.selection(population)
            recombinatedPop = self.recombination(selectedPop)
            mutatedPop = self.mutation(population)
            finalPop = self.elimination(mutatedPop)

            allObjectives = [self.getObjective(route, distanceMatrix) for route in finalPop]

            bestObjective = min(allObjectives)
            meanObjective = np.mean(allObjectives)
            bestSolution = finalPop[allObjectives.index(bestObjective)]

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

            nbIter += 1
        # Your code here.
        return 0

    def mutation(self, offspring):
        ii = 0
        newOffspring = [[]]
        mutated = []
        while ii < len(offspring):
            both_lists = [offspring[ii], offspring[random.randint(0, len(offspring)-1)]]
            for item in range(len(offspring[ii])):
                selected_list = random.choice(both_lists)
                selected_item = random.choice(selected_list)
                both_lists = [[ele for ele in sub if ele != selected_item] for sub in both_lists]
                mutated.append(selected_item)
                ii =+ 1
            newOffspring.append(mutated)
        newOffspring = [x for x in newOffspring if x]
        return newOffspring


    def elimination(self, population, distanceMatrix):
        # fitness-based elimination
        pop_fitness = np.zeros(len(population))
        for i in range(pop_fitness):
            pop_fitness[i] = self.getObjective(population[i], distanceMatrix)

        mean_fitness = np.mean(pop_fitness)

        # Select size of subset population
        percentage = 0.6
        subset_size = int(len(population) * percentage)

        for _ in range(subset_size):
            index = random.randint(0, len(population))
            fitness = self.getObjective(population[index], distanceMatrix)

            # Eliminate elements with below average fitness
            if fitness < mean_fitness:
                del population[index]

        return population


    def recombination(population):
        return None


    def selection(population):
        return None


    def generateRoute(self, cityList):
        route = random.sample(cityList, len(cityList))
        return route


    def initPopulation(self, size, cityList):
        population = list()

        for _ in range(size):
            population.append(self.generateRoute(cityList))
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
