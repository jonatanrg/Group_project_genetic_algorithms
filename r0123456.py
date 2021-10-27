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
		for gene in population:
		    print(self.getObjective(gene, distanceMatrix))

		nbIter = 0
		while (nbIter < 10000):

			selectedPop = self.selection(population)
			recombinatedPop = self.recombination(selectedPop)
			mutatedPop = self.mutation(recombinatedPop)
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
		subset_size = int(len(population)*percentage)

		for _ in range(subset_size):
		    index = random.randint(0, len(population))
		    fitness = self.getObjective(population[index], distanceMatrix)

            # Eliminate elements with below average fitness
		    if fitness < mean_fitness:
    			del population[index]

		return population

# returns list of children after recombination of parent pairs
	def recombination(self, population):
		newPop = []
		for pair in population:
			newPop.append(self.OX(pair))
			newPop.append(self.POS(pair))
		return newPop

# recombination operators
	def POS(self, parents):
		parent1 = parents[0]
		parent2 = parents[1]
		positions = []
		child = list(-1 for _ in range(len(parent1)))

		for _ in range(random.randint(0,len(parent1) - 1)):
			position = random.randint(0,len(parent1) - 1)
			if position not in positions:
				positions.append(position)

		for i in positions:
			child[i] = parent1[i]

		j = 0
		for i in range(len(parent2)):
			if child[i] == -1:
				while parent2[j] in child:
					j += 1
				child[i] = parent2[j]

		return child

	def OX(self, parents):
		parent1 = parents[0]
		parent2 = parents[1]
		child = list(-1 for _ in range(len(parent1)))

		# partition 1 will take place after the position of partition1
		# partition 2 will take place before the position of partition2
		partition1 = random.randint(1, len(parent1) - 1)
		partition2 = random.randint(1, len(parent1) - 1)

		lowerpartition = min(partition1,partition2)
		higherpartition = max(partition1, partition2)

		# middle part of parent1 is copied
		child[lowerpartition: higherpartition] = parent1[lowerpartition: higherpartition]
		parent2 = parent2[higherpartition:] + parent2[:higherpartition]

		j = higherpartition
		for i in parent2:
			if i not in child:
				child[j] = i
				j += 1
				if j >= len(child):
					j = 0

		return child

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