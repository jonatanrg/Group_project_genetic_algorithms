from numpy.random.mtrand import pareto
from r0123456 import r0123456
import numpy as np

elim_age_params = np.arange(10, 20, 2).tolist() # 5 parameters
elim_per_params = np.arange(0.5, 1, 0.25).tolist() # 2 parameters
sel_per_params = np.arange(0.5, 1, 0.25).tolist() # 2 parameters
max_iter_params = [10] # 1 parameters
operator_params = ['both'] # 1 parameters

allParams = list()
allValues = list()

# 225 times the optimize function gets executed 
for elimAge in elim_age_params:
    for elimPercentage in elim_per_params:
        for selPercentage in sel_per_params:
            for maxIterSameObj in max_iter_params:
                for recombOperator in operator_params:
                    sumObj = 0
                    sumTime = 0

                    parameters= {
                        'elimAge': elimAge,
                        'elimPercentage': elimPercentage,
                        'selPercentage': selPercentage,
                        'maxIterSameObj': maxIterSameObj,
                        'recombOperator': recombOperator
                    }

                    for _ in range(10):
                        algo = r0123456(parameters= parameters)
                        outcomes = algo.optimize('tour29.csv')

                        sumObj += outcomes['bestObjective']
                        sumTime += outcomes['timeElapsed']
                    allValues.append([sumObj/10, sumTime/10])
                    allParams.append([parameters[key] for key in parameters.keys()])


np.savetxt('values.csv', np.asarray(allValues), delimiter=',')
np.savetxt('parameters.csv', np.asarray(allParams), delimiter=',')






