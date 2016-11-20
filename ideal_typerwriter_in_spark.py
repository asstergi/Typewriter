from __future__ import print_function
 
from pyspark import SparkContext

import copy
import numpy as np
import math
import random 

def parseDatafileLine(datafileLine):
    """ 
    Parse a line of the data file by keeping the 5th element i.e. the actual tweet
    Remove any leading and trailing quotes
    Replace characters that are placed on the same keys e.g. @ with 2
    Create sorted character bigrams and emit (bigram, 1)
    """
    string = datafileLine.split(',', 5)[5]
    
    if string.startswith('"') and string.endswith('"'):
        string = string[1:-1]
    
    string = "".join([replacements.value.get(c, c) for c in string])
    
    string_bigrams = zip(string,string[1:])
    string_bigrams_final = [(''.join(sorted(bigram)).lower(), 1) for bigram in string_bigrams if not any(s in bigram for s in restrictedChars.value)]
    
    return string_bigrams_final

def parseData(filename):
    return sc.textFile(filename, use_unicode=0).flatMap(parseDatafileLine)

def loadData(path):
    raw = parseData(path).cache()
    return raw

    
##################################################
# Simulated Annealing Functions
##################################################
def costFunction(solution):
    """ 
    The objective function of this combinatorial optimization problem is penalizing by a factor of 2 any adjoining typebars times 
    how often their respective characters appear together in the corpus. Typebars that have another typebar separating them, 
    are penalized by a factor of 1 times the how often the respective characters appear together in the corpus. 
    Typebars separated by two or more typebars are not penalized no matter what the frequency of their characters because 
    they cannot get jammed with each other.
    
    The lower, the better.
    """
    
    #The cost of adjacent typebars * 2
    bigrams = [''.join(sorted(bigram)) for bigram in zip(solution,solution[1:])]
    cost = 0
    for i in bigrams:
        if i in bigramCountDict.value:
            cost += bigramCountDict.value[i] * 2
    
    #The cost of typebars separated by another typebar * 1 
    skipped_bigrams = [''.join(sorted(bigram)) for bigram in zip(solution,solution[2:])]
    for i in skipped_bigrams:
        if i in bigramCountDict.value:
            cost += bigramCountDict.value[i] * 1
            
    return cost
    
def swapInList(solution, stepSize):
    """ 
    Randomly swap an element of the solution using the stepSize
    """
    thisSolution = copy.copy(solution)
    solutionLength = len(thisSolution)

    randInt = np.random.randint(low = 0, high = solutionLength) #which element to move
    direction = np.random.randint(low=0, high=1)                #left or right
    
    #If the first or the last item is selected, do not care about the direction
    if randInt == 0:
        thisSolution[stepSize], thisSolution[0] = thisSolution[0], thisSolution[stepSize]
    elif randInt==solutionLength:
        thisSolution[solutionLength - stepSize], thisSolution[solutionLength] = thisSolution[solutionLength], thisSolution[solutionLength - stepSize]
    #if there are is not enough space towards the selected direction, move as far as possible
    elif randInt + stepSize >= solutionLength and direction == 1:
        thisSolution[randInt], thisSolution[solutionLength] = thisSolution[solutionLength], thisSolution[randInt]
    elif randInt - stepSize <= 0 and direction == 0:
        thisSolution[randInt], thisSolution[0] = thisSolution[0], thisSolution[randInt]
    else:
        if direction == 0:
            thisSolution[randInt-stepSize], thisSolution[randInt] = thisSolution[randInt], thisSolution[randInt-stepSize]
        else:
            thisSolution[randInt+stepSize], thisSolution[randInt] = thisSolution[randInt], thisSolution[randInt+stepSize]
            
    return thisSolution

def SAforWorker(solution, randInt, temp, m):
    """ 
    Perform iterative updates on the current temperature
    """
    bestSolution = solution
    bestScore = costFunction(solution)
    for j in range(m):
        new_solution = swapInList(bestSolution, stepSize = randInt)
        DeltaE = costFunction(new_solution) - bestScore
        
        if (DeltaE<0 or (DeltaE>0 and math.exp(-DeltaE/temp)>random.random())):
            bestSolution = new_solution
            bestScore += DeltaE
    
    return (bestSolution, bestScore)

if __name__ == "__main__":

    sc = SparkContext(appName="TypeWriter")
    
    file = 'training.1600000.processed.noemoticon.csv'
    
    #The list of characters that do not affect typewriter typebars e.g. space . Could add more characters here
    restrictedChars = sc.broadcast([' '])
    
    #Dictionary of characters that are placed on the same keys e.g. @ and 2
    replacements = sc.broadcast({')':'0', '!':'1', '@':'2', '#':'3', '$':'4', '%':'5', '^':'6', '&':'7', '*':'8', 
                    '(':'9', '_':'-', '+':'=', '{':'[', '}':']', ':':';', '\'': '"', '|':'\\', '<':',', '>':'.', '?':'/'})
    
    #Load data, count bigrams and broadcast them as a dictionary
    small = loadData(file)
    bigramCount = small.reduceByKey(lambda a,b: a+b)
    bigramCountDict = sc.broadcast(bigramCount.collectAsMap())
    
    ##################################################
    # Simulated Annealing Execution
    ##################################################
    '''
    Here are the steps I followed to parallelize Simulated Annealing on Spark:

    1) Use as initial solution the current typewriter layout.
    2) Randomly generate 'num_workers' different step sizes and distribute them to each worker.
    3) At the current temperature, each worker begins to execute 'nTrialsPerCycle' iterative operations.
    4) At the end of iterations, the master process collects the solution obtained by each worker at current temperature and broadcasts the best one back to the workers.
    5) The master cools the temperature and the process continues till the temperature falls below a specified point.
    6) The best solution ever received by the master, is the final solution.
    '''
    
    #Initial solution is the current keyboard layout
    #The order here is the the actual order of the typebars
    solution =  ['1', 'q', 'a', '2', 'z', 'w', 's', '3' , 'x', 'e', 'd', '4', 'c', 'r', 'f', '5', 'v',
                 't', 'g', '6', 'b', 'y', 'h', '7', 'n', 'u', 'j', '8', 'm', 'i', 'k', '9', ',',
                'o', 'l', '0', '.', 'p', ';', '-', '/', '[', '"', '=', ']', '\\']
                
    
    initialCost = costFunction(solution)
    bestSolution = solution
    globalBestSolution = solution
    bestScore = costFunction(solution)
    globalBestScore = costFunction(solution)
    
    # Number of trials per cycle
    nTrialsPerCycle = 100
    # Initial temperature
    initTemp = 1000
    # Final temperature
    minTemp = 1
    # Fractional reduction for every cycle
    coolingRate  = 0.95
    #num_workers
    num_workers = 20
    
    # Current temperature
    temp = initTemp
    
    i = 0
    while (temp > minTemp):
        if i%30==0:
            print('Cycle: ' + str(i) + ' with Temperature: ' + str(temp) + '. Best score till now: ' + str(globalBestScore))
    
        #Parallelize the current best solution and create a random step size for each worker
        currectSolutionRDD = sc.parallelize(zip([bestSolution]*num_workers, np.random.randint(1, 8, size=num_workers)))
        solutions = currectSolutionRDD.map(lambda (solution, randInt): SAforWorker(solution, randInt, temp, nTrialsPerCycle))
        thisRoundBestSolution = solutions.takeOrdered(1, lambda x: x[1])[0]
    
        #Keep this round's best solution as the starting solution of the next round
        bestSolution = thisRoundBestSolution[0]
        bestScore = thisRoundBestSolution[1]
    
        #Update global best solution
        if thisRoundBestSolution[1] < globalBestScore:
            globalBestScore = thisRoundBestSolution[1]
            globalBestSolution = thisRoundBestSolution[0]
    
        #Lower the temperature for the next cycle
        temp *= coolingRate 
        i += 1
    
    print("The initial solution had a cost equal to " + str(initialCost))
    print("The best solution found has a cost equal to " + str(globalBestScore))
    print("This is the best solution found: " + str([i for i in globalBestSolution]))
    print("The best solution ever found had a cost equal to 594439")
 
    #The initial solution had a cost equal to 7633100
    #The best solution found has a cost equal to 594439 - you don't get the same results after each round
    #   due to the randomization of the Simulated Annealing algorithm
    #This is the best solution found: ['a', '4', '3', 'e', '2', 'q', 'r', 'z', '1', 'c', 'w', 'v', 'x', 's', 'f', '6', 'd', 't', 'g', '5', 'b', 'n', '7', 'h', 'k', 'm', '8', 'j', 'y', ']', '0', 'u', '/', '.', '-', '"', 'p', '\\', ',', 'i', '=', ';', 'l', '[', '9', 'o']