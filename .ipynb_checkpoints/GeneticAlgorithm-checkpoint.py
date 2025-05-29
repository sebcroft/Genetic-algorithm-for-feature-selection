# -*- coding: utf-8 -*-
"""
Created on Thu May 29 18:07:52 2025

@author: sebas
"""






# Import required libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error





class GeneticAlgorithm:
    
    """
    Genetic algorithm: 

    This GA will peform a feature selection, designed to optimise the number of features and pick the optimal feature identities.

    Has the option to visualise the fitness vs generation.

    Aimed to deal with a large search space.

    Maximises the fitness

    """
    def __init__(self, 
                 max_gen,
                 initial_ones,
                 selection_type,
                 crossover_type, 
                 mutation_rate_per_pop,
                 mutation_rate_per_chromosome):
        
        
        self.max_gen = max_gen
        self.initial_ones = initial_ones
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_rate_per_pop = mutation_rate_per_pop
        self.mutation_rate_per_chromosome = mutation_rate_per_chromosome
        
        # variables to be fit
        self.best_features=None
        self.best_chromosome=None
        
        # iterables
        self.generation = []
        self.best_fitness = []
        self.best_matching_chromosome=[]

    # initialisation of population

    def initilization_of_population(self, size, n_feat, sub_feat):
            '''This function will initilize the population of chromosomes - creating some initial cond,
            as well as define the complexity of the problem. Explicitly, it defines the number of regressions
            performed per generation

            args:
            size - size of pop
            n_feat - number of genes (i.e. numb of 1's and 0's)
            sub_feat - number of desired 1's

            returns:
            population - a list of 1D np arrays containing each cromosome/solution randomly generated
            '''
            population=[]
            for i in range(size):
                chromosome = np.zeros(n_feat)
                chromosome[:int(sub_feat)]=1
                np.random.shuffle(chromosome)
                population.append(chromosome)
            return np.array(population)

    # Fitness function
    #############################################################################################################
    # Change this function
    
    def fitness(self, chromosome, X_train, X_test, y_train, y_test):
        
        # Linear regression is faster:
        lsq = LinearRegression()
        lsq.fit(X_train[:, chromosome.astype(bool)], y_train)
        y_pred = lsq.predict(X_test[:, chromosome.astype(bool)])
        
        # # HuberRegressor is robust to outliers in training set
        # robust_reg = HuberRegressor()
        # scaler = StandardScaler()
        # pipe = Pipeline([('scaler', scaler), ('reg', robust_reg)])
        # pipe.fit(X_train[:, chromosome.astype(bool)], y_train)
        # y_pred = pipe.predict(X_test[:, chromosome.astype(bool)])
        
        score = np.sqrt(mean_squared_error(y_pred, y_test))
        fitness = - score
        return fitness
    
    #############################################################################################################

    def fitness_pop(self, population, X_train, X_test, y_train, y_test):

        pop_fitness = []
        for chromosome in population:
            if np.sum(chromosome) == 0: # since a chromosome of only 0's id redundant and cannot be regressed
                print('Zero chromosome')
                fitness_= 0
            else:
                fitness_ = self.fitness(chromosome, X_train, X_test, y_train, y_test)
            
            pop_fitness.append(fitness_)
        pop_fitness = np.array(pop_fitness)

        sorted_ind = np.argsort(pop_fitness)[::-1]
        sorted_pop_fitness = pop_fitness[sorted_ind]
        sorted_population = population[sorted_ind, :]
        return sorted_population, sorted_pop_fitness


    # Selection

    def ranked_selection(self, pop_after_fit, cull_frac=0.5):
        '''Performes a ranked selection based in fitness function, drops the degenerate chromosome which are not in the range
        given by n_parents

        args:
        pop_after_fit - the population after being ranked in order of fitness
        cull_frac - fraction of population to be kept during slection

        returns:
        new_pop - the surviving population

        '''
        size = len(pop_after_fit)
        keep_inds = int(round(cull_frac * size))
        new_pop = pop_after_fit[:keep_inds]
        np.random.shuffle(new_pop) # randomise population after cull
        return new_pop


    def roulette_wheel_selection(self, population, scores, cull_frac=0.5):
        '''
        Performes a rouletter wheel selection, i.e. a randomised selection where the propabilities of choosing
        some chromosome to keep is proportional to its fitness
        
        args:
        pop_after_fit - the population after being ranked in order of fitness
        scores - the fitness of each of the chromosomes in the population
        cull_frac - fraction of population to be kept during slection
        

        returns:
        new_pop - the surviving population
        
        '''
        s_inv = 1/np.array(scores)
        s_inv_sum = np.sum(s_inv)
        probabilities = s_inv/s_inv_sum
        new_pop=[]
        for i in range(int(round(len(population)*cull_frac))):
            # Perform roulette wheel selection
            selected_chromosome_idx = np.random.choice(np.arange(len(population)), p=probabilities)
            new_pop.append(population[selected_chromosome_idx])

        np.random.shuffle(new_pop) # randomise population after cull
        return np.array(new_pop)

    # crossover 
    
    def constrained_crossover(self, parent1, parent2):

        '''This crossover function will combine both parent chromosomes, and pass the combined genes to two child
        chromosomes based on the following logic:

        If the chromosomes of both parents contain a mutual gene, then this is a strong gene and is passed
        down to both child chromosomes.

        If one parent has a particular gene and the other does not, then this gene is sent to a "gene pool"
        The gene pool will always have an even number of genes as both parents will contribute the same number
        of genes to the gene pool

        half the remaining gene=1 locations will be randomly distributed to each child chromosome 

        args:
        parent1, parent2 - both parent chromosomes as 1D numpy arrays

        returns:
        child1, child2 - the resultant child chromosomes as numpy 1D arrays

        '''

        child1 = []
        child2 = []
        gene_pool = []
        for i in range(len(parent1)):
            if parent1[i] == parent2[i]:
        #         bool_genes.append(True)
                child1.append(parent1[i]) # append the strong gene to child1 from parent 1 (or parent 2 since they are the same)
                child2.append(parent1[i]) # do same with child 2
            else:
        #         bool_genes.append(False)
                gene_pool.append(i) # add the parent's contradicting gene to the gene pool for random sampling later on
                child1.append(0) # just append the value zero for the contradicting gene
                child2.append(0)

        child1 = np.array(child1)
        child2 = np.array(child2)
        random.shuffle(gene_pool)

        size_gene_pool = len(gene_pool)

        child1_gene_idx = gene_pool[:int(size_gene_pool/2)]
        child2_gene_idx = gene_pool[int(size_gene_pool/2):]

        child1[child1_gene_idx] = 1
        child2[child2_gene_idx] = 1

        return child1, child2

    def one_point_crossover(self, parent1, parent2):
        rand_int = random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:rand_int], parent2[rand_int:]])
        child2 = np.concatenate([parent2[:rand_int], parent1[rand_int:]])
        return child1, child2

    def random_crossover(self, parent1, parent2):
        '''
        Child 1 randomly samples half the genes from parent 1 and half the genes from parent 2
        and child 2 gets the remaining genes. 

        '''
        all_inds = list(range(len(parent1)))
        indsA = random.sample(all_inds, int(round(len(parent1)*0.5)))
        indsB = list(set(all_inds) - set(indsA))

        child1 = np.zeros(len(parent1))
        child2 = np.zeros(len(parent1))

        child1[indsA] = parent1[indsA]
        child1[indsB] = parent2[indsB]

        child2[indsA] = parent2[indsA]
        child2[indsB] = parent1[indsB]

        return child1, child2


    def apply_crossover_to_pop(self, population, crossover_type):
        '''Each parent will crossover with the adjacent parent in the population. As the crossover 
        function produces two children, this will result in the new gen being double the size of the previous gen.

        args:
        population - the sorted population output from fitting function, a list of 1D np arrays

        returns:
        next_gen_pop - the next generation population, twice the size of "population", list of 1D np arrays

        '''

        next_gen_pop=[]
        for i, parent in enumerate(population):
            parent1 = population[i-1]
            parent2 = population[i]
            if crossover_type == 'one point':
                child1, child2 = self.one_point_crossover(parent1, parent2)
            elif crossover_type == 'random':
                child1, child2 = self.random_crossover(parent1, parent2)
            elif crossover_type == 'constrained':
                child1, child2 = self.constrained_crossover(parent1, parent2)

            else:
                print("ERROR: Pick one of the following crossover types: one point, random, constrained")

            next_gen_pop.append(child1)
            next_gen_pop.append(child2)
        
        next_gen_pop = np.array(next_gen_pop)
        np.random.shuffle(next_gen_pop)
        return next_gen_pop
    
    
    def mutation_per_chromosome(self, chromosome, mutation_rate):
        '''
        This will perform a scramble mutation on the chromosome (i.e. shuffles a subset of genes) 
        note, this will always preserve the same number of 1's since we are just shuffling the genes

        '''
        if mutation_rate >= 1 or mutation_rate <= 0:
            print("ERROR: Pick a mutation rate between 0 and 1")

        all_inds = list(range(len(chromosome)))  
        sub_inds = random.sample(all_inds, int(round(len(chromosome)*mutation_rate)))

        genes_to_shuffle = chromosome[sub_inds]
        np.random.shuffle(genes_to_shuffle)

        chromosome[sub_inds] = genes_to_shuffle

        return chromosome

    def mutations_per_pop(self, population, mutation_rate_per_pop=0.7, mutation_rate_per_chromosome=0.7):

        '''
        Will mutate each chromosome given some probability defined by the mutation_rate_per_pop

        '''

        def random_true():
            return random.random() < mutation_rate_per_pop # returns true with probability of mutation rate

        mutated_pop = []
        for chromosome in population:
            if random_true():
    #             print('true')
    #             print('before', chromosome)
                chromosome = self.mutation_per_chromosome(chromosome, mutation_rate_per_chromosome)
    #             print('after ', chromosome)
            else:
    #             print('false')
    #             print('passed', chromosome)
                pass
            mutated_pop.append(chromosome)

        return np.array(mutated_pop)

        

    def fit(self, R_df_train, y_df_train, R_df_test, y_df_test):
        
        '''Function applies the genetic algorithm for a ridge regression problem - it includes all the steps above

            args: 
            X_df - the dataframe of independent variables (pandas dataframe)
            y_df - the response variable (pandas series)
            selection_type - takes args:
                                        'ranked' - selets best half of population according to fitness rankings
                                        'roulette wheel' - more randomised selection based on probabilities proportional to fitness
            crossover_type - takes args: 
                                        'one point' - randomly selects a single point to crossover
                                        'random' - random selection of half the genes to crossover from each parent
                                        'constrained' - random crossover of genes, maintaining the same number of 1's
            
            
            mutation_rate_per_pop - frequency of mutations in population
            mutation_rate_per_chromosome - the magnitude of a mutation on chromosome
            
            
            returns:
            reduced_X_df - the reduced dataframe with the optimised features.

        '''

        R_train = np.array(R_df_train)
        y_train = np.array(y_df_train)
        
        R_test = np.array(R_df_test)
        y_test = np.array(y_df_test)

        numb_features = np.shape(R_train)[1]

        start_gen = self.initilization_of_population(numb_features, numb_features, self.initial_ones)

        gen_i, pop_fitness = self.fitness_pop(start_gen, R_train, R_test, y_train, y_test)

        for i in range(self.max_gen):
            
            self.generation.append(i)
            self.best_fitness.append(pop_fitness[0])
            self.best_matching_chromosome.append(gen_i[0])

            if self.selection_type == 'ranked':
                gen_i = self.ranked_selection(gen_i, cull_frac=0.5) # shuffle pop at end
            elif self.selection_type == 'roulette wheel':
                gen_i = self.roulette_wheel_selection(gen_i, pop_fitness, cull_frac=0.5) #shuffle pop at end
            else:
                print('Choose selection type from: ranked, roulette wheel')


            next_gen = self.apply_crossover_to_pop(gen_i, crossover_type=self.crossover_type) #shuffle pop at end

            next_gen = self.mutations_per_pop(next_gen, self.mutation_rate_per_pop, self.mutation_rate_per_chromosome)

            next_gen, pop_fitness = self.fitness_pop(next_gen, R_train, R_test, y_train, y_test)
            
            gen_i = next_gen  

        # self.best_chromosome = gen_i[0,:]

        # self.best_features = R_df_train.iloc[:, self.best_chromosome.astype(bool)].columns
        
        max_idx = self.best_fitness.index(max(self.best_fitness))
        best_chrom_all_gens = self.best_matching_chromosome[max_idx]
        
        self.best_features = R_df_train.iloc[:, best_chrom_all_gens.astype(bool)].columns
                
        return self
    
    def transform(self, X_df, y_df=None):

        return X_df.loc[:, self.best_features]
    
    
    def plot_fitness(self):
        
        fig, ax = plt.subplots();

        ax.plot(self.generation, self.best_fitness)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        
        return fig, ax








# Example:


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split    


# 1. Create synthetic data for a linear regression
X, y = make_regression(
    n_samples=200,
    n_features=30,
    n_informative=5, shuffle=False, # First five features have an actual relationship with y, the remaining ones do not
    noise=0.5,
    random_state=1,
    
)

cols = [f"feat_{i}" for i in range(X.shape[1])]
R_df = pd.DataFrame(X, columns=cols)
y_df = pd.Series(y, name="target")

# Split into train/test
R_train, R_test, y_train, y_test = train_test_split(
    R_df, y_df, test_size=0.3, random_state=1
)

# 2. Run GA
ga = GeneticAlgorithm(
    max_gen=50,
    initial_ones=5,
    selection_type='roulette wheel',
    crossover_type='constrained',
    mutation_rate_per_pop=0.6,
    mutation_rate_per_chromosome=0.3
)
ga.fit(R_train, y_train, R_test, y_test) # GA uses prediction accuracy on test set to calculate the fitness

# 3. Output results
print("Selected features:", list(ga.best_features))

# 4. Plot fitness curve
fig, ax = ga.plot_fitness()
ax.set_title("GA Fitness over Generations")
plt.show()
    
