import numpy as np
import torch
import random

from textattack.goal_function_results.goal_function_result import (
	GoalFunctionResultStatus,
)

import torch
import numpy as np


from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.shared.validators import transformation_consists_of_word_swaps
from textattack.search_methods import PopulationMember



class GA(SearchMethod):
    
    #Note that what is our fitness value is is called score or result extensively throughout the API.
    #I will attempt to use the term fitness where possible, but the engine for determining if the search is complete,
    #the goal function, defaults to the use of result or score. 


    def __init__(
        self,
        population_size=10,
        generations=10,
        crossover_type = 'uniform',
        parent_selection ='roulette'
    ):
            self.search_over = False
            
            self.generations = generations
            self.population_size = population_size
           
            self.crossover_type = crossover_type
            self.parent_selection = parent_selection
        
        
  
    def create_population(self, doc, population_size):
        
        print("Generating Initial Population\n")
        

        #Set the number of possible transformations attribute for each index
        
        
        words = doc.attacked_text.words
        available_word_replacements = np.zeros(len(words))
        
        #Call the API's transformation method as defined in the attack recipe below
        transformed_texts = self.get_transformations(doc.attacked_text, original_text=doc.attacked_text)
        
        for transformed_text in transformed_texts:
            index = next(iter(transformed_text.attack_attrs["newly_modified_indices"]))
            available_word_replacements[index] += 1

                
        population = []
        for i in range(population_size):
            #Generate clone of original document
            d = PopulationMember(doc.attacked_text,doc,attributes={"available_word_replacements": np.copy(available_word_replacements)})
            #Mutate
            d = self.mutate(d, doc)
            population.append(d)

        print("Population Created\n\n")
        
        return population
    
    
    
    
    def mutate(self, doc, original_result):
       
        #Number of possible locations for word replacement
        non_zero_indices = np.count_nonzero(doc.attributes["available_word_replacements"])
        
        #No more possible perturbations for this document
        if non_zero_indices == 0:
            return doc
        
        iterations = 0
        while iterations < non_zero_indices:
           
            index = random.randint(0,doc.attacked_text.num_words)
            #Get transformed documents where word replacement is at the above index
            transformed_texts = self.get_transformations(doc.attacked_text,original_text=original_result.attacked_text,indices_to_modify=[index])

            #Transformation failed, no candidates found
            if not len(transformed_texts):
                iterations += 1
                continue

            #Check mutated candidates against the goal function
            new_results, self.search_over = self.get_goal_results(transformed_texts)

            #Calculate comparative fitness values for mutated candidates against original document
            candidate_fitnesses = (torch.Tensor([candidate.score for candidate in new_results]) - doc.result.score)
            
            #New best candidate found
            if len(candidate_fitnesses) and candidate_fitnesses.max() > 0:
                best_fitness_index = candidate_fitnesses.argmax()
                available_word_replacements = np.copy(doc.attributes["available_word_replacements"])
                available_word_replacements[index] = 0
                doc = PopulationMember(transformed_texts[best_fitness_index],new_results[best_fitness_index],attributes={"available_word_replacements": available_word_replacements})
                
                return doc

            iterations += 1

            #Successful perturbation found or query budget exhausted
            if self.search_over:
                break

        return doc
    
    

    def crossover(self, parent1, parent2, original_text):

        parent1_chromosome = parent1.attacked_text
        parent2_chromosome = parent2.attacked_text

        passed_constraints = False
        
        #crossover operation
        
        if self.crossover_type == 'uniform':
            
            indices_to_replace = []
            words_to_replace = []
            available_word_replacements = np.copy(parent1.attributes["available_word_replacements"])

            for i in range(parent1.num_words):
                if random.random() < 0.5:
                    indices_to_replace.append(i)
                    words_to_replace.append(parent2.words[i])
                    available_word_replacements[i] = parent2.attributes["available_word_replacements"][i]

            child = parent1.attacked_text.replace_words_at_indices(indices_to_replace, words_to_replace)
            attributes = {"available_word_replacements": available_word_replacements}

        elif self.crossover_type == '1point':
            
            indices_to_replace = []
            words_to_replace = []
            available_word_replacements = np.copy(parent1.attributes["available_word_replacements"])
            
            index = random.randint(1, parent1.num_words)

            for i in range(index):
                    indices_to_replace.append(i)
                    words_to_replace.append(parent2.words[i])
                    available_word_replacements[i] = parent2.attributes["available_word_replacements"][i]

            child = parent1.attacked_text.replace_words_at_indices(indices_to_replace, words_to_replace)
            attributes = {"available_word_replacements": available_word_replacements}

        #record differing indicies in created child
    
        replaced_indices = child.attack_attrs["newly_modified_indices"]
        child.attack_attrs["modified_indices"] = (parent1_chromosome.attack_attrs["modified_indices"] - replaced_indices) | (parent2_chromosome.attack_attrs["modified_indices"] & replaced_indices)



        if "last_transformation" in parent1_chromosome.attack_attrs:
            child.attack_attrs["last_transformation"] = parent1_chromosome.attack_attrs["last_transformation"]
        elif "last_transformation" in parent2_chromosome.attack_attrs:
            child.attack_attrs["last_transformation"] = parent2_chromosome.attack_attrs["last_transformation"]

            
        if "last_transformation" in child.attack_attrs:
            if "last_transformation" in parent1_chromosome.attack_attrs:
                previous_text = parent1_chromosome
            else:
                previous_text = parent2_chromosome
                
            #pass child through constraints  
            passed_constraints = self.check_constraints(child, previous_text, original_text=original_text)

        else:
            #crossover returned a parent or equivalent
            passed_constraints = True



        if passed_constraints:
            #successful crossover, return new child
            new_results, self.search_over = self.get_goal_results([child])
            
            return PopulationMember(child, result=new_results[0], attributes=attributes)

        else:
            #no valid child created, return the parent with the best fitness
            #parent = parent1 if parent1.score > parent2.score else parent2
            #alternatively, choosen uniformly between the two parents
            parent = parent1 if random.random() < 0.50 else parent2

            return parent
        
        
     
    #Required perform_search override
    
    #flag self.search_over following standard SearchMethod subclass implementation
    
    def perform_search(self, doc):
        
        print("Starting GA\n")
        
        self.search_over = False
        population = self.create_population(doc, self.population_size)
        best_fitness = doc.score
        factor = 0.50 #Mass to assign for the top selected percentage in truncation parent selection

        for i in range(self.generations):
            
            print("\nGeneration: ",i)
            population = sorted(population, key=lambda x: x.result.score, reverse=True)
            print("\nCurrent Fitness = ",1-best_fitness)
            
            #Check for termination conditions
            
            if self.search_over:
                print("\nExhausted Query Budget\n")
                break
            
            if population[0].result.goal_status  == GoalFunctionResultStatus.SUCCEEDED:
                print("\nAttack Successful, Exiting Search\n")
                break


            #Update best candidate
            
            if population[0].result.score > best_fitness:
                best_fitness = population[0].result.score
                print("\nNew best fitness: ",1-best_fitness)
            
            #Select Parents
            
            if self.parent_selection == 'roulette': 
                #Uniform selection here
                
                #Parent 1 and 2 indicies
                parent1_indicies = random.choices(range(self.population_size-1), k = self.population_size)
                parent2_indicies = random.choices(range(self.population_size-1), k = self.population_size)

            
            elif self.parent_selection == 'truncation':
                
                #Take top factor % of population
                population = population[0:int(self.population_size*factor)] + population[0:int(self.population_size*factor)]
                
                #Parent 1 and 2 indicies
                parent1_indicies = random.choices(range(self.population_size-1), k = self.population_size)
                parent2_indicies = random.choices(range(self.population_size-1), k = self.population_size)


            #Generate Children
            
            children = []
            for index in range(self.population_size - 1):
                
                child = self.crossover(population[parent1_indicies[index]],population[parent2_indicies[index]],doc.attacked_text)
                
                #Check if crossover resulted in a successful goal function call
                if self.search_over:
                    break

                child = self.mutate(child, doc)
                children.append(child)

                #Check if mutate resulted in a successful goal function call
                if self.search_over:
                    break

            population = [population[0]] + children

        return population[0].result


    #Required API functions to guarantee compatibility with the attack class / goal function.


    def check_transformation_compatibility(self, transformation):
        
        return transformation_consists_of_word_swaps(transformation)

    def check_constraints(self, transformed_text, current_text, original_text):

        filtered = self.filter_transformations([transformed_text], current_text, original_text=original_text)
                                               
        return True if filtered else False
    
    
    @property
    def is_black_box(self):
        
        return True

    def extra_repr_keys(self):
        
        return [
            "population_size",
            "generations",
        ]
