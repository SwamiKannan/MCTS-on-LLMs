# First commit to git
# Need to relook at where the updated answer is being stored. Something is weird. Best nodes just keeo saying "I dont know" or the seed answers basically
# Need to add system prompts for critique and rater
# Add stop gap in case the answer is complete

import os
import json
import time
import math
from local_dataloader import load_data, parse_gsm8k
from prompts import get_critique, improve_answer, rate_answer, get_answer_directly_from_llm, seed_answers

import random

EXPLORATION_CONSTANT = math.sqrt(2) # Exploration constant for UCB
EPSILON_CONSTANT = 1e-3 # If unvisited node, then the n_visits = 0 hence, node's UCB becomes infinite. This throttles it down
MAX_SCORE = 0.95 #Even if the score is 1 i.e. 100%, set it as 0.95

#define all possible samplers

extra_body = {"use_beam_search": True,
"best_of": 1
}
samplers = [{'temperature':0.0, 'extra_body':extra_body}, {'temperature':0.8, 'top_p':0.7}, {'temperature':1.2, 'top_k':20}]

class Node():
    def __init__(self, id, question, answer,parent=None):
        self.id = str(id)
        self.parent = parent
        self.children = []
        self.total_visits = 0
        self.episilon = EPSILON_CONSTANT
        self.score = 0
        self.response = ''
        self.question = question
        self.answer = answer
        try:
            self.sampler = samplers[int(self.id.split('_')[-1])%len(samplers)] if '_' in id else samplers[int(self.id)]
        except:
            print('id:\t',self.id)
            raise Exception('Sampler could not be sampled')
        self.critique = ''
        self.context = self.build_context()
        self.status = None
        self.ucb = 0
    
    def build_context(self):
        if self.parent:
            self.context = self.question + self.parent.answer + self.parent.critique 
        else:
            self.context = self.question
        
    def create_children(self, max_children):
        for i in range(max_children):
            child = Node(str(self.id)+'_'+str(i), question=self.question, answer=None, parent=self)
            if self.parent:
                child.answer = improve_answer(self.question, self.parent.answer, self.parent.critique)
            else:
                child.answer = improve_answer(self.question, child.answer,'')
            child.critique= get_critique(self.question, child.answer)
            self.children.append(child)

    def update_ucb(self, exploration_constant = EXPLORATION_CONSTANT):
        return (self.score/self.total_visits)+exploration_constant*(math.sqrt(math.log(self.parent.total_visits)/(self.total_visits+EPSILON_CONSTANT)))
        
    def update_children_score(self):
        for child in self.children:
            child.score = self.ucb_score()
            
    def get_most_visited_child(self):
        max_visits = max([child.total_visits for child in self.children])
        selected_children = [child for child in self.children if child.score == max_visits]
        selected_child = selected_children[0] if len(selected_children)<2 else random.choice(selected_children)
        return selected_child
        
    def select_child_for_exploring(self):
        if len(self.children) == 0:
            raise Exception('No children available')
        max_ucb = max([child.ucb for child in self.children])
        selected_children = [child for child in self.children if child.score == max_ucb]
        try:
            selected_child = selected_children[0] if len(selected_children)<2 else random.choice(selected_children)
        except:
            print('Exception in selecting child')
            print('Max UCB score:\t', max_ucb)
            print('Scores:\t',[child.score for child in self.children])
            raise Exception('Selected Child not found')

        selected_child.total_visits +=1
        return selected_child
        
class MCTS():
    def __init__(self, question, seed_answers, max_children=3, iterations = 10):
        self.question = question
        self.max_children = max_children
        self.iterations = iterations
        self.create_root_node()
        self.max_score = 0
        self.best_node = []
        
    def create_root_node(self):
        self.root_node = Node(id ='0', question=self.question, answer = random.choice(seed_answers))
        self.root_node.total_visits = 1
        
    def update_tree_score(self,node, score):
        while node.parent:
            node.score += score
            node.update_ucb()
            node = node.parent  
    
    def run_tree_analysis(self, critique_reqd=True):
        for i in range(self.iterations):
            st_time = time.time()
            node = self.root_node
            if len(node.children) < self.max_children:
                node.create_children(self.max_children)
            node = node.select_child_for_exploring()
            score, critique = rate_answer(self.question,node.answer)
            node.score=  score
            if score > self.max_score:
                self.max_score = score
                self.best_node = [node]
            if score == self.max_score:
                self.best_node.append(node)
            if critique_reqd:
                print('Critique:\n',critique)
            self.update_tree_score(node, score)
            end_time  = time.time()
            print(f'Iteration {i+1} complete in {end_time - st_time :.2f} seconds')
            
    def get_main_steps(self):
        paths = []
        for node in self.best_node:
            print('Best node:\t',node.id,'\tScore:\t',node.score)
            path_id = node.id.split('_')
            steps_taken = []
            node = self.root_node
            for id in path_id[1:]:  
                steps_taken+=[{'answer':node.answer, 'critique':node.critique}]
                try:
                    node =node.children[int(id)]
                except:
                    print('Path issue:\n')
                    print('ID:\t', id)
                    print()
            paths.append(steps_taken)
        return paths
        
    def search(self):
        self.run_tree_analysis()
        path = self.get_main_steps()
        return path
        
if __name__ == "__main__":
    dataset = parse_gsm8k()
    q = dataset[1]['question']
    fa = dataset[1]['final_answer']
    mcts = MCTS(question=q, seed_answers=seed_answers)
    path_ex = mcts.search()
    print(path_ex)
    for p in path_ex:
        print
    with open('path_example.txt') as f:
        f.write(str(path_ex))


            # create childrent that dont exist
            # For each child, populate an answer
            # score those answers
            # backpropogate those answers