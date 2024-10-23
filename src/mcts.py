import os
import json
import numpy as np
from local_dataloader import load_data
from llama_call_hf import process_multilist
import random

EXPLORATION_CONSTANT = np.sqrt(2)
EPSILON_CONSTANT = 1e-5

#define all possible samplers

extra_body = {"use_beam_search": True,
"best_of": 1
}
samplers = [{'temperature':0.0, 'extra_body':extra_body}, {'temperature':0.8, 'top_p':0.7}, {'temperature':1.2, 'top_k':20}]

class Node():
    def __init__(self, id, question, answer,parent=None):
        self.id = str(id)
        self.level = parent.level + 1 if parent else 0
        self.parent = parent
        self.children = []
        self.total_visits = 0
        self.episilon = EPSILON_CONSTANT
        self.score = float('inf')
        self.response = None
        self.question = question
        self.answer = answer
        self.sampler = samplers[int(id.split('_')[-1])]
        
    def create_children(self, max_children):
        for i in range(max_children-len(self.children)):
            self.children.append(Node(str(self.id)+'_'+str(i+len(self.children)), question=self.question, answer=None, parent=self))

    def ucb_score(self, exploration_constant = EXPLORATION_CONSTANT):
        return (self.score/self.total_vists)+exploration_constant*(np.sqrt(np.log(self.parent.total_vists)/self.total_visits))
        
    def update_children_score(self):
        for child in self.children:
            child.score = self.ucb_score()
            
    def get_most_visited_child(self):
        max_visits = max([child.total_visits for child in self.children])
        selected_children = [child for child in self.children if child.score == max_visits]
        selected_child = selected_children[0] if len(selected_children)<2 else random.choice(selected_children)
        return selected_child
        
    def select_child_for_exploring(self):
        max_score = max([child.score for child in self.children])
        selected_children = [child for child in self.children if child.score == max_score]
        selected_child = selected_children[0] if len(selected_children)<2 else random.choice(selected_children)
        selected_child.total_visits +=1
        return selected_child
        
class MCTS():
    def __init__(self, question, seed_answers, max_children=3, iterations = 10):
        self.question = question
        self.seed_answers = seed_answers
        self.max_children = max_children
        self.iterations = iterations
        self.create_root_node()
        
    def create_root_node(self):
        self.root_node = Node(id ='1')
        self.root_node.question = 'question'
        self.root_node.answer = random.choice(self.seed_answers)
        
    def build_tree(self):
        pass