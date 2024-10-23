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
