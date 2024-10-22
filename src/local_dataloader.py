from openai import OpenAI
import datasets

dataset_names = ['openai/gsm8k', 'hotpotqa/hotpot_qa']
CURRENT_DATASET = 'openai/gsm8k'

def load_data(CURRENT_DATASET = CURRENT_DATASET,n_rows=None):
    if CURRENT_DATASET == dataset_names[0]:
        dataset = datasets.load_dataset(CURRENT_DATASET, 'main', split='train', trust_remote_code=True) #main or socratic
    else:
        dataset = datasets.load_dataset(CURRENT_DATASET,'distractor', split = 'train', trust_remote_code=True) #distractor, full_wiki

    print(len(dataset))
    print(dataset[0])
    return dataset[:n_rows] if n_rows else dataset



#Dataset descriptions
'''
openai/gsm8k
main: For the main configuration, each instance contains a string for the grade-school level math question and a string for the corresponding answer with multiple steps of reasoning and calculator annotations

socractic: For the socratic configuration, each instance contains a string for a grade-school level math question, a string for the corresponding answer with multiple steps of reasoning, calculator annotations (explained here), and Socratic sub-questions. We also investigated a modified solution format that injects automatically generated "Socratic subquestions" before each step. 


hotpotqa/hotpot_qa
distractor: In the distractor setting, a question-answering system reads 10 paragraphs to provide an answer to a question. They must also justify these answers with supporting facts. This setting challenges the model to find the true supporting facts in the presence of noise, for each example we employ bigram tf-idf (Chen et al., 2017) to retrieve 8 paragraphs from Wikipedia as distractors, using the question as the query. We mix them with the 2 gold paragraphs (the ones used to collect the question and answer) to construct the distractor setting.

fullwiki setting:  In the fullwiki setting, a question-answering system must find the answer to a question in the scope of the entire Wikipedia. We fully test the model’s ability to locate relevant facts as well as reasoning about them by requiring it to answer the question given the first paragraphs of all Wikipedia articles without the gold   paragraphs specified. This full wiki setting truly tests the performance of the systems’ ability at multi-hop     reasoning in the wild.
'''
