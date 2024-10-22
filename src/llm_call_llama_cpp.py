from config import client
import requests
import json
llm_server_url = "http://192.168.1.4:8080"

# sys_message = ''

# def create_prompt(user_message):
#     print(f'<s>[INST] <<SYS>>\n{sys_message}\n<</SYS>>\n\n{user_message} [/INST]')
#     return f'<s>[INST] <<SYS>>\n{sys_message}\n<</SYS>>\n\n{user_message} [/INST]'

# def post_query_llm(message, sampling_params):
#     beam_search = sampling_params.get('use_beam_search',None)
#     if beam_search:
#         best_of = sampling_params('best_of',3)
#         extra_body ={
#         "use_beam_search": beam_search,
#         "best_of": best_of
#         }
#     else:
#         extra_body = None
#     top_k = sampling_params.get('top_k',40)
#     top_p = sampling_params.get('top_p',0.9)
#     frequency_penalty = sampling_params.get('frequency_penalty',1.1)
#     max_tokens = sampling_params.get('max_tokens',1024)
#     presence_penalty = sampling_params.get('presence_penalty', 0.0)
#     temperature = sampling_params.get('temperature',0.8)

#     return requests.post(f"{llm_server_url}/completions", 
#                   data=json.dumps(
#                 {"prompt": message,
#                 "temperature": temperature,
#                 "top-k":top_k,
#                 'top-p': top_p,
#                 'frequency-penalty':frequency_penalty,
#                 'n_predict':max_tokens,
#                 "cache_prompt": True,
#                 "presence_penalty":presence_penalty
#             },
#         ),
#         stream=False
#     )

#top_k doesn't exist in OpenAI API
def post_to_llm(message, sampling_params):
    print('Message', message)
    data = {'prompt':message}
    data.update(sampling_params)
    print('All data:\n', data)  
    print(f"{llm_server_url}")
    return requests.post(url=f"{llm_server_url}/completion", data=json.dumps(data))


def query_to_prompt(message_list):
    return[{'role':'user', 'content':message} for message in message_list]

def call_llm(messages,sampling_params):
    print(messages)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[messages],
        **sampling_params,
    )
    return response.choices[0].message.content, response.completion_probabilities

def process_multiprompt(message_list,sampling_params):
    formats = query_to_prompt(message_list)
    return [call_llm(message, sampling_params) for message in formats]


prompts = [
    'Hello, my name is',
    'The president of the United States is',
    'The capital of France is',
    'The future of AI is'
]

# print(post_to_llm(test_prompt, sampling_params={'temperature':1.0}).json())
outputs = process_multiprompt(prompts,{'temperature':5, 'logprobs':True})
for output in outputs:
    print(output[0])

# curl --request POST --url http://localhost:8080/completion --header "Content-Type: application/json" --data '{"prompt": "Building a website can be done in 10 simple steps:","n_predict": 128}'