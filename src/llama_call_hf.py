from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import json
import torch

with open('D:\\keys.json') as f:
    hf_token = json.load(f)['hf_token']



model_name = 'meta-llama/Llama-3.2-3B-Instruct'
hf_modelpath = 'D:\\hf_models'
model_filepath = model_name.split('/')[1].replace('-','_').replace('.','_') if '/' in model_name else model_name.replace('-','_').replace('.','_')
modelpath = os.path.join(hf_modelpath,model_filepath)
if not os.path.exists(modelpath):
    os.makedirs(modelpath)

os.environ['HF_HOME'] = modelpath

def load_new_model(model_name,params):
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir =modelpath, token =hf_token, device_map = 'cuda',**params) if params else AutoModelForCausalLM.from_pretrained(model_name, cache_dir =modelpath, token =hf_token, device_map = 'cuda')
    model.save_pretrained(modelpath)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir =modelpath, token =hf_token)
    tokenizer.save_pretrained(modelpath)


def get_pipe(model_name=model_name, sampling_params=None):
    all_params = ['temperature', 'max_new_tokens', 'top_k', 'min_p','top_p', 'num_beams','do_sample','num_return_sequences']
    assert list(sampling_params.keys())[0] in all_params
    if not sampling_params:
        pipe = pipeline(model=model_name, token=hf_token, device_map = 'cuda',)
    else:
        pipe = pipeline(model=model_name, token=hf_token, device_map = 'cuda',truncation = True, **sampling_params)
    return pipe

def llm_response(user_query, model_name=model_name, sampling_params=None):
    arg_model_size ={'torch_dtype':torch.float16}
    if sampling_params:
        sampling_params.update(arg_model_size)
    pipeline = get_pipe(model_name,sampling_params)
    st_time =time.time()
    response = pipeline(user_query)
    end_time = time.time()
    print('Time taken: ',{end_time - st_time})
    return response[0]['generated_text'], end_time-st_time

def get_all_llm_responses(query_list,model_name=model_name, sampling_params=None):
    responses = []
    arg_model_size ={'torch_dtype':torch.float16}
    if sampling_params:
        sampling_params.update(arg_model_size)
    pipeline = get_pipe(model_name,sampling_params)
    # for query in query_list:
    #     st_time = time.time()
    #     result = pipeline(query)[0]['generated_text']
    #     end_time = time.time()
    #     responses.append((result, end_time - st_time))
    responses = [pipeline(query)[0]['generated_text'] for query in query_list]
    return responses


if __name__ == "__main__":
    prompts = [
    'Hello, my name is',
    'The president of the United States is',
    'The capital of France is',
    'The future of AI is'
]
    user_params = {'temperature':2.0, 'max_new_tokens':1024,'num_return_sequences':3}
    import time
    st_time =time.time()
    user_query = 'Hello, my name is'
    responses = get_all_llm_responses(prompts, model_name, user_params)
    for response in responses:
        print(response[0])
        print(response[1])
        print('***********************************************')
        print('\n')