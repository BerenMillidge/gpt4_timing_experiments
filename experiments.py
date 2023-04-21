import time
import openai
import requests
import prompt_toolkit
from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import button_dialog, radiolist_dialog, yes_no_dialog, input_dialog
import matplotlib.pyplot as plt
import json
import numpy as np
import fire

api_key = prompt_toolkit.prompt('What is your OAI api key?', is_password = True)
openai.api_key = api_key
API_KEY = api_key
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

def generate_chat_completion(messages, model="gpt-4", temperature=0.7, max_tokens=None, sleep_time = 5):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
    if VERBOSE:
        print("RESPONSE: ", response.json())

    counter = 0
    while response.status_code != 200 and counter < 5:
        print('Generation request failed. Retrying')
        time.sleep(sleep_time)
        requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
        counter += 1

    return response.json()["choices"][0]["message"]["content"]

def generate_completion(prompt, system_prompt = None, temperature = 0.7, max_tokens = 2000, model = "gpt-4"):
    if not system_prompt:
        system_prompt = "You are a helpful assistant."
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
    ]

    response = generate_chat_completion(messages, model = model,temperature = temperature, max_tokens = max_tokens)
    return response


def prompt_length_timing(N_increments=100, N_runs=10, save_name="prompt_length_timings",model="gpt-4"):
    base_token = "hello " # presumably a single token

    # let's first do timing experiments on the prompt length
    # there is also a pretty large amount of variability between runs. let's see what is going on
    time_list_list = []
    for t in range(N_runs):
        time_list = []
        for i in range(20):
            prompt = ""
            for j in range(i):
                for k in range(N_increments):
                    prompt += base_token
            if VERBOSE:
                print("PROMPT: ", prompt)
            t0 = time.time()
            response = generate_completion(prompt, temperature = 0.0, max_tokens = 1, model=model)
            t1 = time.time()
            time_list.append(t1 - t0)
            if VERBOSE:
                print("TIMEDIFF:", t1 - t0)
        time_list_list.append(np.array(time_list))
    time_list_list = np.array(time_list_list)
    np.save(save_name + ".npy", time_list_list)
    print(time_list_list)
    mean_time_list = np.mean(time_list_list, axis=0)
    std_time_list = np.std(time_list_list, axis=0)


    xs = np.arange(0, len(mean_time_list))
    fig = plt.figure()
    plt.plot(xs,mean_time_list)
    plt.fill_between(xs, mean_time_list - std_time_list, mean_time_list + std_time_list, alpha=0.5)
    plt.ylabel("Call time (seconds)")
    plt.xlabel("Prompt length tokens (*" + str(N_increments) + ")")
    plt.title("Time to generate a single token against prompt length")
    plt.savefig(save_name + ".png", format="png")
    plt.show()
    return time_list_list

def generation_length_timing(N_increments=100, N_runs=10, save_name="generation_length_timings",model="gpt-4"):
    base_token = "hello " # presumably a single token

    time_list_list = []
    prompt = "Hello there. Please can you keep repeating the word 'hello' for me for as long as possible. Like so: \nhello hello hello hello hello hello hello hello"
    for t in range(N_runs):
        time_list = []
        for i in range(20):
            t0 = time.time()
            response = generate_completion(prompt, temperature = 0.0, max_tokens = ((i * N_increments)+1), model=model)
            t1 = time.time()
            time_list.append(t1 - t0)
            if VERBOSE:
                print("TIMEDIFF:", t1 - t0)
        time_list_list.append(np.array(time_list))
    time_list_list = np.array(time_list_list)
    np.save(save_name + ".npy", time_list_list)
    print(time_list_list)
    mean_time_list = np.mean(time_list_list, axis=0)
    std_time_list = np.std(time_list_list, axis=0)


    xs = np.arange(0, len(mean_time_list))
    fig = plt.figure()
    plt.plot(xs,mean_time_list)
    plt.fill_between(xs, mean_time_list - std_time_list, mean_time_list + std_time_list, alpha=0.5)
    plt.ylabel("Call time (seconds)")
    plt.xlabel("Generation length in tokens (*" + str(N_increments) + ")")
    plt.title("Generation time for an N token string")
    plt.savefig(save_name + ".png", format="png")
    plt.show()
    return time_list_list

def main(verbose=True):
    global VERBOSE
    VERBOSE = verbose
    prompt_length_timing(N_increments = 100, N_runs = 20,model="gpt-3.5-turbo", save_name = "prompt_length_timings_35_turbo")
    generation_length_timing(N_increments = 100, N_runs = 20,model = "gpt-3.5-turbo",save_name="generation_length_timings_35_turbo")
    

if __name__ == "__main__":
    fire.Fire(main)



