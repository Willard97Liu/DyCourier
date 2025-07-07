from torch import softmax
from tqdm import tqdm
from copy import deepcopy

import numpy as np
from methods.agent import GreedyAgent, MSAAgent, Agent, OfflineAgent, SLAgent, DQNAgent
from envs import DynamicQVRPEnv

import os

import pickle


def run_agent(
    agentClass,
    env_configs : dict,
    episodes = 10,
    agent_configs : dict = {},
    save_results = False,
    title = None,
    path = 'results/'
    ):
    
    env = DynamicQVRPEnv(**env_configs)
    agent = agentClass(env, env_configs = env_configs, **agent_configs)
    
    rs, actions, infos = agent.run(episodes)
    
    res = {
        "rs" : rs,
        "actions" : actions,
        "infos" : infos
    }
    if save_results:
        tit = path+title+'.pkl' if title is not None else f'results/{agentClass.__name__}.pkl'
        with open(tit, 'wb') as f:
            pickle.dump(res, f)
            
    del agent
    
    return rs, actions, infos
    

def experiment(
        episodes = 200,
        env_configs = {
            "horizon" : 50,
            "Q" : 100, 
            "DoD" : 0.5,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "emissions_KM" : [.1, .3],
            "n_scenarios" : 500
        },
        RL_hidden_layers = [512, 512, 512],
        RL_model = None,
        RL_name_comment = '',
    ):
    """Compares different methods implemented so far between them on the 
    same environment.

    Parameters
    ----------
    episodes : int, optional
        The number of episodes to run for each agent, by default 200
    """
    
    with open(f'results/env_configs.pkl', 'wb') as f:
        pickle.dump(env_configs, f)
        
    try:
        va = 'VA' if env_configs['vehicle_assignment'] else 'OA'
    except:
        va = 'OA'
        
    env_configs['k_med'] = 7
    # if env_configs["re_optimization"] : 
    env_configs_DQN_VA_as_OA = deepcopy(env_configs)
    env_configs_DQN_VA_as_OA["vehicle_assignment"] = True
    
    env_configs_DQN_VA = deepcopy(env_configs_DQN_VA_as_OA)
    env_configs_DQN_VA["re_optimization"] = False
        
    # RL_name = f"res_RL_DQN_{va}{RL_name_comment}"
    RL_model_comment = ''
    if "cluster_scenario" in env_configs and env_configs["cluster_scenario"]:
        RL_model_comment += 'clusters_'
    elif "uniform_scenario" in env_configs and env_configs["uniform_scenario"]:
        RL_model_comment += 'uniform_'
        
    RL_model_comment += 'VRP' if len(env_configs["emissions_KM"])>1 else 'TSP'
    RL_model_comment += str(len(env_configs["emissions_KM"])) if len(env_configs["emissions_KM"])>1 else ''
    RL_model_comment += f'Q{env_configs["Q"]}'
    try:
        RL_model_comment += '_uniforme' if env_configs['noised_p'] and va=='VA' else ''
    except:
        pass
        
    if RL_model is None:
        RL_model = f'DQN_{RL_model_comment}_VA'
    agents = {
        "fafs" : dict(
            agentClass = GreedyAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {},
            save_results = True,
            title = "res_fafs",
        ),
        "random" : dict(
            agentClass = Agent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {},
            save_results = True,
            title = "res_random",
        ),
        
        "RL_VA" : dict(
            agentClass = DQNAgent,
            env_configs = env_configs_DQN_VA,
            episodes = episodes,
            agent_configs = dict(
                algo = RL_model,
                hidden_layers = RL_hidden_layers, 
            ),
            save_results = True,
            title = "res_RL_DQN_VA",
        ),
        "offline" : dict(
            agentClass = OfflineAgent,
            env_configs = env_configs,
            episodes = episodes,
            agent_configs = {"n_workers": 7},
            save_results = True,
            title = "res_offline",
        ),
    }
    
    for agent_name in agents:
        run_agent(**agents[agent_name])
        print(agent_name, "done")
        

def experiment_DoD(
        episodes = 500,
        DoDs = [1., .95, .9, .85, .8],#, .75, .7, .65, .6],
        env_configs = {
            "horizon" : 50,
            "Q" : 100, 
            # "DoD" : 0.5,
            "vehicle_capacity" : 25,
            "re_optimization" : False,
            "emissions_KM" : [.1, .3],
            "n_scenarios" : 500
        },
        RL_hidden_layers = [512, 512, 512],
        RL_model = None,
        RL_name_comment = '',
    ):
    
    try:
        va = 'VA' if env_configs['vehicle_assignment'] else 'OA'
    except:
        va = 'OA'
        
    env_configs['k_med'] = 17
    # if env_configs["re_optimization"] : 
    env_configs_DQN_VA_as_OA = deepcopy(env_configs)
    env_configs_DQN_VA_as_OA["vehicle_assignment"] = True
    
    env_configs_DQN_VA = deepcopy(env_configs_DQN_VA_as_OA)
    env_configs_DQN_VA["re_optimization"] = False
    
        
    # RL_name = f"res_RL_DQN_{va}{RL_name_comment}"
    if "cluster_scenario" in env_configs and env_configs["cluster_scenario"]:
        RL_model_comment = 'clusters'
    else:
        RL_model_comment = 'VRP' if len(env_configs["emissions_KM"])>1 else 'TSP'
        RL_model_comment += str(len(env_configs["emissions_KM"])) if len(env_configs["emissions_KM"])>1 else ''
        try:
            RL_model_comment += '_uniforme' if env_configs['noised_p'] and va=='VA' else ''
        except:
            pass
        
    if RL_model is None:
        RL_model = f'DQN_{RL_model_comment}_VA'
    
    for dod in DoDs:
        
        path = f'results/DoD{dod:.2f}/'
        try:
            os.mkdir(path)
        except :
            pass
        
        env_configs["DoD"] = dod
        env_configs_DQN_VA["DoD"] = dod
        with open(f'{path}env_configs.pkl', 'wb') as f:
            pickle.dump(env_configs, f)

        agents = {
            "greedy" : dict(
                agentClass = GreedyAgent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = {},
                save_results = True,
                title = "res_greedy",
            ),
            "random" : dict(
                agentClass = Agent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = {},
                save_results = True,
                title = "res_random",
            ),
            "offline" : dict(
                agentClass = OfflineAgent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = {"n_workers": 7},
                save_results = True,
                title = "res_offline",
            ),
            "MSA" : dict(
                agentClass = MSAAgent,
                env_configs = env_configs,
                episodes = episodes,
                agent_configs = dict(n_sample=51, parallelize = True),
                save_results = True,
                title = "res_MSA",
            ),
        }

        for agent_name in agents:
            run_agent(**agents[agent_name], path=path)
            print(agent_name, "done")
     
       
if __name__ == "__main__":
    
    experiment(
        100,
        env_configs = {
            "horizon" : 100,
            "Q" : 50, 
            "DoD" : 1.,
            "vehicle_capacity" : 20,
            "re_optimization" : True,
            "emissions_KM" : [.1, .1, .3, .3],
            # "n_scenarios" : 500,
            "different_quantities" : True,
            "test"  : True,
            # "vehicle_assignment" : True,
        },
        RL_hidden_layers = [1024, 1024, 1024],
    )
    