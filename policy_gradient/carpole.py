from policy import * 
from utils import * 
from train import *
from eval import *


env = make_env("CartPole-v1")
action_space = [_ for _ in range((env.action_space.n))]
n_actions = len(action_space)
input_dims = env.observation_space.shape
noe = 1000 
max_steps = 1e7
video_prefix = "policy_gradient_reinforce"
is_tg = True 
record = True
lr = 1e-5
gamma = 0.94
chpkt = 'models/'
algo_name = "reinforce"
video_prefix = "cartpole_policy_gradient"

if __name__ == "__main__": 
    trainer = Trainer(env, action_space, input_dims, n_actions, video_prefix, is_tg, noe, max_steps, record, lr, gamma, chpkt, algo_name)
    ep_rewards = trainer.train()

    plot_learning_curve(ep_rewards, 'cartpole_reinoforce')

    model_path = "models/cartpole_reinforce_"

    evaluator = Eval(env, action_space, model_path,video_prefix, 10)
    evaluator.test()


