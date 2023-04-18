import random 
import imageio
import tensorflow as tf 

class Eval: 

    def __init__(self, env, model_path, video_prefix, number_of_episode=50):
        self.env = env 
        self.model = tf.keras.models.load_model(model_path)
        self.recorder = RecordVideos(video_prefix, 'test_videos/', 15)
        self.number_of_episode = number_of_episode
        self.input_dims = self.env.observation_space.shape[0]
        self.out_dims = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]
        
    def test(self): 
        rewards = []
        steps = []
        for episode in range(self.number_of_episode): 
            done = False
            reward = 0
            step = 0
            state, _ = env.reset(seed=random.randint(0,500))
            if episode % 10 == 0: 
                img = env.render()
                self.recorder.add_image(img) 

            while not done:
                state = tf.reshape(state, (1, self.input_dims))
                _, action =  self.get_action(self.model, state)
                state, reward_prob, terminated, truncated, _ = env.step(action)
                done = terminated or truncated 
                reward += reward_prob
                step += 1 
                if episode % 10 == 0:
                    img = env.render()
                    self.recorder.add_image(img)
            
            rewards.append(reward)
            steps.append(step)
       #     self.recorder.save(episode) if episode % 10 == 0 else None
        
        return rewards, steps
    
    def get_action(self, model, state):
        state = np.reshape(state, [1, self.input_dims])
        mu, std = self.model(state)
        action = np.random.normal(mu[0], std[0], size=self.out_dims)
        action = np.clip(action, -self.action_bound, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)
        return log_policy, action
        
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

model_path = "models/ppo_actor/"
evaluator = Eval(env, model_path, "ppo", 10)
rewards, steps = evaluator.test()

print(rewards, steps)                                                                                                                                                        

