# Code adapted from: https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC/tf2

import json
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from environment import register_baba_env
import gym
from gym import wrappers

from utils import train_test_levels


chkpt_dir = "checkpoints"

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size))
        self.terminal_memory= np.zeros(self.mem_size, dtype= np.bool)

    def store_transition(self,state,action,reward, new_state, done):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_counter +=1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        # TODO: weighted sampling

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminuses = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminuses

    
class CriticNetwork(models.Model):
    def __init__(self, n_actions, dense_units=[256, 256], name="critic", checkpoint_dir=chkpt_dir, **kwds):
        super().__init__(name=name,**kwds)
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")

        self.components = [layers.Dense(dims, activation="relu") for dims in dense_units] + [layers.Dense(1, activation=None)]

    def call(self, state, action):
        x = tf.concat([state,action], axis=1)
        for l in self.components:
            x = l(x)
        return x

class ValueNetwork(models.Model):
    
    def __init__(self, dense_units=[256,256], name="value", checkpoint_dir=chkpt_dir, **kwds):
        super().__init__(name=name,**kwds)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")

        self.components = [layers.Dense(dims, activation="relu") for dims in dense_units] + [layers.Dense(1, activation=None)]

    def call(self, state):
        x = state
        for l in self.components:
            x = l(x)
        return x


class ActorNetwork(models.Model):
    def __init__(self, max_action,n_actions=2, dense_units=[256,256], name="actor", checkpoint_dir=chkpt_dir, **kwds):
        super().__init__(name=name,**kwds)
        self.max_action = max_action
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")
        self.noise = 1e-6

        self.backbone_layers = [layers.Dense(d, activation="relu") for d in dense_units]
        self.sigma = layers.Dense(self.n_actions, activation=None)
        self.mu = layers.Dense(self.n_actions, activation=None)
        
    def call(self, state):
        x = state
        for l in self.backbone_layers:
            x = l(x)
        
        mu = self.mu(x)
        sigma = self.sigma(x)

        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma
    
    def sample_normal(self, state):
        mu, sigma = self.call(state)
        probs = tfp.distributions.Normal(mu,sigma)
        actions = probs.sample()
        action = tf.math.tanh(actions)*self.max_action
        log_probs =probs.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action,2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

    
class Agent:
    def __init__(self,
        alpha=3e-4,
        beta=3e-4,
        input_dims=[8],
        env=None,
        gamma=0.99,
        n_actions=4,
        max_replay_size=10000,
        tau=5e-3,
        layer_sizes=[256,256],
        batch_size=128,
        reward_scale=2):

        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_replay_size, input_dims,n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions, max_action=1)
        self.critics = [CriticNetwork(n_actions=n_actions, name=f"critic_{i}") for i in range(1,3)]
        self.value = ValueNetwork()
        self.target_value = ValueNetwork(name="target_value")

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        for model in self.critics + [self.value, self.target_value]:
            model.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        state = tf.convert_to_tensor([obs])
        actions, _ = self.actor.sample_normal(state)

        return actions[0].numpy()

    def remember(self, state,action,reward, new_state,done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        targets = self.target_value.weights
        weights = [weight * tau + targets[i] * (1-tau) for i, weight in enumerate(self.value.weights)]
        self.target_value.set_weights(weights)

    def save_models(self):
        models = [self.actor, *self.critics, self.value, self.target_value]
        for model in models:
            model.save_weights(model.checkpoint_file)
    
    def load_models(self):
        models = [self.actor, *self.critics, self.value, self.target_value]
        for model in models:
            model.load_weights(model.checkpoint_file)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # Update Value Network
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(new_states), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs,1)
            critic_policies = [critic(states, current_policy_actions) for critic in self.critics]
            critic_value = tf.squeeze(tf.math.minimum(*critic_policies), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, 
                                                self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
                       value_network_gradient, self.value.trainable_variables))

        # Update Actor Network
        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            critic_policies = [critic(states, new_policy_actions) for critic in self.critics]
            critic_value = tf.squeeze(tf.math.minimum(*critic_policies), 1)
        
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, 
                                            self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
                        actor_network_gradient, self.actor.trainable_variables))
        
        # Update Critic Networks
        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            losses = []
            for critic in self.critics:
                old_policy = tf.squeeze(critic(state,action), 1)
                loss = keras.losses.MSE(old_policy, q_hat) * 0.5
                losses.append(loss)
        gradients = [tape.gradient(loss, critic.trainable_variables) for loss, critic in zip(losses, self.critics)]
        for grad, critic in zip(gradients, self.critics):
            critic.optimizer.apply_gradients(zip(grad, critic.trainable_variables))

        self.update_network_parameters()
        


if __name__ == '__main__':
    env_name = "baba_is_you-v1"
    train, test = train_test_levels()
    env_template = register_baba_env(env_name, path=train[0], enable_render=False, env_class_str="PropertyBasedEnv")
    env = gym.make(env_name)
    env.reset()
    flattened_input_shape = (np.prod(env.observation_space.shape),)
    agent = Agent(input_dims=flattened_input_shape, env=env,
            n_actions=env.action_size)
    n_games = 100
    should_train = False
    # uncomment this line and do a mkdir tmp && mkdir tmp/video if you want to
    # record video of the agent playing the game.
    #env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)

    best_score = env.reward_range[0]
    
    load_checkpoint = True
    level_scores = {}
    if load_checkpoint:
        agent.load_models()
    if should_train:
        print("Train")
        for i,level in enumerate(train):
            score_history = []
            env_name = f"baba-train{i}-v0"
            env_template = register_baba_env(env_name, path=level, enable_render=False, env_class_str="PropertyBasedEnv")
            env = gym.make(env_name)
            for j in range(5):
                observation = (env.reset()).flatten()
                done = False
                score = 0
                while not done:
                    action = agent.choose_action(observation)
                    action = np.argmax(action)
                    next_obs, reward, done, info = env.step(action)
                    next_obs = next_obs.flatten()
                    if env.enable_render:
                        env.render()
                    score += reward
                    agent.remember(observation, action, reward, next_obs, done)
                    agent.learn()
                    observation = next_obs
                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
            level_scores[level] = score_history
            print(f"Training {i} - Score: {score} - Avg: {avg_score}")
            agent.save_models()
        with open("soft_actor_critic_train.json", "w") as f:
            json.dump(level_scores, f)

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
    level_performance = {}
    print("test")
    for i,level in enumerate(test):
        env_name = f"test-test{i}-v0"
        env_template = register_baba_env(env_name, path=level, enable_render=True, env_class_str="PropertyBasedEnv")
        env = gym.make(env_name)
        observation = (env.reset()).flatten()
        reward = 0
        score = 0
        steps = 0
        done = False
        while not done:
            action = agent.choose_action(observation)
            action = np.argmax(action)
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs.flatten()
            if env.enable_render:
                env.render()
            score += reward
            steps +=1
            agent.remember(observation, action, reward, next_obs, done)
            observation = next_obs
        level_performance[level] = {"score": score, "steps": steps, "won": reward > 0}
        print(f"Running test {i}")
    with open("Results/soft_actor_critic_results.json", "w") as f:
            json.dump(level_performance, f)

            

    






