from keras import layers , models , optimizers 
from keras import backend as K
import random
from collections import namedtuple,deque
import numpy as np

class OUNoise:
    def __init__(self,size,nu,theta,sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    def reset(self):
        self.state = self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x+ dx
        return self.state


class Actor:
    def __init__(self,state_size,action_size,action_low , action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high
        self.action_low = action_low
        self.action_range = action_high - action_low
        self.model()
        
    def model(self):
        states = layers.Input(shape=(self.state_size,),name='states')
        layer1 = layers.Dense(units=32,activation='relu')(states)
        layer2 = layers.Dense(units=64,activation='relu')(layer1)
        layer3 = layers.Dense(units=32,activation='relu')(layer2)
        
        norm_action = layers.Dense(units=self.action_size,activation='sigmoid',name='norm_action')(layer3)
        output_actions = layers.Lambda(lambda x : (x * self.action_range) + self.action_low,name='output_actions')(norm_action)
        
        self.model = models.Model(inputs=states,outputs=output_actions)
        
        action_gradients = layers.Input(shape=(self.action_size,))
        
        loss = K.mean(-action_gradients * output_actions)
        
        optimizer = optimizers.Adam()
        
        train = optimizer.get_updates(params=self.model.trainable_weights,loss=loss)
        self.train_fn = K.function(inputs=[self.model.input,action_gradients,K.learning_phase()],
                                  outputs=[],
                                  updates=train)
        


class Critic:
    def __init__(self,state_size,action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model()
        
    def model(self):
        states = layers.Input(shape=(self.state_size,),name='states')
        actions = layers.Input(shape=(self.action_size,),name='actions')
        
        layer1 = layers.Dense(units=32,activation='relu')(states)
        layer2 = layers.Dense(units=64,activation='relu')(layer1)
        
        action1 = layers.Dense(units=32,activation='relu')(actions)
        action2 = layers.Dense(units=64,activation='relu')(action1)
        
        
        output = layers.Add()([layer2,action2])
        output = layers.Activation('relu')(output)
        
        q = layers.Dense(units=1,name='q_value')(output)
        self.model = models.Model(inputs=[states,actions],outputs=q)
        
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer,loss='mse')
        action_gradients = K.gradients(q,actions)
        
        self.get_action_gradients = K.function(
        inputs=[*self.model.input,K.learning_phase()],
        outputs=action_gradients)
        
class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self,batch_size=64):
        return random.sample(self.memory,k = batch_size)
    def __len__(self):
        return len(self.memory)
    
        
class PolicyGradient():
    def __init__(self,task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        self.actor_local = Actor(self.state_size, self.action_size , self.action_low , self.action_high)
        self.actor_target = Actor(self.state_size , self.action_size, self.action_low,self.action_high)
        
        self.critic_local = Critic(self.state_size,self.action_size,self.action_low,self.action_high)
        self.critic_target = Critic(self.state_size,self.action_size , self.action_low,self.action_high)
        
        self.critic_target = Critic(self.state_size,self.action_size,self.action_low,self.action_high)
        self.exploation_mu = 0
        self.exploation_theta = 0.15
        self.exploration_sigma = 0.001
        self.noise = OUNoise(self.action_size,self.exploration_mu,self.exploration_theta,self.exploration_mu,self.exploration_sigma)
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size,self.batch_size)
        
        self.gamma = 0.99
        self.tau = 0.1
        self.best_score = -np.inf
        self.score = 0
        
                
        
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        self.score = 0
        return state
    def step(self,action,reward,next_state,done):
          self.memory.add(self.last_state,action,reward,next_state,done)
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
            self.last_state = next_state
            self.score += reward
            if done:
                if self.score > self.best_score:
                    self.best_score= self.score
    def act(self,states):
        state = np.reshape(states,[-1,self.state])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())
    
    def learn(self,experiences):
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1,self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1,1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        next_actions = self.actor_target.model.predict_on_batch(next_states)
        q_next = self.critic_target.model.predict_on_batch([next_states,next_actions])
        q_current = rewards + self.gamma  + q_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states,actions],y=q_current)
        
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states,actions,0]),(-1,self.action_size))
        
        self.actor_local.train_fn([states,action_gradients,1])
        
        self.soft_update(self.critic_local.model,self.critic_target.model)
        self.soft_update(self.actor_local.model,self.actor_target.model)
        
        
    def soft_update(self,local_model,target_model):
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        assert len(local_weights) == len(target_weights), "size differ"
        
        new_weights = self.tau * local_weights  + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
        
        