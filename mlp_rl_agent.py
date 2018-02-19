import argparse
import sys
import numpy as np
import tensorflow as tf
import gym
from gym import wrappers, logger

class MLPAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, learning_rate=0.01):
        self.action_space = action_space
        n_input = 4
        n_hidden_1 = 5
        n_hidden_2 = 6
        n_classes = action_space.n

        # Set up MLP model
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), trainable=True),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), trainable=True),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), trainable=True)
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1]), trainable=True),
            'b2': tf.Variable(tf.random_normal([n_hidden_2]), trainable=True),
            'out': tf.Variable(tf.random_normal([n_classes]), trainable=True)
        }
        self.state_in = tf.placeholder(shape=(None, n_input), dtype=tf.float32)
        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(self.state_in, self.weights['h1']), self.biases['b1']))
        self.layer_2 = tf.nn.relu(tf.add(tf.matmul(self.layer_1, self.weights['h2']), self.biases['b2']))
        self.outputs = tf.nn.softmax(tf.add(tf.matmul(self.layer_2, self.weights['out']), self.biases['out']))
        self.chosen_action = tf.argmax(self.outputs, axis=0)


        # training procedure
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)  # will contain rewards
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)    # will contain selected actions

        self.indices = tf.range(0, tf.shape(self.outputs)[0])*tf.shape(self.outputs)[1] + self.action_holder # indices (in flattened output array) of outputs corresponding to selected actions
        self.responsible_outputs = tf.gather(tf.reshape(self.outputs, [-1]), self.indices)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder) # scale each output by respective reward

        tvars = tf.trainable_variables() # variables to compute gradients for and update
        self.gradient_holders = [tf.placeholder(tf.float32) for i,tv in enumerate(tvars)]

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

# assign the same reward to every action in an episode
# equal to the total number of timesteps
def process_rewards(rewards, gamma=0.99):
    '''discounted = np.zeros(rewards.shape[0])
    running_sum = 0
    for timestep in reversed(range(0, rewards.shape[0])):
        running_sum = running_sum*gamma + rewards[timestep]
        discounted[timestep] = running_sum'''
    return sum(rewards)

if __name__ == '__main__':
    tf.reset_default_graph()
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('CartPole-v0')

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/mlp_rl-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = MLPAgent(env.action_space)

    episode_count = 1500
    episodes_per_batch = 12
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        total_rewards = []

        batch_ep_histories = [] # batch multiple rollouts together so that rewards can be normalized relative to average performance

        for i in range(episode_count):
            state = env.reset()
            ep_history = []
            running_reward = 0
            done = False
            while not done: # iterate through steps in current episode
                # pick an action
                action_proba = sess.run(agent.outputs, feed_dict={agent.state_in: [state]})[0]
                action = np.random.choice(range(len(action_proba)), p=action_proba)

                next_state, reward, done, _ = env.step(action)
                # print action, reward
                ep_history.append(np.hstack([[action, reward], state]))
                state = next_state
                running_reward += reward
            # print running_reward
            # Compute gradients for policy network based on this rollout
            ep_history = np.array(ep_history) # [[action, reward, state[0], state[1], ...]]
            ep_history[:, 1] = process_rewards(ep_history[:, 1])
            batch_ep_histories.append(ep_history)
            '''grads = sess.run(agent.gradients, feed_dict={
                agent.state_in: ep_history[:, 2:],
                agent.action_holder: ep_history[:, 0],
                agent.reward_holder: ep_history[:, 1]
            })
            for j,grad in enumerate(grads):
                batch_grads[j] += grad'''
            total_rewards.append(running_reward)
            if i % episodes_per_batch == 0 and i > 0: # apply gradients to update policy network
                
                reward_sum = 0 # normalize rewards
                reward_count = 0
                for hist in batch_ep_histories:
                    reward_sum += sum(hist[:,1])
                    reward_count += len(hist[:,1])
                avg_reward = reward_sum/reward_count
                for hist in batch_ep_histories:
                    hist[:, 1] -= avg_reward

                batch_grads = []
                for j,tvar in enumerate(sess.run(tf.trainable_variables())):
                    batch_grads.append(np.zeros(np.shape(tvar)))
                for ep_history in batch_ep_histories:
                    grads = sess.run(agent.gradients, feed_dict={
                        agent.state_in: ep_history[:, 2:],
                        agent.action_holder: ep_history[:, 0],
                        agent.reward_holder: ep_history[:, 1]
                    })
                    for j,grad in enumerate(grads):
                        batch_grads[j] += grad
                sess.run(agent.update_batch, feed_dict=dict(zip(agent.gradient_holders, batch_grads)))
                for grad in batch_grads:# reset gradient sums for next batch
                    grad *= 0
                batch_ep_histories = []
                print i, avg_reward

    # Note there's no env.render() here. But the environment still can open window and
    # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
    # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
