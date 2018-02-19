import argparse
import sys
import numpy as np
import tensorflow as tf
import gym
from gym import wrappers, logger

class MyAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        n_input = 4
        n_hidden_1 = 32
        n_hidden_2 = 32
        n_classes = action_space.n

        # Set up MLP model
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        self.state_in = tf.placeholder(shape=(None, n_input), dtype=tf.float32)
        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(self.state_in, self.weights['h1']), self.biases['b1']))
        self.layer_2 = tf.nn.relu(tf.add(tf.matmul(self.state_in, self.weights['h2']), self.biases['b2']))
        self.output_logits = tf.add(tf.matmul(self.state_in, self.weights['out']), self.biases['out'])
        self.chosen_action = tf.argmax(self.output_logits)


        # training procedure
        self.reward_holder = tf.placeholder(shape=(None), dtype=tf.float32)  # will contain rewards
        self.action_holder = tf.placeholder(shape=(None), dtype=tf.int32)    # will contain selected actions

        self.indices = tf.range(0, tf.shape(output_logits)[0])*tf.shape(output_logits)[1] + self.action_holder # indices (in flattened output array) of outputs corresponding to selected actions
        self.responsible_outputs = tf.gather(tf.reshape(output_logits, -1), indices)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder) # scale each output by respective reward

        tvars = tf.trainable_variables() # variables to compute gradients for and update
        self.gradient_holders = [tf.placeholder(tf.float32) for i,tv in enumerate(tvars)]

        self.gradients = tf.gradients(self.loss, tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

    def act(self, observation, reward, done):
        print observation, reward, done
        print self.action_space
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = MyAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
