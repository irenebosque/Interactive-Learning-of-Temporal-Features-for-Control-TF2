import numpy as np
import time
from main_init import neural_network, transition_model, transition_model_type, agent, agent_type, exp_num,count_down, \
    max_num_of_episodes, env, render, max_time_steps_episode, human_feedback, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model


"""
Main loop of the algorithm described in the paper 'Interactive Learning of Temporal Features for Control' 
"""

# Initialize variables
total_feedback, total_time_steps, trajectories_database = [], [], []
t_total, h_counter, last_t_counter, omg_c, eval_counter = 1, 0, 0, 0, 0
human_done, evaluation, random_agent, evaluation_started = False, False, False, False

init_time = time.time()

# Print general information
print('\nExperiment number:', exp_num)
print('Environment: Carla')
print('Learning algorithm: ', agent_type)
print('Transition Model:', transition_model_type, '\n')

time.sleep(2)

# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)

# Start training loop
for i_episode in range(max_num_of_episodes):
    print('Starting episode number', i_episode)

    if not evaluation:
        transition_model.new_episode()

    observation = env.reset()  # reset environment at the beginning of the episode

    past_action, past_observation, episode_trajectory, h_counter = None, None, [], 0  # reset variables for new episode

    # Iterate over the episode
    for t in range(int(max_time_steps_episode)):
        # Make the environment visible and delay
        if render:
            env.render()
            time.sleep(render_delay)

        # Get feedback signal
        h = human_feedback.get_feedback()

        # Ask human for done
        human_done = human_feedback.ask_for_done()

        # Feed h to agent
        agent.feed_h(h)

        # Map action from observation
        state_representation = transition_model.get_state_representation(neural_network, observation)
        action = agent.action(neural_network, state_representation)

        # Act
        observation, _, environment_done, _ = env.step(action)

        # Compute new hidden state of LSTM
        transition_model.compute_lstm_hidden_state(neural_network, action)

        # Append transition to database
        if not evaluation:
            if past_action is not None and past_observation is not None:
                episode_trajectory.append([past_observation, past_action, transition_model.processed_observation])  # append o, a, o' (not really necessary to store it like this)

            past_observation, past_action = transition_model.processed_observation, action

            if t % 100 == 0 or environment_done:
                trajectories_database.append(episode_trajectory)  # append episode trajectory to database
                episode_trajectory = []

        if np.any(h):
            h_counter += 1

        # Compute done
        done = environment_done or human_done

        # Update weights transition model/policy
        if not evaluation:
            if done:
                t_total = done  # tell the agents that the episode finished

            transition_model.train(neural_network, t_total, done, trajectories_database)
            agent.train(neural_network, transition_model, action, t_total, done)

            t_total += 1

        # End of episode
        if done:
            if evaluation:
                print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
                total_feedback.append(h_counter/(t + 1e-6))
                total_time_steps.append(t_total)
                if save_results:
                    np.save(eval_save_path + exp_num + '_feedback', total_feedback)
                    np.save(eval_save_path + exp_num + '_time', total_time_steps)

            if save_policy:
                neural_network.save_policy()

            if save_transition_model:
                neural_network.save_transition_model()

            if render:
                time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))
            break
