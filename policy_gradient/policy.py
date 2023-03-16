def greedy_policy(observation, model, action_space): 
    state = tf.convert_to_tensor([observation])
    actions = model(state)
    action = tf.math.argmax(actions, axis=1).numpy()[0]
    return action