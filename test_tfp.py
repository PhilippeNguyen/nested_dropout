
import tensorflow_probability as tfp
import tensorflow as tf




#%%
tf.reset_default_graph()
with tf.Session() as sess:
    p1 = 0.4
    p2 = 0.5
    temp = 0.1
    step_size = 0.1
    num_steps = 1000
    test_var = tf.get_variable("test",initializer=tf.constant([p1]))
    sess.run(test_var.initializer)

#    dist = tfp.distributions.RelaxedBernoulli(temperature=temp,
#                                              probs=test_var)
    dist = tfp.distributions.Normal(loc=test_var,scale=1)
    print(dist.reparameterization_type)
    samples = dist.sample(sample_shape=(1000))
    loss = tf.math.reduce_mean(tf.square(samples-p2))
    grad = tf.gradients(loss,test_var)
    assign_val = tf.placeholder(tf.float32,shape=(1,))
    assign = test_var.assign(assign_val)
    for _ in range(num_steps):
        
        out = sess.run([samples,loss,grad,test_var])
        grad_out = out[2][0]
        test_var_val = out[3]
        sess.run(fetches=[assign],feed_dict={assign_val:test_var_val-step_size*grad_out})
        

#    dist = tfp.distributions.Bernoulli(probs=test_var,dtype=tf.float32)
#    dist_05 = tfp.distributions.Bernoulli(probs=[p2],dtype=tf.float32)
    
#    kl = dist.kl_divergence(dist_05)
#    kl_calc = 
    
#    out = sess.run([kl])
        
#%%
tf.reset_default_graph()
with tf.Session() as sess:
    p1 = 0.25
    p2 = 0.5
    temp = 0.1
    step_size = 0.1
    num_steps = 1000
    test_var = tf.get_variable("test",initializer=tf.constant([p1]))
    sess.run(test_var.initializer)

    dist = tfp.distributions.RelaxedBernoulli(temperature=temp,
                                              probs=test_var)
    dist = tfp.distributions.Normal(loc=test_var,scale=1)
    print(dist.reparameterization_type)
    samples = dist.sample(sample_shape=(1000))
    loss = tf.math.reduce_mean(tf.square(samples-p2))
    grad = tf.gradients(loss,test_var)
    assign_val = tf.placeholder(tf.float32,shape=(1,))
    assign = test_var.assign(assign_val)
    for _ in range(num_steps):
        
        out = sess.run([samples,loss,grad,test_var])
        grad_out = out[2][0]
        test_var_val = out[3]
        sess.run(fetches=[assign],feed_dict={assign_val:test_var_val-step_size*grad_out})
        

#    dist = tfp.distributions.Bernoulli(probs=test_var,dtype=tf.float32)
#    dist_05 = tfp.distributions.Bernoulli(probs=[p2],dtype=tf.float32)
    
#    kl = dist.kl_divergence(dist_05)
#    kl_calc = 
    
#    out = sess.run([kl])