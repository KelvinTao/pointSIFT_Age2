
def preTrain(checkpoint_path):
    model_path_restore = checkpoint_path + '.ckpt'
    dataset = get_record_dataset(record_path=Config.record_path,     
         num_samples=Config.num_samples,num_classes=Config.num_classes)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    image, label = data_provider.get(['image', 'label'])
    # print('image:',image)
    image, label = processing_image(image,label)
    images, labels = tf.train.batch([image, label], batch_size=Config.BATCH_SIZE, num_threads=1, capacity=5)
 
    logist = Model(images, is_training=True, num_classes=Config.num_classes)
 
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logist)
    loss = tf.reduce_mean(cross_entropy)
 
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logist, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
    # # 量化
    graph = tf.get_default_graph()
    create_training_graph(graph, 40000)
 
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate=Config.learning_rate,
                                    global_step=tf.cast(tf.div(global_step, 40),
                                    tf.int32),
                                    decay_steps=Config.decay_steps,
                                    decay_rate=Config.decay_rate,
                                    staircase=True)
 
    # lr = 0.001
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
 
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss,
          global_step=global_step)
 
    global_init = tf.global_variables_initializer()
    total_step = Config.NUM_EPOCH * Config.num_samples // Config.BATCH_SIZE
 
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=12)
 
    with tf.Session(config=gpuConfig) as sess:
        # rimages, rlabels = sess.run([images, labels])
        # print('--------rimages:---------',rimages)
        # init = tf.initialize_local_variables()
        # sess.run([init])
        sess.run([global_init])
        #加载预训练模型
        saver.restore(sess, model_path_restore)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
 
        try:
            for step in range(1, total_step + 1):
                if coord.should_stop():
                    break
 
                _, loss_val, accuracy_val, global_step_test = sess.run([train_op, loss, 
                                                              accuracy, global_step])
                lr_val = sess.run(lr)
                print('global_step',global_step_test)
                print("step: %d,lr: = %.5f,loss = %.5f,accuracy =%.5f" % (step, lr_val, 
                                                         loss_val, accuracy_val))
                #模型保存
                if (step == 1):
                    tf.train.write_graph(sess.graph, Config.pb_path, "handcnet.pb")
                if step % 200 == 0:
                    saver.save(sess, Config.ckpt_path + 
                    "step_%d_loss_%.5f_acc_%.5f.ckpt" % (step, loss_val, accuracy_val))
                    print('Save for ', Config.ckpt_path + "step_%d_loss_%.5f.ckpt" % 
                                                                (step, loss_val))
 
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
 
        coord.join(threads)
 
        saver.save(sess, Config.ckpt_path + "completed_model.ckpt" % loss_val)
 
        tf.train.write_graph(sess.graph, Config.pb_path, "handcnet.pb")
 
        print("train completed!")
 

————————————————
版权声明：本文为CSDN博主「LiangJun.py」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_26535271/article/details/99438945