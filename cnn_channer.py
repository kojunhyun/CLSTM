import numpy as np
import pandas as pd
from pandas import DataFrame
import collections
import time
import datetime
import operator
import tensorflow as tf

# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터를 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)

# flag 설정
flags = tf.flags
flags.DEFINE_string("save_path", "ckpt", "checkpoint_dir")
flags.DEFINE_bool("train", True, "should we train or test")
FLAGS = flags.FLAGS

#########################################################################
## data road
dataset = pd.read_csv('rawToRemake_1hours_zero_del.txt', header=0)
p_l = list(dataset.product_no.values)
p_d = collections.defaultdict(lambda: 0)
for p in p_l:
    p_d[p] += 1
sorted_mcd_dict = sorted(p_d.items(), key=operator.itemgetter(1), reverse=True)

#########################################################################
## index to prod, prod to index
mcd_index = collections.defaultdict(lambda: 0)
inverse_mcd_index = mcd_index = collections.defaultdict(lambda: 0)
for i in range(len(sorted_mcd_dict)):
    mcd_index[sorted_mcd_dict[i][0]] = i
    inverse_mcd_index[i] = sorted_mcd_dict[i][0]
#print(mcd_index)
#########################################################################
## input data remake(as image)
start = "2017-01-10"
#start = "2017-12-30"
#end = "2017-03-15"
end = "2018-01-01"
setting_hours = 1

start_hours = datetime.datetime.strptime(start, '%Y-%m-%d')
end_hours = start_hours + datetime.timedelta(hours=setting_hours)
end_date = datetime.datetime.strptime(end, '%Y-%m-%d')

stack_count = 0
while end_date >= end_hours:
    #print(start_hours)
    print(end_hours)
    time_data = dataset[(dataset.date >= start_hours.strftime("%Y-%m-%d %H:%M:%S")) & (dataset.date < end_hours.strftime("%Y-%m-%d %H:%M:%S"))]
    sort_time_data = time_data.sort_values(by=['quantity'], ascending=False)
    #print(sort_time_data)
    multi_one = sort_time_data['product_no'].values[:100]
    
    make_data = np.zeros(2500)

    a_count = 0
    for one in multi_one:
        if mcd_index[one] < 2500:
            make_data[int(mcd_index[one])] = 1
            a_count += 1
    #print('data count : ' ,a_count)        

    remake_data = make_data.reshape(50,50)

    #####stack_data.append(remake_data)

    if stack_count == 0:
        stack_xdata = remake_data
        stack_ydata = make_data
        #print('stack x: ', stack_xdata.shape)
        #print('stack y: ', stack_ydata.shape)
        
    else:
        stack_xdata = np.dstack([stack_xdata, remake_data])
        stack_ydata = np.dstack([stack_ydata, make_data])
        #print('stack x: ', stack_xdata.shape)
        #print('stack y: ', stack_ydata.shape)

    """
    if stack_count == 0:
        stack_data = remake_data
        stack_count += 1
    elif stack_count < 24:
        stack_data = np.dstack([stack_data, remake_data])
        stack_count += 1
    else:
        total_stack_data.append(stack_data)
        
        stack_count = 0
    """
    #print(np.size(stack_data[0][0]))
    stack_count += 1
    start_hours = end_hours
    end_hours = start_hours + datetime.timedelta(hours=setting_hours)
    
    #print(len(total_stack_data))
print('stack x: ', stack_xdata.shape)
print('stack y: ', stack_ydata.shape)

X_data = stack_xdata[:,:,:-1]
Y_data = stack_ydata[:,:,1:]

print('X :', X_data.shape)
print('Y : ', Y_data.shape)


data_slice = int(len(X_data[0][0])*0.7)
print(data_slice)



X_train = X_data[:,:,:data_slice]
Y_train = Y_data[:,:,:data_slice]
X_test = X_data[:,:,data_slice:]
Y_test = Y_data[:,:,data_slice:]
print('x train shape : ', X_train.shape)
print('y train shape : ', Y_train.shape)
print('x test shape : ', X_test.shape)
print('y test shape : ', Y_test.shape)

batch_X_train = []
batch_Y_train = []
batch_X_test = []
batch_Y_test = []
step = 24
print(len(X_train[0][0])/step)
print(int(len(X_train[0][0])/step))
batch_slice = int(len(X_train[0][0])/step)

for bs in range(batch_slice):
    batch_X_train.append(X_train[:,:,bs*step:bs*step + step])
    batch_Y_train.append(Y_train[:,:,bs*step + step:bs*step + step + 1])
    #batch_Y_train.append(Y_train[:,:,bs*step:bs*step + step])
    #print(X_train[:,:,bs*step:bs*step + step].shape)
    #print(Y_train[:,:,bs*step:bs*step + step].shape)

batch_slice = int(len(X_test[0][0])/step)
for bs in range(batch_slice):
    batch_X_test.append(X_test[:,:,bs*step:bs*step + step])
    batch_Y_test.append(Y_test[:,:,bs*step + step:bs*step + step + 1])
    #batch_Y_test.append(Y_test[:,:,bs*step:bs*step + step])
    #print(X_test[:,:,bs*step:bs*step + step].shape)
    #print(Y_test[:,:,bs*step:bs*step + step].shape)

batch_X_train = np.array(batch_X_train)
batch_Y_train = np.array(batch_Y_train)
print('batch x train shape : ', batch_X_train.shape)
print('batch y train shape : ', batch_Y_train.shape)
batch_Y_train = np.reshape(batch_Y_train, (len(batch_Y_train), 2500))
print('batch y train shape : ', batch_Y_train.shape)
batch_X_test = np.array(batch_X_test)
batch_Y_test = np.array(batch_Y_test)
print('batch x test shape : ', batch_X_test.shape)
print('batch y test shape : ', batch_Y_test.shape)
batch_Y_test = np.reshape(batch_Y_test, (len(batch_Y_test), 2500))
print('batch y test shape : ', batch_Y_test.shape)


######################################################################################################################################################
# 학습의 속도와 Epoch, 그리고 데이터를 가져올 batch의 크기를 지정해 줍니다.
learning_rate = 0.001
training_epochs = 300
batch_size = 10
######################################################################################################################################################

X = tf.placeholder(tf.float32,[None, 50, 50, step])
X_img = tf.reshape(X,[-1, 50, 50, step])
Y = tf.placeholder(tf.float32,[None, 50*50])

# CNN에서 우리는 결국 제대로 된 Filter를 가진 Model을 구축해 나갈 것입니다.
# 다시 말해 Filter를 학습시켜 이미지를 제대로 인식하도록 할 것입니다.
# 그렇기에 Filter를 변수 W로 표현합니다.
# 32개의 3 x 3 x 1의 Filter를 사용하겠다는 뜻입니다.
W1 = tf.Variable(tf.random_normal([3,3,step,32], stddev=0.01))

# 간단하게 conv2d 함수를 사용하면 됩니다.

L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')

L1 = tf.nn.relu(L1)

L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# W2는 두번째 Conv의 Filter입니다.
# 다만 이전 과정 Filter의 개수가 32개였기 때문에
# 그 숫자에 맞추어 depth를 32로 지정해줍니다.
 
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))

L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')

L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Softmax를 통한 FC layer를 활용하기 위해 shape를 변환해줍니다.
# 위에서 L2는 일종의 X 값이라고도 볼 수 있습니다.
# Softmax를 거칠 예측값(WX+b)을 만들어주기 위해 reshape 합니다.
 
L2 = tf.reshape(L2, [-1, 13*13*64])


# W3를 설정하는데 Xavier initializing을 통해 초기값을 설정할 것입니다.
# reshape된 L2의 shape이 [None, 7*7*64] 였으므로
# W3의 shape은 [7*7*64, num_label]이 됩니다.
 
W3 = tf.get_variable("W3", shape=[13*13*64, 2500], initializer = tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([2500]))

hypothesis = tf.matmul(L2, W3) + b




y_sorted_index = tf.nn.top_k(Y, k=100)
h_sorted_index = tf.nn.top_k(hypothesis, k=100)


################################################
########## 수정 예정(maximize하는 방법으로)
# Softmax 함수를 직접 사용하는 대신에 sofmax_corss_entropy_with_logits을 사용할 수 있습니다.
# 인자로 logits과 label을 전달해주면 됩니다.

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=hypothesis, labels=Y))
 
# 이전까지는 Gradient Descent Optimizer를 사용하였지만
# 좀 더 학습성과가 뛰어나다고 알려져있는 Adam Optimizer를 사용하겠습니다.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
################################################

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


"""
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(batch_X_train) / batch_size)
    print('total_batch : ', total_batch)
    for i in range(total_batch):
        batch_xs = batch_X_train[i*batch_size:i*batch_size+batch_size]
        print(batch_xs.shape)
        batch_ys = batch_Y_train[i*batch_size:i*batch_size+batch_size]
        print(batch_ys.shape)
        feed_dict = {X:batch_xs, Y:batch_ys}

        w3, h = sess.run([W3, hypothesis], feed_dict=feed_dict)
        print(w3.shape)
        print(h.shape)

"""


if FLAGS.train:
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(len(X_train) / batch_size)
        #print('total_batch : ', total_batch)

        total_avg_pred = 0.0
        for i in range(total_batch):
            batch_xs = batch_X_train[i*batch_size:i*batch_size+batch_size]
            print(batch_xs.shape)
            batch_ys = batch_Y_train[i*batch_size:i*batch_size+batch_size]
            print(batch_ys.shape)

            feed_dict = {X:batch_xs, Y:batch_ys}

            c, hs, ys, _ = sess.run([cost, h_sorted_index, y_sorted_index, optimizer], feed_dict=feed_dict)
            #hs, ys = sess.run([h_sorted_index, y_sorted_index], feed_dict=feed_dict)
            #print('cost : ', c)
            hs = np.array(hs)
            ys = np.array(ys)
            print('hs : ', hs.shape)
            

            avg_pred = 0.0
            for j in range(len(hs[1])):
                prediction_value = 0.0
                for hs_index in hs[1][j]:
                    if hs_index in ys[1][j]:
                        prediction_value += 1.0
                avg_pred += (prediction_value/100)
            avg_pred = avg_pred/len(hs[1]) 
            print('avg_pred : ', avg_pred)
            total_avg_pred += avg_pred
            avg_cost += c / total_batch
        total_avg_pred = total_avg_pred/total_batch
        print('total avg pred : ', total_avg_pred)



            
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
        saver.save(sess, FLAGS.save_path + '_model', global_step=epoch+1)

    total_avg_pred = 0.0
    total_batch = int(len(X_test) / batch_size)
    for i in range(total_batch):
        batch_xs = batch_X_test[i*batch_size:i*batch_size+batch_size]
        batch_ys = batch_Y_test[i*batch_size:i*batch_size+batch_size]
        feed_dict = {X:batch_xs, Y:batch_ys}
        hs, ys = sess.run([h_sorted_index, y_sorted_index], feed_dict=feed_dict)
        hs = np.array(hs)
        ys = np.array(ys)
        avg_pred = 0.0
        for j in range(len(hs[1])):
            prediction_value = 0.0
            for hs_index in hs[1][j]:
                if hs_index in ys[1][j]:
                    prediction_value += 1.0
            #print(prediction_value/100)
            avg_pred += (prediction_value/100)
        avg_pred = avg_pred/len(hs[1]) 
        print('avg_pred : ', avg_pred)
        total_avg_pred += avg_pred
    total_avg_pred = total_avg_pred/total_batch
    print('total avg pred : ', total_avg_pred)

    print('Learning Finished!')

else:
    print('Testing started. It takes sometime.')
    ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print ("No checkpoint file found")


    #total_batch = int(len(X_test) / batch_size)
    total_batch = int(len(X_data) / batch_size)
    total_avg_pred = 0.0
    for i in range(total_batch):
        batch_xs = X_data[i*batch_size:i*batch_size+batch_size]
        #batch_xs = X_test[i*batch_size:i*batch_size+batch_size]
        #print(batch_xs.shape)
        #batch_ys = Y_test[i*batch_size:i*batch_size+batch_size]
        batch_ys = Y_data[i*batch_size:i*batch_size+batch_size]
        #print(batch_ys.shape)

        feed_dict = {X:batch_xs, Y:batch_ys}
        hs, ys = sess.run([h_sorted_index, y_sorted_index], feed_dict=feed_dict)
        
        hs = np.array(hs)
        ys = np.array(ys)

        avg_pred = 0.0
        
        for j in range(len(hs[1])):
            prediction_value = 0.0
            for hs_index in hs[1][j]:
                if hs_index in ys[1][j]:
                    prediction_value += 1.0
            #print(prediction_value/100)
            avg_pred += (prediction_value/100)
        avg_pred = avg_pred/len(hs[1]) 
        print('avg_pred : ', avg_pred)
        total_avg_pred += avg_pred
    total_avg_pred = total_avg_pred/total_batch
    print('total avg pred : ', total_avg_pred)
    print('Testing Finished!')

