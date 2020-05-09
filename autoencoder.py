import tensorflow as tf
import numpy as np
from datetime import datetime
from utils import get_assignment_map_from_checkpoint
class AutoEncoder:
    def __init__(self,n_alphas):
        self.n_alphas = n_alphas
    def calc_loss(self,frd_res):
        out,abs_weight = frd_res
        loss = -tf.reduce_mean(out)/tf.keras.backend.std(out) * abs_weight
        return loss


    def forward(self,x):
        with tf.variable_scope("out_layer"):
            self.weight = tf.get_variable("weight",shape=[self.n_alphas,1])
            out = tf.matmul(x,self.weight)
            abs_weight = tf.reduce_sum(tf.math.abs(self.weight))
        return out,abs_weight


    def inference(self):
        return self.weight


class CustomizeTrainer:
    def __init__(self,AE):
        self.AE = AE
        self.last_processed = 0
        self.processed = 0
        self.window_loss = 0
        self.learning_rate = 0.01
        self.log_frequency = 2
        self.inp = tf.placeholder(tf.float32,shape=[None,n_alphas])
        self.forward_res = self.AE.forward(self.inp)
        self.loss = self.AE.calc_loss(self.forward_res)
        self.batch_size = 16
        self.init_status=True
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        #optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.opt = optimizer.minimize(self.loss)
        self.infer_res = self.AE.inference()
    def initialize(self):
        return [tf.global_variables_initializer()]
    def train_ops(self):
        return [self.opt,self.loss]
    def print_log(self,step):
        new_sample = self.processed - self.last_processed
        avg_loss = self.window_loss / new_sample
        progress = self.processed / self.total_sample * 100
        self.window_loss = 0
        self.last_processed = self.processed
        format_str = "%s: step %d, progress: %.2f, %d sample processed, avg_loss= %.1f"
        print(format_str%(datetime.now(),step,progress,self.processed,avg_loss))
    def set_datainfo(self,n_sample):
        self.total_sample = n_sample
    def train(self,dataset):
        self.sess.run(self.initialize())
        n_sample,n_alphas = dataset.shape
        self.set_datainfo(n_sample)
        for step,i in enumerate(range(0,n_sample,self.batch_size)):
            st,en = i,n_sample if self.batch_size+i > n_sample else self.batch_size+i
            _,loss_step = self.sess.run(self.train_ops(),feed_dict={self.inp:dataset[st:en]})
            self.window_loss += loss_step
            self.processed += en-st
            if step % self.log_frequency == 1:
                self.print_log(step)
    def save(self,path):
        self.saver.save(self.sess, path +"/model")
    def load(self,path):
        ckpt = tf.train.get_checkpoint_state(path)
        inited = False
        if ckpt and ckpt.model_checkpoint_path:
            tvars = tf.trainable_variables()
            assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, ckpt.model_checkpoint_path)
            tf.train.init_from_checkpoint(ckpt.model_checkpoint_path,assignment_map)
            print("Load model from ", ckpt.model_checkpoint_path)
            inited = True
        else:
            print("No Initial Model Found.")
        if inited and self.init_status:
            print("**** Trainable Variables ****")
            #tvars = tf.trainable_variables()
            #initialized_variable_names = []
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                print("  name = "+var.name+" shape = ",var.shape, init_string)
    def get_weight(self):
        return self.sess.run(self.AE.inference())


if __name__=='__main__':
    #Settings
    #For Test
    input_data = np.random.rand(1000,10)
    n_sample,n_alphas = input_data.shape
    path = "./model/"
    #####
    AE = AutoEncoder(n_alphas)
    trainer = CustomizeTrainer(AE)
    trainer.load(path)
    trainer.train(input_data)
    trainer.save(path)
    print(trainer.get_weight())

