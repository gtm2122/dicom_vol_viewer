import multiprocessing
import scipy.misc

class Pro:
    def __init__(self,save_dir):
        self.save_dir = save_dir
        #self.name = name
    def __call__(self,arr,name):
        scipy.misc.imsave(self.save_dir+'/'+name,arr)
        
def fun(args):
    scipy.misc.imsave(*args)