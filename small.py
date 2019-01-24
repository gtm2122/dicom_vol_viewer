import tkinter as tk
from multiprocessing import Process,Queue
from queue import Empty
import pickle
from tkinter.ttk import Progressbar
import os
import multiprocessing
#import scipy.misc
import numpy as np
#import pandas as pd 
import pydicom 
def count_fn(k):
    c=k
    
    
    
    
    while c < 2**16:
        c+=0.01
    pickle.dump(c,open('result1.pkl','wb'))
    a = np.zeros((512,512),dtype=np.uint16)
    #scipy.misc.imsave('a.png',a)
    p = 'D:\Anaconda3\project\TAVR\TAVR_MAIN\TAVR_ROOT\MR\download20181120111300\m rob\\39761136\\37146972'
    b = pydicom.dcmread(p+'/00001_2.16.840.1.113669.632.21.1779766956.20181113.406845142.0.16.2036.dcm')
    pickle.dump(b.pixel_array,open('result2.pkl','wb'))
    
    
class gui():
    def __init__(self,tk_obj,num):
        self.num = num

        self.root = tk_obj
        
        self.button = tk.Button(master = self.root,text='count',command=self.call_count)
        self.button.pack(side='top')
        self.v1=tk.IntVar()
        self.v1.set(1)

        if os.path.isfile('result1.pkl'):
            os.remove('result1.pkl')
            
        self.pbar = Progressbar(master=self.root,mode='indeterminate')
        #self.pbar.grid(row=1,column=1)
        self.pbar.pack(side='top')
    def call_count(self):
        # caller that starts a process and call a top level function
        self.v1.set(5)
        count_fn(self.num)
        self.p1 = Process(target=count_fn,args=(self.num,))
        
        self.p1.start()
        self.pbar.start(1)
        self.pbar.after(1,self.onGetValue)
    
    def onGetValue(self):
        #if not os.path.isfile('result1.pkl'):
        if self.p1.is_alive():
            self.pbar.after(1,self.onGetValue)
            return
        else:
            print('DONE')
            self.pbar.stop()
            
        
if __name__ == "__main__":
    multiprocessing.freeze_support()
    a = tk.Tk()
    num = 10
    call_gui = gui(a,0)
    a.mainloop()
    