import tkinter as tk
import numpy as np
import os
import pydicom
from load_scan_dir import *
from PIL import ImageTk as itk
import PIL
import pickle

from decomp_dcm import *
# root = tk.Tk()

# v = tk.IntVar()
# v.set(0)


# next_var = tk.IntVar()
# prev_var = tk.IntVar()

# next_var.set(0)
# next_var.set(0)


# val_to_tuple = {}


# vol = []


import scipy.misc
class GUI():
    def __init__(self,file_path = '/data/gabriel/TAVR/TAVR_Sample_Study' ):
    
        self.file_path = file_path
        self.vol = []
        self.val_to_tuple = {}
        self.counter = -1
        
        #self.df  =
        self.group_df = return_grouped(group_scans_df(self.file_path))
        self.df = group_scans_df(self.file_path)
        
        
        self.root = tk.Tk()
        
        self.v = tk.IntVar()
        self.v.set(0)

        self.next_var = tk.IntVar()
        self.prev_var = tk.IntVar()

        self.next_var.set(0)
        self.prev_var.set(0)

        # get all vols:
        self.welcome_logo = itk.PhotoImage(PIL.Image.open('welcome.png'))
        self.label  = tk.Label(master=self.root,image=itk.PhotoImage(PIL.Image.open('welcome.png')))    
        self.label.pack(side='left')

        self.b1 = tk.Radiobutton(master=self.root,text='Next',indicatoron=False,variable=self.next_var,command=self.next_b)
        self.b2 = tk.Radiobutton(master=self.root,text='Prev',indicatoron=False,variable=self.prev_var,command=self.prev_b)
        self.b1.pack()
        self.b2.pack()
        
        
        all_keys =list( self.group_df.groups.keys())
        self.w = tk.Message(master=self.root,text = "")
        for i,j in enumerate(all_keys):
            vol_idx = i

            ser_num = j[0]
            acq_num = j[1]
            ph_num  = j[2]
            man_name = j[3]
            vol=return_volume(self.df,self.group_df,ser_num,acq_num,ph_num,man_name)
            #print(vol)
            num_vols = len(vol)
            num_images = [len(v) for v in vol]

            #num_images.append(len(vol[-1]))
            button_text = (ser_num,acq_num,ph_num,num_images,man_name)
            
            for vol_id in range(0,num_vols):
                #self.v.set(0)
                #print(man_name[0])
                vol_id1 = vol_id +  i
            
                button_text1 = (vol_id1,ser_num,acq_num,ph_num,num_images[vol_id],man_name)

                #print(button_text1)
                
                tk.Radiobutton(master=self.root,\
                               text=button_text1
                               ,padx=20,\
                               variable=self.v,\
                               command=self.load_volume,\
                               value=i,\
                               indicatoron=False).pack(anchor=tk.W)

                self.val_to_tuple[vol_id1]=button_text1
                
                
        self.annotate_button = tk.Radiobutton(master=self.root,\
                                        text="Annotate",\
                                        padx=20,\
                                        command = self.annotate_func,\
                                        value=0,\
                                         indicatoron=False
                                        ).pack(anchor=tk.SW)

        self.root.mainloop()
        
    def annotate_func(self):
        print("Not implemented")
        
    def load_volume(self):
        
        
        
        self.counter=-1
        req_tuple = self.val_to_tuple[self.v.get()]
        self.vol = return_volume(self.df,self.group_df,req_tuple[1],req_tuple[2],req_tuple[3],req_tuple[5])
        #print(len(vol))
        
        self.vol = self.vol[req_tuple[0]%len(self.vol)]
        #self.v.set(0)
        self.next_b()
        
        self.w.configure(text = "viewing  Vol Idx %s, Series %s, Acq Num %s"%(req_tuple[1],req_tuple[2],req_tuple[3]))
        self.w.pack()
    def next_b(self):
        #self.w.destroy()
        
        self.next_var.set(1)
        self.counter+=1
        
        if self.counter == len(self.vol):
            self.counter=0

        np_img = decomp(self.vol[self.counter][0])
         
        scipy.misc.imsave('temp1.jpg',np_img)
        self.next_var.set(0)
        photo= itk.PhotoImage( PIL.Image.open('temp1.jpg'))
        self.label.config(image=photo)

        self.label.image=photo
        
        self.label.pack()
        
        
    def prev_b(self):
        self.prev_var.set(1)
        self.counter-=1
        
        np_img = decomp(self.vol[self.counter][0])
        
        scipy.misc.imsave('./temp1.jpg',np_img)
        self.prev_var.set(0)
        photo= itk.PhotoImage(PIL.Image.open('temp1.jpg'))
        self.label.config(image=photo)

        self.label.image=photo
        
        self.label.pack()

        

if __name__ =="__main__":
    gui = GUI(file_path = '/data/gabriel/TAVR/TAVR_Sample_Study')