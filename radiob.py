import tkinter as tk
#import Tkinter as tk
import numpy as np
import os
import pydicom
#import load_scan_dir as lsd
import pandas as pd
from load_scan_dir import return_volume,load_all_dcm,read_phase
from PIL import ImageTk as itk
import PIL
from functools import partial
import pickle
import scipy.misc
from decomp_dcm import *
import load_scan_dir
import time
import time
#import mp_test
import threading
import multiprocessing
from multiprocessing import Pool
import scipy.ndimage
import tkinter.ttk as ttk
#import ttk
import subprocess
#import png
from PIL import Image
from multiprocessing import Queue,Process
from queue import Empty
from decimal import Decimal,getcontext
from tkinter.ttk import Progressbar
import os
import sys
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt 
import warnings
#from tkinter import Entry,StringVar
from tkinter import Entry,StringVar
from scipy.sparse.csgraph import _validation
from scipy.special import _ufuncs_cxx
from matplotlib.backends import backend_tkagg


def save_imgs(image_save_path,decomp_img):

    for i in range(0,len(decomp_img)):
        arr = decomp_img[i,:,:]
        to_zero = np.where(arr>4096)
        arr[to_zero] = 4095
    #                     if arr.shape[0]>512:
    #                         arr = scipy.misc.imresize(arr,(512,512))
    #                 print(arr.min())
    #                 print(arr.max())
        #np.save('image',arr)
        #plt.imshow(arr),plt.show()
        array_buffer = arr.tobytes()
        img = Image.new("I", (arr.shape[1],arr.shape[0]))
        img.frombytes(array_buffer, 'raw', "I;16")
        img.save(image_save_path + '/image - '+str(i)+'.png')
    
    

def group_scans_df(path,save_cache_path,save=False):

    if not os.path.isfile(save_cache_path):
    
        slices = load_all_dcm(path)
        #print(slices[0][1])
        dic = {'Name_of_file':[i[1] for i in slices]\
               ,'SeriesNumber':[i[0].SeriesNumber if i[0].__contains__('SeriesNumber') else -1 for i in slices],\
               'AcquisitionNumber':[i[0].AcquisitionNumber if i[0].__contains__('AcquisitionNumber') else -1 for i in slices]\
               ,'Manufacturer':[i[0][(0x8,0x70)].repval if i[0].__contains__((0x8,0x70)) else -1 for i in slices]\
               ,'Phase':[read_phase(i[0]) for i in slices]\
               ,'InstanceNumber':[i[0].InstanceNumber if i[0].__contains__('InstanceNumber') else -1 for i in slices]\
               ,'ImagePositionPatient':[i[0].ImagePositionPatient[2] if i[0].__contains__('ImagePositionPatient') else -1 for i in slices],
               'SeriesDescription':[i[0].SeriesDescription if i[0].__contains__('SeriesDescription') else -1 for i in slices],\
              'Modality':[i[0].Modality if i[0].__contains__('Modality') else -1 for i in slices]}

        df_dic = pd.DataFrame(dic)
        #global global_df
        if save:
            df_dic.to_csv('./'+path.split('/')[-1]+'.csv')
        #global global_var
        #global_var.put(df_dic)
        #print(global_var)
        #print('group_scans_df')
        #print(global_var)
        #print(save_cache_path)
        #global_df.put(df_dic)
        
        pickle.dump(df_dic,open(save_cache_path,'wb'))
    else:
        #print('here')
        bb = pickle.load(open(save_cache_path,'rb'))
    return df_dic
        #     for p in global_df:
#         print(p)
    #return df_dic
# for p in global_df:
#     print(p)




    #print(df_new)
    #return
def load_series_group(path,cache_path):
    #print('HEEERE22')
    def return_grouped_s(cache_path,df_obj):

        df = df_obj
        df_new = df.groupby(['Modality','SeriesNumber'])
        #global global_df

        if not os.path.isfile(cache_path+'.group.pkl'):
            pickle.dump(df_new,open(cache_path+'.group.pkl','wb'))
    #     print('PRININITN')
    #     print(df_new)
        #global_df.put(df_new)
    def group_scans_df(path,save_cache_path,save=False):

        if not os.path.isfile(save_cache_path):

            slices = load_all_dcm(path)
            #print(slices[0][1])
            dic = {'Name_of_file':[i[1] for i in slices]\
                   ,'SeriesNumber':[i[0].SeriesNumber if i[0].__contains__('SeriesNumber') else -1 for i in slices],\
                   'AcquisitionNumber':[i[0].AcquisitionNumber if i[0].__contains__('AcquisitionNumber') else -1 for i in slices]\
                   ,'Manufacturer':[i[0][(0x8,0x70)].repval if i[0].__contains__((0x8,0x70)) else -1 for i in slices]\
                   ,'Phase':[read_phase(i[0]) for i in slices]\
                   ,'InstanceNumber':[i[0].InstanceNumber if i[0].__contains__('InstanceNumber') else -1 for i in slices]\
                   ,'ImagePositionPatient':[i[0].ImagePositionPatient[2] if i[0].__contains__('ImagePositionPatient') else -1 for i in slices],
                   'SeriesDescription':[i[0].SeriesDescription if i[0].__contains__('SeriesDescription') else -1 for i in slices],\
                  'Modality':[i[0].Modality if i[0].__contains__('Modality') else -1 for i in slices]}

            df_dic = pd.DataFrame(dic)
            #global global_df
            if save:
                df_dic.to_csv('./'+path.split('/')[-1]+'.csv')
            #global global_var
            #global_var.put(df_dic)
            #print(global_var)
            #print('group_scans_df')
            #print(global_var)
            #print(save_cache_path)
            #global_df.put(df_dic)

            pickle.dump(df_dic,open(save_cache_path,'wb'))
        else:
            #print('here')
            bb = pickle.load(open(save_cache_path,'rb'))
        return df_dic

    #global global_df
    #var = global_df
    
    #global global_df
    #print('HEEERE')
    if not (os.path.isfile(cache_path)) :
        df_dic = group_scans_df(path,cache_path)
        return_grouped_s(cache_path,df_dic)
        
    else:
        #print(cache_path)
        pickle_obj = pickle.load(open(cache_path,'rb'))
        return_grouped_s(cache_path,pickle_obj)
           
    #return


class GUI():
    def __init__(self,tk_obj,file_path = '../TAVR_MAIN' ):
        
        self.file_path = file_path ### ROOT PATH!
        
        self.list_of_study = [i for i in os.listdir(self.file_path+'/TAVR_ROOT') if os.path.isdir(self.file_path+'/TAVR_ROOT/'+i)]
        
        self.study = []
        self.save_path = self.file_path+'/result'
        
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.isdir(self.file_path+'/Tool_Cache'):
            os.makedirs(self.file_path+'/Tool_Cache')
        self.cache_folder = self.file_path+'/Tool_Cache'
        self.vol = []
        self.val_to_tuple = {} 
        self.counter = -1
        self.study_to_text = {} ### lookup table linking study number with "button text"
        #self.df  =
        #global global_df
        self.new_df={}
        self.new_group_df = {}

        self.df = {}
        self.group_df_series = {}
        self.group_df = {}
        self.tree_series_dic = {}
#         self.df = group_scans_df(self.file_path)
#         self.group_df_series = return_grouped_s(self.df)
#         self.group_df = return_grouped(self.df)
        
        self.vol_name = None
        self.name = ''
        self.root = tk_obj
        #self.new_location = tk.Button(master=self.root,text='Enter new path',command=self.load_text_box).pack()
        #self.frame = tk.Frame(self.root)
        #self.frame.pack()
        self.v2 = tk.IntVar()
        self.v2.set(5)
        self.study_name = ''
        self.v = tk.IntVar()
        self.v.set(0)
        self.study_id = 0
        self.vol_id = 0
        self.next_var = tk.IntVar()
        self.prev_var = tk.IntVar()
        
        self.next_var.set(0)
        self.prev_var.set(0)

        # get all vols:
        #self.welcome_logo = itk.PhotoImage(PIL.Image.open('welcome.png'))
        
        self.header = ['Volume Index','Acquisition No.','Phase %','No. of Images']
        
        #self.tree_stud_frame = tk.Frame(self.root, width=150)
        self.frame_p = tk.Frame(self.root)
        self.frame_p.pack(side='top',expand=False,fill='both')
        self.frame1 = tk.Frame(self.root) # left frame for study treeview
        self.frame1.pack(side='left',expand=True,fill='both')
        self.frame2 = tk.Frame(self.root) # right frame for series, vol and img
        self.frame2.pack(side='right',expand=True,fill='both')
        
        
        ### splitting frame 2 into top half and bottom half
        self.frame2_1 = tk.Frame(self.frame2) 
        self.frame2_1.pack(side='top',fill='both',expand=True)
        #self.frame2_p = tk.Frame(self.frame2)
        #self.frame2_p.pack(side='bottom',fill='both',expand=True)
        self.frame2_2 = tk.Frame(self.frame2)
        self.frame2_2.pack(side='bottom',fill='both',expand=True)
        
        #### splitting frame 2_2 into right half and left half 
        self.frame2_2_1 = tk.Frame(self.frame2_2)
        self.frame2_2_1.pack(side='left',fill='both',expand=True)
        self.frame2_2_2 = tk.Frame(self.frame2_2)
        self.frame2_2_2.pack(side='right',fill='both',expand=True)
        
        self.tree_stud = ttk.Treeview(master=self.frame1,columns = ['Number','Folder'],show='headings')
        self.tree_stud.pack(side='left',expand=True,fill='both')
        #self.tree_stud.grid(row=0,column=0,rowspan=2)
        #self.tree_stud.pack(side='left',expand=True,fill='both')
        self.tree_stud.heading('Number',text='Number',anchor=tk.NW)
        self.tree_stud.heading('Folder',text='Folder',anchor=tk.NW)
            
        self.tree_stud.column(str(0),stretch=tk.NO)
        
        #self.tree_stud.pack(side='top',fill='both',expand=True)
        
        #self.frame1 = tk.Frame(self.root,width = 1000)
        
        
        
        self.tree_series = ttk.Treeview(master=self.frame2_1,columns = ['Modality','Series No.','Description','No. of Images'])
        #self.tree_series.grid(row=0,column=1)
        self.tree_series.pack(fill='both',expand=True)
        for idx,h in enumerate(['Modality','No. of Images','Series No.','Description']):
            self.tree_series.heading(h,text=h,anchor=tk.NW)
            self.tree_series.column(str(idx),stretch=tk.YES)
        #self.tree_series.heading('')
        
        self.tree_vol = ttk.Treeview(master=self.frame2_2_1,columns = self.header,show='headings')
        self.tree_vol.pack(fill='both',expand=True)
#         #self.tree_vol.grid(row=1,column=1)
#         #self.tree_vol.pack(side='bottom')
#         #self.tree_vol.pack(side='left')
        #self.welcome_img = 
        self.label  = tk.Label(master=self.frame2_2_2)#,image=itk.PhotoImage(PIL.Image.open('welcome.png')))
        #self.label.image = itk.PhotoImage(PIL.Image.open('welcome.png'))
#         #self.label.grid(row=1,column=2)
        self.label.pack(side='bottom',fill='both',expand=True)
        for idx,h in enumerate(self.header):
            self.tree_vol.heading(self.header[idx],text=self.header[idx],anchor=tk.CENTER)
            self.tree_vol.column(str(idx),stretch=tk.YES)

        self.b1 = tk.Radiobutton(master=self.frame2_2_2,text='Next',indicatoron=False,variable=self.next_var,command=self.next_b)
        self.b2 = tk.Radiobutton(master=self.frame2_2_2,text='Prev',indicatoron=False,variable=self.prev_var,command=self.prev_b)
        self.b1.pack(side='top')
        self.b2.pack(side='top')
        
        
        ### fills up the Study tree
        self.tree_vol_dic = {}
        self.tree_stud_dic = {}
        for i,j in enumerate(self.list_of_study):
            study_idx = i
            self.tree_stud.insert("","end",value=(str(study_idx),str(j)))
            self.tree_stud.bind("<ButtonRelease-1>",self.load_series) ### TODO define self.load_series 
            #self.tree_stud.bind("<ButtonRelease-1>",partial(load_series2,self))
            self.tree_stud_dic[i] = j
        
        
        
        self.annotate_button = tk.Radiobutton(master=self.frame2_2_2,\
                                        text="Annotate",\
                                        padx=20,\
                                        command = self.annotate_func,\
                                        value=0,\
                                         indicatoron=False,variable=self.v2
                                        ).pack(side='top')
        
        #self.frame1.grid(row=0, column=1, rowspan=2, sticky="nsew")
        #self.frame1.pack(side='right')
        #self.root.grid_rowconfigure(1, weight=1)
        #self.root.grid_columnconfigure(1, weight=1)
        
        self.pbar = Progressbar(self.frame_p,mode='indeterminate')
        self.pbar.grid(row=1,column=1,columnspan=100,sticky=tk.W+tk.E)
        #self.root.mainloop()
    
#     def onStart_1(self):
#         #self.
#         self.p1 = Process(target = self.load_series_group,args=(global_df,))
#         return
    
    def saver(self,args):
        # i is the image index
        # arr is the image
        
        arr=args[1]
        if arr.shape[0]>512:
            arr = scipy.misc.imresize(arr,(512,512))
        i = args[0]
        
        array_buffer = arr.tobytes()
        img = Image.new("I", arr.T.shape)
        img.frombytes(array_buffer, 'raw', "I;16")
        img.save(self.save_path+'/'+str(self.vol_name)+'/images/img - '+str(i)+'.png')

    
    def resample_check(self):
        return
    
    def resample(self, scan, image, new_spacing=[0.625,0.625,0.625]):
        spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
        #print(spacing)
        
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = new_real_shape
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        #print(new_real_shape)
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order = 2, mode='nearest')
        #print(new_spacing)
        
        
        
        return image, new_spacing   
    
    def load_series(self,event):
        #multiprocessing.freeze_support()

        #print("LOAD SERIES")
        if len(self.tree_series.get_children())>0:

            self.tree_series.delete(*self.tree_series.get_children())
        item =self.tree_stud.focus()
        #print(item)


        #print(self.tree_stud.item(item))

        folder = self.tree_stud.item(item)['values'][-1]
        folder = str(folder)
        self.study_name = folder
        
        study_path1 = self.file_path+'/TAVR_ROOT/'+folder
        cache_path = self.cache_folder+'/'+folder+'.pkl'
        #print(study_path1)

        #print(cache_path)
        self.cache_path = self.cache_folder+'/'+folder+'.pkl'
        #global global_df
        #print(global_df)
        
        if not os.path.isfile(self.cache_path):
            #print('self cache not file')
            self.p1 = Process(target = load_series_group,args=(study_path1,cache_path))
            #self.p1.daemon = True
            self.p1.start()
            #self.p1.join()
            #print('start')
            self.pbar.start(20)
        
            self.frame_p.after(10,self.onGetValue)
        else:
            #print(self.cache_path)
            #print('accessed else')
            self.df = pd.read_pickle(self.cache_path)
            
            self.df = pickle.load(open(self.cache_path,'rb'))
            self.group_df = self.df.groupby(['Modality','SeriesNumber'])
            #self.group_df = pickle.load(open(self.cache_path+'.group.pkl','rb'))
            self.fill_series_tab()
    
    def fill_series_tab(self):
    
        #print(self.group_df)
        #print(type(self.group_df))
        groups =self.group_df.groups
        self.group_df_series = self.group_df
        #print(groups)
        for j,i in enumerate(groups):
            button_text = [self.df.loc[groups[i]][j].iloc[0] for j in ['Modality','SeriesNumber','SeriesDescription']]
            button_text.append(len(self.df.loc[groups[i]]))

            self.tree_series.insert("","end",values = button_text)
            self.tree_series.bind("<ButtonRelease-1>",self.load_volumes)
            self.tree_series_dic[j] = button_text[:2]
    
    def onGetValue(self):
        if not(os.path.isfile(self.cache_path)) and not(os.path.isfile(self.cache_path+'.group.pkl')) :
            #print(self.cache_path)
            self.frame_p.after(40,self.onGetValue)
            return
        else:
            #print('made cache path')
            #self.p1.kill()
            
            #print(global_df.get(0))
            #self.p1.join()
            #self.p1.terminate()
            #self.p1.close()
            try:
            #self.p1.join()
                #print(global_df)
                #self.p1.end()
                time.sleep(1)
            
                #print('SELF CACHE PATH ',self.cache_path)
                f = open(self.cache_path,'rb')
                self.pbar.stop()  
                #b = global_df.get()
                self.df = pickle.load(f)
                self.group_df = self.df.groupby(['Modality','SeriesNumber'])
                #self.group_df_series = pickle.load(open(self.cache_path+'.group.pkl','rb'))
                                                   
                self.fill_series_tab()
                #self.p1.stop()
            except :#Empty:
                print('quque empty')
                #self.p1.stop()
    def get_pixels_hu(self,slices):
        slices = slices
        #image = [s.pixel_array.astype(np.int16) for s in slices]
        image = np.stack([s.pixel_array for s in slices])
        image = image.astype(np.int16)
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        #image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        #image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        
        
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            img = image[slice_number]
            if slices[slice_number].__contains__('PixelPaddingValue'):
                padding =  slices[slice_number].PixelPaddingValue
                img[img==padding] = 0  
            
            #to_zero = np.where(img>10000)
            #img[to_zero] = img.min()
            image[slice_number] = img
            
            #image[slice_number][image[slice_number]>4095] = 4095
            #image[slice_number][image[slice_number]<0] = 0
            
            # Shift 2 bits based difference 16 -> 14-bit as returned by jpeg2k_bit_depth
            #padded_pixels = np.where( img & (1 << 14))
            #image[slice_number] = np.right_shift( image[slice_number], 2)

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
            
            
            
            image[slice_number] += np.int16(intercept)
            #to_zero = np.where(image[slice_number]>10000)
            #image[slice_number][to_zero] = 0
                #img[padded_pixels] = intercept

        return np.array(image,dtype = np.int16)
    
    def onSaveImg(self):
        if len(self.decomp_img) > len(os.listdir(self.save_path_img)):
            self.frame_p.after(10,self.onSaveImg)
        else:
            time.sleep(10)
            self.pbar.stop()
            self.call_yuval_tool()
    
    def annotate_func(self):
        ### make the volume folder
        ### make one folder with png images that are isotropic, another with label inputs (later), add an XML file 
        ### invoke test.exe
        #multiprocessing.freeze_support()

#         if self.vol_name is None:
#             print('nothin')
#             #self.w.configure(text = "please select a volume")
        self.v2.set(0)
        if self.vol_name is not None:
            
            if not os.path.isdir(self.save_path+'/'+self.study_name+'/'+self.vol_name + '/images') :
                #print(self.save_path)
                #print(self.vol_name)
                self.vol_name = self.vol_name.replace(' ','')
                self.study_name = self.study_name.replace(' ','')
                if not os.path.isdir(self.save_path+'/'+self.study_name+'/'+self.vol_name + '/images'):
                    os.makedirs(self.save_path+'/'+self.study_name+'/'+self.vol_name + '/images')
                #os.makedirs(self.save_path+'/'+self.vol_name + '/Labels')
            ##print(self.vol[0][0])
            
                t =time.time()
                volume_meta = [decomp(v[0],return_dcm=True) for v in self.vol]
                
                print('len of vol = ',len(self.vol))
                
                print('loading all dicoms - ',time.time()-t)
                slices=volume_meta
                ###self.vol is the clicked volume regardless of study
                t = time.time()
                images = self.get_pixels_hu(slices)
                print('get hu ',time.time()-t)
#                 print(images.max())
#                 print(-1 + 2**16 )

#                 images = images - images.min()
#                 images[images>4095] = 4095
                images = images + 1024
                images[images>4095] = 4095
                
                #print('min then',images.min())    
                images[images<0] = 0
                #print('min now',images.min())
                
                t = time.time()
                decomp_img,spacing = self.resample(scan=volume_meta,image=images)
                
                #print('decomp max ',decomp_img.max())
                
                
                #print('new shape = ',decomp_img.shape)
                #print('new spacing = ',spacing)
                self.decomp_img = decomp_img
                #print('resample ',time.time()-t)
                spacing = np.round(spacing,3)
                num_rows = np.round(decomp_img.shape[0],3)
                num_cols = np.round(decomp_img.shape[1],3)
                num_z = np.round(decomp_img.shape[2],3)
                
                
                
                #inp_tuples = [[decomp_img[i,:,:],self.save_path,self.vol_name,i] for i in range(0,len(decomp_img))]
                #print(decomp_img.shape)
                #print(self.save_path)
                #print(self.vol_name)
                #print(i)
    #             print(num_rows)
    #             print(num_cols)
    #             print(num_z)
    #             print(decomp_img.shape)
#                 with Pool(3) as pool:
#                     pool.map(lsd.save_im,inp_tuples)
                self.v2.set(4)
    
    
                t = time.time()
                save_path = self.save_path
                vol_name = self.vol_name
#                 def cb_save_im():
#                     nonlocal decomp_img
                
                
                self.save_path_img = save_path+'/'+self.study_name+'/'+str(vol_name)+'/images/'
#                 for i in range(0,len(decomp_img)):
#                     arr = decomp_img[i,:,:]
#                     arr[arr>4096] = 0
# #                     if arr.shape[0]>512:
# #                         arr = scipy.misc.imresize(arr,(512,512))
#     #                 print(arr.min())
#     #                 print(arr.max())
#                     #np.save('image',arr)
#                     #plt.imshow(arr),plt.show()
#                     array_buffer = arr.tobytes()
#                     img = Image.new("I", (arr.shape[1],arr.shape[0]))
#                     img.frombytes(array_buffer, 'raw', "I;16")
#                     img.save(self.save_path+'/'+self.study_name+'/'+str(vol_name)+'/images/image - '+str(i)+'.png')
                
                save_imgs(self.save_path_img,decomp_img)
                #self.p2 = Process(target=save_imgs,args=(self.save_path_img,decomp_img))
                #self.p2.start()
                #self.pbar.start()
                #self.frame_p.after(10,self.onSaveImg)
                with open(self.save_path+'/'+self.study_name+'/'+str(self.vol_name)+'/volume.xml','w') as f:
                    f.write('<volume>\n'+'<Spacing><x>'+str(spacing[2])+'</x><y>'+str(spacing[1])+'</y><z>'+str(spacing[0])+'</z></Spacing>'+'\n<Size><x>'+str(num_z)+'</x><y>'+str(num_cols)+'</y><z>'+str(num_rows)+'</z></Size>'+'\n</volume>')
                self.call_yuval_tool()
            else:
                self.call_yuval_tool()
                #print('saving images ',time.time()-t)
#                 q = threading.Thread(target = cb_save_im)
#                 q.start()
#                 q.join()
                #t = time.time()
                
                
                    
                #print('saving xml',time.time()-t)

            
            
            #subprocess.call('./AT/test.exe --path '+self.save_path+'/'+self.study_name+'/'+str(self.vol_name))
        self.v2.set(5)
        ##print("Not implemented")
    
    def call_yuval_tool(self):
        subprocess.call('./AT2/test.exe --path '+self.save_path+'/'+self.study_name+'/'+str(self.vol_name))
        self.v2.set(5)
    
    def load_volumes(self,event):
        ### for treeview
        ### TODO create another tree view with "study" , upon clicking reveals all the "vols" in that study
        ### implement study level clustering
        
        self.tree_vol.delete(*self.tree_vol.get_children())

        item = self.tree_series.focus()
        
        mod_snum = self.tree_series.item(item)['values'][:2]
        tree_series_sel = (mod_snum[0],mod_snum[1])
        #print(tree_series_sel)
        
        self.new_df = self.df.loc[self.group_df_series.groups[tree_series_sel]]
        self.new_group_df = self.new_df.groupby(['AcquisitionNumber','SeriesNumber','Phase','Manufacturer'])
        
        ### now make the table of volumes 
        
        #self.study = return_volume(new_df,new_group_df,)
        self.header = ['Volume Index','Acquisition No.','Phase %','No. of Images']
        for i,j in enumerate(self.new_group_df):
            #print(j)
            acq_num = j[0][0]
            ser_num = j[0][1]
            ph_num = j[0][2]
            manu_info = j[0][3]
            #self.vol_name = 'Series_'+str(ser_num)
            self.name = 'Series_'+str(ser_num)
            vol_list = self.new_group_df.groups[j[0]]
            #'Phase','AcquisitionNumber','Manufacturer','SeriesNumber'
            #print(self.new_group_df)
            vols = return_volume(self.new_df,self.new_group_df,acq_num,ser_num,ph_num,manu_info)
            
            for k in range(0,len(vols)):
                
            
                self.tree_vol.insert("","end",value=(i+k,acq_num,ph_num,len(vol_list)))
                self.tree_vol.bind("<ButtonRelease-1>",self.select_volume)
                
                
                self.tree_vol_dic[i+k]=(acq_num,ser_num,ph_num,manu_info,k)
            

    def select_volume(self,event):
        item = self.tree_vol.focus()
        button_text = self.tree_vol.item(item)['values']
        
        if len(button_text)>0:
        
            index = button_text[0]

            acq_num,ser_num,ph_num,manu_info,k = self.tree_vol_dic[index]

            self.vol = return_volume(self.new_df,self.new_group_df,acq_num,ser_num,ph_num,manu_info)[k][::-1]
            #self.vol = self.vol[::-1]
            self.vol_name=self.name + '_Volume_'+str(index)
            #print(self.vol_name)
            #print(len(vol_sel))
            #print(vol_sel)

            self.counter=0
            self.v.set(0)
            self.next_b()
   
    
        #print('aa')
    def next_b(self):
        #import scipy.misc
        from decomp_dcm import decomp
        
        import matplotlib
        matplotlib.use('TKAgg')
        
        import matplotlib.pyplot as plt
        #print(plt.rcParams["figure.figsize"])
        plt.rcParams["figure.figsize"] = [2.5,2.5]
        #self.w.destroy()
        from PIL import Image
        if self.vol_name is not None:
            self.next_var.set(1)
            self.counter-=1

            if self.counter == len(self.vol) or self.counter == -1*len(self.vol):
                self.counter=0

            np_img = decomp(self.vol[self.counter][0])
#             print(type(np_img))
#             print(np_img.dtype)
#             print(np_img.shape)
            #pickle.dump(np_img,open('np_img.pkl','wb'))
            #print(np_img)
            #import scipy.misc
            #import imageio
            if '8' not in str(np_img.dtype):
                np_img_pil = Image.fromarray(np_img.astype(np.uint32))
            else:
                if len(np_img.shape)==3:
                    np_img_pil = Image.fromarray(np_img[:,:,0])
                else:
                    np_img_pil = Image.fromarray(np_img)

            
            #np_img = Image.frombytes("I;16",(tags.Columns,tags.Rows),np_img,'raw')
            if np_img.shape[0]>512 or np_img.shape[1]>512:
                #np_img = scipy.misc.imresize(np_img,(512,512))
                np_img_pil = np_img_pil.resize((256,256))
                np_img = np.array(np_img_pil).astype(np.uint32)
            #np_img.save('temp1.png')
#             print(np_img.max())
#             print(np_img.dtype)
            Image.fromarray(np_img.astype(np.uint32)).save('temp1.png')
            #imageio.imwrite('temp1.png',np_img)
            
            
            #scipy.io.imsave(np_img,'temp1.png')
            
            #bb = Image.fromarray(np_img,mode='I;16')
            #bb.save('temp1.png')
            #scipy.misc.imsave('temp1.png',np_img)
            
            
            a = Image.open('temp1.png')
            
            b = np.array(a.resize((256,256)))
            # bb = b

            # c = plt.Figure(frameon=False)

            # c.

            fig,ax = plt.subplots()
            #fig.disable_axis()
            plt.set_cmap('gray')
            ax.set_axis_off()
            ax.imshow(b)
            
            #fig.plot(b)
            fig.canvas.draw()
            buf = fig.canvas.print_to_buffer()[0]

            ncols,nrows = fig.canvas.get_width_height()

            b2 = Image.fromarray(np.fromstring(buf,dtype=np.uint8).reshape(nrows,ncols,4))
            #print(b2.size)
            
            photo= itk.PhotoImage(b2)
            self.next_var.set(0)
            
            self.label.config(image=photo)

            self.label.image=photo
            #plt.close(fig)
            #ax.close()
            #fig.close()
            #self.label.pack()
            #plt.close()
        
    def prev_b(self):
        from decomp_dcm import decomp
        import matplotlib
        matplotlib.use('TKAgg')
        import matplotlib.pyplot as plt
        plt.rcParams["figure.figsize"] = [2.5,2.5]
        
        #plt.rcParams["figure.figsize"] = [256,256]
        if self.vol_name is not None:
            self.prev_var.set(1)
            self.counter+=1
            
            np_img = decomp(self.vol[self.counter][0])
            #pickle.dump(np_img,open('np_img.pkl','wb'))
            #print(np_img)
            #import scipy.misc
            #import imageio
            
            np_img_pil = Image.fromarray(np_img.astype(np.uint32))
            
            
            
            #np_img = Image.frombytes("I;16",(tags.Columns,tags.Rows),np_img,'raw')
            if np_img.shape[0]>512 or np_img.shape[1]>512:
                #np_img = scipy.misc.imresize(np_img,(512,512))
                np_img_pil = np_img_pil.resize((256,256))
                np_img = np.array(np_img_pil).astype(np.int32)
            #np_img.save('temp1.png')
#             print(np_img.max())
#             print(np_img.dtype)

            Image.fromarray(np_img.astype(np.uint32)).save('temp1.png')
            #imageio.imwrite('temp1.png',np_img)
            
            #bb = Image.fromarray(np_img,mode='I;16')
            #bb.save('temp1.png')
            #scipy.misc.imsave('temp1.png',np_img)
            
            a = Image.open('temp1.png')
            b = np.array(a.resize((256,256)))
            # bb = b

            # c = plt.Figure(frameon=False)

            # c.

            fig,ax = plt.subplots()
            plt.set_cmap('gray')
            ax.set_axis_off()
            ax.imshow(b)
            #fig.plot(b)
            fig.canvas.draw()
            buf = fig.canvas.print_to_buffer()[0]

            ncols,nrows = fig.canvas.get_width_height()
                
            b2 = Image.fromarray(np.fromstring(buf,dtype=np.uint8).reshape(nrows,ncols,4))
            photo= itk.PhotoImage(b2)
            self.next_var.set(0)
            
            self.label.config(image=photo)

            self.label.image=photo
            #plt.close(fig)
            #plt.close()
            #fig.close()

def main():
    #print('HEEERE22')
    
    gui_obj = tk.Tk()
    a = GUI(gui_obj,'../TAVR_MAIN')
    gui_obj.mainloop()        #self.label.pack()
    
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    multiprocessing.freeze_support()
#     print('here')
#     import tkinter as tk
#     import numpy as np
#     import os
#     import pydicom
#     #import load_scan_dir as lsd
#     import pandas as pd
#     from load_scan_dir import return_volume,load_all_dcm,read_phase
#     from PIL import ImageTk as itk
#     import PIL
#     from functools import partial
#     import pickle
#     import scipy.misc
#     from decomp_dcm import *
#     #import load_scan_dir
#     # root = tk.Tk()
#     # next_var = tk.IntVar()
#     # prev_var = tk.IntVar()
#     import time
#     # next_var.set(0)
#     # next_var.set(0)
#     import time
#     import mp_test
#     # val_to_tuple = {}
#     import threading
#     from multiprocessing import Pool
#     # vol = []
#     import scipy.ndimage
#     import tkinter.ttk as ttk
#     import scipy.misc
#     import subprocess
#     #import png
#     from PIL import Image
#     from multiprocessing import Queue,Process
#     from queue import Empty
#     from decimal import Decimal,getcontext
#     from tkinter.ttk import Progressbar
    main()
#     gui_obj = tk.Tk()
#     a = GUI(gui_obj,'../TAVR_MAIN')
#     gui_obj.mainloop()        #self.label.pack()

# if __name__ =="__main__":
#     gui = GUI(file_path = '../TAVR_MAIN')