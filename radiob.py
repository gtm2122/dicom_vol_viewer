import tkinter as tk
import numpy as np
import os
import pydicom
import load_scan_dir as lsd
from load_scan_dir import *
from PIL import ImageTk as itk
import PIL
import pickle
import scipy.misc
from decomp_dcm import *
#import load_scan_dir
# root = tk.Tk()
# next_var = tk.IntVar()
# prev_var = tk.IntVar()

# next_var.set(0)
# next_var.set(0)
import time
import mp_test
# val_to_tuple = {}
import threading
from multiprocessing import Pool
# vol = []
import scipy.ndimage
import tkinter.ttk as ttk
import scipy.misc
import subprocess
#import png
from PIL import Image

#from tkinter import Entry,StringVar

class GUI():
    def __init__(self,file_path = '/data/gabriel/TAVR/TAVR_MAIN/' ):
        
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
        self.root = tk.Tk()
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
        
        self.frame1 = tk.Frame(self.root) # left frame for study treeview
        self.frame1.pack(side='left',expand=True,fill='both')
        self.frame2 = tk.Frame(self.root) # right frame for series, vol and img
        self.frame2.pack(side='right',expand=True,fill='both')
        
        
        ### splitting frame 2 into top half and bottom half
        self.frame2_1 = tk.Frame(self.frame2) 
        self.frame2_1.pack(side='top',fill='both',expand=True)
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
        
        self.label  = tk.Label(master=self.frame2_2_2,image=itk.PhotoImage(PIL.Image.open('welcome.png')))
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

        #self.root.mainloop()
    
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
        img.save(self.save_path+'/'+str(self.vol_name)+'/images/image - '+str(i)+'.png')


    def resample(self, scan, image, new_spacing=[0.625,0.625,0.625]):
        spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
        
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order = 1, mode='nearest')

        return image, new_spacing   
    
    def load_series(self,event):
        
       
        if len(self.tree_series.get_children())>0:

            self.tree_series.delete(*self.tree_series.get_children())
        item =self.tree_stud.focus()


        #print(self.tree_stud.item(item))

        folder = self.tree_stud.item(item)['values'][-1]
        self.study_name = folder
        if not os.path.isfile(self.cache_folder+'/'+folder+'.pkl'):
            self.df =  group_scans_df(self.file_path+'/TAVR_ROOT/'+folder)

            pickle.dump(self.df,open(self.cache_folder+'/'+folder+'.pkl','wb'))

        else:
            self.df = pickle.load(open(self.cache_folder+'/'+folder+'.pkl','rb'))
        #print(folder)
        self.group_df_series = return_grouped_s(self.df) 
        #['Modality','No. of Images','Series No.','Description']
        groups = self.group_df_series.groups
        for j,i in enumerate(groups):
            button_text = [self.df.loc[groups[i]][j].iloc[0] for j in ['Modality','SeriesNumber','SeriesDescription']]
            button_text.append(len(self.df.loc[groups[i]]))
            
            self.tree_series.insert("","end",values = button_text)
            self.tree_series.bind("<ButtonRelease-1>",self.load_volumes)
            self.tree_series_dic[j] = button_text[:2]
        ser_num_images = []
        
         
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
        

        
        #inp = [(s,s.pixel_array.astype(np.int16),i,s.RescaleIntercept,s.RescaleSlope,s.PixelPaddingValue) for i,s in enumerate(slices)]
#         with Pool(3) as p:
#             result = p.map(lsd.hu_conv,inp)
        
#         result = sorted(result,key=lambda s:s[0])
#         result = [i[0] for i in result]

#         image1 = image[:len(image)//4]
#         image2 = image[len(image)//4:len(image)//2]
#         image3 = image[len(image)//2:3*len(image)//4]
#         image4 = image[3*len(image)//4:]


#         image1,slice1 = image[:len(image)//4],slices[:len(image)//4]
#         image2,slice2 = image[len(image)//4:len(image)//2],slices[len(image)//4:len(image)//2]
#         image3,slice3 = image[len(image)//2:3*len(image)//4],slices[len(image)//2:3*len(image)//4]
#         image4,slice4 = image[3*len(image)//4:],slices[3*len(image)//4:]
#         info = [[s.RescaleIntercept,s.RescaleSlope,s.PixelPaddingValue] for s in slices]
#         info1 = [[s.RescaleIntercept,s.RescaleSlope,s.PixelPaddingValue] for s in slice1]
#         info2 = [[s.RescaleIntercept,s.RescaleSlope,s.PixelPaddingValue] for s in slice2]
#         info3 = [[s.RescaleIntercept,s.RescaleSlope,s.PixelPaddingValue] for s in slice3]
#         info4 = [[s.RescaleIntercept,s.RescaleSlope,s.PixelPaddingValue] for s in slice4]
        
#         def cb_hu(im,sl):
            #nonlocal im,sl
    
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            img = image[slice_number]
            if slices[slice_number].__contains__('PixelPaddingValue'):
                padding =  slices[slice_number].PixelPaddingValue
                img[img==padding] = 0  
            
            
            image[slice_number] = img
            # Shift 2 bits based difference 16 -> 14-bit as returned by jpeg2k_bit_depth
            #padded_pixels = np.where( img & (1 << 14))
            #image[slice_number] = np.right_shift( image[slice_number], 2)

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)
                #img[padded_pixels] = intercept
#         def cb(im,sl):
#             #nonlocal sl
#             #nonlocal im
#             for slice_number in range(len(sl)):

#                 intercept = sl[slice_number].RescaleIntercept
#                 slope = sl[slice_number].RescaleSlope
#                 padding =  sl[slice_number].PixelPaddingValue

#                 img = im[slice_number]
#                 img[img==padding] = 0  
#                 im[slice_number] = img
#                 # Shift 2 bits based difference 16 -> 14-bit as returned by jpeg2k_bit_depth
#                 #padded_pixels = np.where( img & (1 << 14))
#                 #image[slice_number] = np.right_shift( image[slice_number], 2)

#                 if slope != 1:
#                     im[slice_number] = slope * im[slice_number].astype(np.float64)
#                     im[slice_number] = im[slice_number].astype(np.int16)

#                 im[slice_number] += np.int16(intercept)
#                 #img[padded_pixels] = intercept
                
#             return im
#         inp = [(image[i],info[i]) for i in range(len(image))]
#         #inp1 = [(image1[i],info1[i]) for i in range(len(image1))]
#         #inp2 = [(image2[i],info2[i]) for i in range(len(image2))]
#         #inp3 = [(image3[i],info3[i]) for i in range(len(image3))]
#         #inp4 = [(image4[i],info4[i]) for i in range(len(image4))]
        
#         pool = Pool()
#         res=[]
#         for inpu in inp:
#             res.append(pool.apply_async(mp_test.cb,inpu))
        
#         for r in res:
#             r.wait()
        #print(len(res))
        #print(res[0].shape)
        
        #res1 = pool.map(mp_test.cb,inp1)
        #res2 = pool.map(mp_test.cb,inp2)
        #res3 = pool.map(mp_test.cb,inp3)
        #res4 = pool.map(mp_test.cb,inp4)
        
#         image[:len(image)//4]=pool.map(mp_test.cb,(image1,info1))
#         image[len(image)//4:len(image)//2]=pool.map(mp_test.cb,(image2,info2))
#         image[len(image)//2:3*len(image)//4]=pool.map(mp_test.cb,(image3,info3))
#         image[3*len(image)//4:]=pool.map(mp_test.cb,(image4,info4))
        
            #return np.array(image, dtype=np.int16)
        
#         t1 = threading.Thread(target=cb_hu,args=(image1,slice1))
#         t2 = threading.Thread(target=cb_hu,args=(image2,slice2))
#         t3 = threading.Thread(target=cb_hu,args=(image3,slice3))
#         t4 = threading.Thread(target=cb_hu,args=(image4,slice4))
        
#         t1.start()
#         t2.start()
#         t3.start()
#         t4.start()
        
#         t1.join()
#         t4.join()
#         t3.join()
#         t2.join()
        
#         image[:len(image)//4] = image1
#         image[len(image)//4:len(image)//2] = image2
#         image[len(image)//2:3*len(image)//4] = image3
#         image[3*len(image)//4:] = image4
        #res.wait()
        #print(res.get())
        return np.array(image,dtype = np.int16)
    
    
    
    def annotate_func(self):
        ### make the volume folder
        ### make one folder with png images that are isotropic, another with label inputs (later), add an XML file 
        ### invoke test.exe
        
#         if self.vol_name is None:
#             print('nothin')
#             #self.w.configure(text = "please select a volume")
        self.v2.set(0)
        if self.vol_name is not None:
            
            if not os.path.isdir(self.save_path+'/'+self.study_name+'/'+self.vol_name + '/images'):
                #print(self.save_path)
                #print(self.vol_name)
                os.makedirs(self.save_path+'/'+self.study_name+'/'+self.vol_name + '/images')
                #os.makedirs(self.save_path+'/'+self.vol_name + '/Labels')
            ##print(self.vol[0][0])
            
                t =time.time()
                volume_meta = [pydicom.dcmread(v[0]) for v in self.vol]
                print('loading all dicoms - ',time.time()-t)
                slices=volume_meta
                ###self.vol is the clicked volume regardless of study
                t = time.time()
                images = self.get_pixels_hu(slices)
                print('get hu ',time.time()-t)
                print(images.max())
                print(-1 + 2**16 )
                images = images + 1024
                
                t = time.time()
                decomp_img,spacing = self.resample(scan=volume_meta,image=images)
                print('resample ',time.time()-t)
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
                for i in range(0,len(decomp_img)):
                    arr = decomp_img[i,:,:]
                    if arr.shape[0]>512:
                        arr = scipy.misc.imresize(arr,(512,512))
    #                 print(arr.min())
    #                 print(arr.max())
                    #np.save('image',arr)
                    array_buffer = arr.tobytes()
                    img = Image.new("I", arr.T.shape)
                    img.frombytes(array_buffer, 'raw', "I;16")
                    img.save(self.save_path+'/'+self.study_name+'/'+str(vol_name)+'/images/image - '+str(i)+'.png')
                print('saving images ',time.time()-t)
#                 q = threading.Thread(target = cb_save_im)
#                 q.start()
#                 q.join()
                t = time.time()
                
                with open(self.save_path+'/'+self.study_name+'/'+str(self.vol_name)+'/volume.xml','w') as f:
                    f.write('<volume>\n'+'<Spacing><x>'+str(spacing[2])+'</x><y>'+str(spacing[1])+'</y><z>'+str(spacing[0])+'</z></Spacing>'+'\n<Size><x>'+str(num_z)+'</x><y>'+str(num_cols)+'</y><z>'+str(num_rows)+'</z></Size>'+'\n</volume>')
                    
                print('saving xml',time.time()-t)

            
            subprocess.call('./AT/test.exe --path '+self.save_path+'/'+self.study_name+'/'+str(self.vol_name))
        self.v2.set(5)
        ##print("Not implemented")
        
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
        
        index = button_text[0]
        
        acq_num,ser_num,ph_num,manu_info,k = self.tree_vol_dic[index]
        
        self.vol = return_volume(self.new_df,self.new_group_df,acq_num,ser_num,ph_num,manu_info)[k]
        
        self.vol_name=self.name + '_Volume_'+str(index)
        #print(self.vol_name)
        #print(len(vol_sel))
        #print(vol_sel)
        
        self.counter=-1
        self.v.set(0)
        self.next_b()
   
    
        #print('aa')
    def next_b(self):
        #self.w.destroy()
        if self.vol_name is not None:
            self.next_var.set(1)
            self.counter+=1

            if self.counter == len(self.vol):
                self.counter=0

            np_img = decomp(self.vol[self.counter][0])
            if np_img.shape[0]>512:
                np_img = scipy.misc.imresize(np_img,(512,512))
            scipy.misc.imsave('temp1.jpg',np_img)
            self.next_var.set(0)
            photo= itk.PhotoImage( PIL.Image.open('temp1.jpg'))
            self.label.config(image=photo)

            self.label.image=photo

            #self.label.pack()
        
        
    def prev_b(self):
        if self.vol_name is not None:
            self.prev_var.set(1)
            self.counter-=1

            if (self.counter < -len(self.vol)):
                self.counter = -1

            np_img = decomp(self.vol[self.counter][0])

            scipy.misc.imsave('./temp1.jpg',np_img)
            self.prev_var.set(0)
            photo= itk.PhotoImage(PIL.Image.open('temp1.jpg'))
            self.label.config(image=photo)

            self.label.image=photo

        #self.label.pack()


if __name__ =="__main__":
    gui = GUI(file_path = '/data/gabriel/TAVR/TAVR_Sample_Study')