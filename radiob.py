import tkinter as tk
import numpy as np
import os
import pydicom
from load_scan_dir import *
from PIL import ImageTk as itk
import PIL
import pickle
import scipy.misc
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
import scipy.ndimage
import tkinter.ttk as ttk
import scipy.misc
import subprocess
import png
from PIL import Image

from tkinter import Entry,StringVar
class entry_gui(object):
    def __init__(self):
        self.master = tk.Tk()
        self.content = StringVar()
        self.entry_box = Entry(self.master,width=50,textvariable=self.content)
        self.entry_box.pack(side='left')
        self.enter_button = tk.Button(self.master, text="Enter", width=10, command=self.callback).pack(side='right')
        self.gui_class = GUI
        self.master.mainloop()
    def callback(self):
        print(self.content.get())
        self.master.destroy()
        a = self.gui_class(self.content.get())

#import pypng as png
class GUI():
    def __init__(self,file_path = '/data/gabriel/TAVR/TAVR_Sample_Study' ):
        
        
        self.study = []
        self.file_path = file_path
        self.vol = []
        self.val_to_tuple = {} 
        self.counter = -1
        self.study_to_text = {} ### lookup table linking study number with "button text"
        #self.df  =
        self.df = group_scans_df(self.file_path)
        self.group_df = return_grouped(self.df)
        
        self.vol_name = None
        
        self.root = tk.Tk()
        self.new_location = tk.Button(master=self.root,text='Enter new path',command=self.load_text_box).pack()
        
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
        
        
        
        self.label  = tk.Label(master=self.root,image=itk.PhotoImage(PIL.Image.open('welcome.png')))
        self.label.image = itk.PhotoImage(PIL.Image.open('welcome.png'))
        self.label.pack(side='bottom')
        
        
        
        self.b1 = tk.Radiobutton(master=self.root,text='Next',indicatoron=False,variable=self.next_var,command=self.next_b)
        self.b2 = tk.Radiobutton(master=self.root,text='Prev',indicatoron=False,variable=self.prev_var,command=self.prev_b)
        self.b1.pack()
        self.b2.pack()
        self.header = ['Volume Index','Series Description','Acquisition No.','Phase %','No. of Images']
        
        self.tree_stud = ttk.Treeview(master=self.root,columns = ['Study'],show='headings')
        self.tree_stud.pack(side='left')
        
        self.tree_stud.heading('Study',text='Study',anchor=tk.CENTER)
        self.tree_stud.column(str(0),stretch=tk.YES)
        
        self.tree_vol = ttk.Treeview(master=self.root,columns = self.header,show='headings') 
        self.tree_vol.pack(side='left')
        
        
        for idx,h in enumerate(self.header):
            self.tree_vol.heading(self.header[idx],text=self.header[idx],anchor=tk.CENTER)
            self.tree_vol.column(str(idx),stretch=tk.YES)
            
        all_keys =list( self.group_df.groups.keys())
        self.w = tk.Message(master=self.root,text = "")
        
        for i,j in enumerate(all_keys):
            vol_idx = i
#             print('keys')
#             print(j)
            ser_num = j[1]
            acq_num = j[0]
            ph_num  = j[2]
            man_name =j[3]
            #ser_des = j[-1]
            
            ### READ - "i" is the study id while vol_id is the study's volume id
            
            grouped_subset = self.group_df.groups[(acq_num, ser_num, str(ph_num), man_name)] # gives the indices of the grouping
            group = self.df.loc[grouped_subset]
            
            ser_des = group['SeriesDescription']
            
            
            
            
            #vol=return_volume(self.df,self.group_df,acq_num,ser_num,ph_num,man_name)
            #print(vol)
             
            #print(v)
            #print(i)
            self.tree_stud.insert("","end",value=str(vol_idx+1))
            self.tree_stud.bind("<Double-1>",self.load_volume)
            
#             num_vols = len(vol)
#             num_images = [len(v) for v in vol]
            
            
            
            #num_images.append(len(vol[-1]))
#              button_text = (ser_num,acq_num,ph_num,num_images)
            button_text = (ser_num,acq_num,ph_num,ser_des,man_name)
            
            self.study_to_text[1+i] = button_text
            
#             for vol_id in range(0,num_vols):
#                 #self.v.set(0)
#                 #print(man_name[0])
#                 vol_id1 = vol_id +  i
            
#                 button_text1 = (vol_id1,ser_des.values[0],acq_num,ph_num,num_images[vol_id]) 
                
#                 #print(button_text1)
                
#                 self.tree_vol.insert("","end",values = button_text1)
#                 self.tree_vol.bind("<Double-1>",self.load_volume)
                
# #                 tk.Radiobutton(master=self.root,\
# #                                text=button_text1
# #                                ,padx=20,\
# #                                variable=self.v,\
# #                                command=self.load_volume,\
# #                                value=i,\
# #                                indicatoron=False).pack(anchor=tk.W)

#                 self.val_to_tuple[vol_id1]=[button_text1,man_name,ser_num]
                
                
        self.annotate_button = tk.Radiobutton(master=self.root,\
                                        text="Annotate",\
                                        padx=20,\
                                        command = self.annotate_func,\
                                        value=0,\
                                         indicatoron=False
                                        ).pack(anchor=tk.SW)

        self.root.mainloop()
    
    def resample(self, scan, image, new_spacing=[0.625,0.625,0.625]):
        # Determine current pixel spacing
        #spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
        spacing = np.array([scan[0].SliceThickness, scan[0].PixelSpacing[0], scan[0].PixelSpacing[1]], dtype=np.float32)
        #spacing = np.array(list(spacing))
        
        
        
        #print('image dtype')
        #print(image.dtype)
        #print(spacing)
        #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
        #image = np.array([i.pixel_array for i in scan]).transpose(1,2,0)
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order = 1, mode='nearest')

        return image, new_spacing   
    
    def get_pixels_hu(self,slices):
    
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        #image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            padding =  slices[slice_number].PixelPaddingValue

            img = image[slice_number]
            img[img==padding] = 0        
            # Shift 2 bits based difference 16 -> 14-bit as returned by jpeg2k_bit_depth
            #padded_pixels = np.where( img & (1 << 14))
            #image[slice_number] = np.right_shift( image[slice_number], 2)

            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)
            #img[padded_pixels] = intercept

        return np.array(image, dtype=np.int16)
    
    def annotate_func(self):
        ### make the volume folder
        ### make one folder with png images that are isotropic, another with label inputs (later), add an XML file 
        ### invoke test.exe
        
#         if os.path.isdir('volume')
#         os.system('')
        #if self.vol_name is None:
        
        
        #image = np.array([decomp(i[0]) for i in self.vol])
        
        if self.vol_name is None:
            
            self.w.configure(text = "please select a volume")
        
        else:
            
            if not os.path.isdir(self.vol_name + '/images'):
                os.makedirs(str(self.vol_name)+ '/images')
                os.makedirs(str(self.vol_name)+ '/Labels')
            ##print(self.vol[0][0])
            volume_meta = [pydicom.dcmread(v[0]) for v in self.vol]
            
            slices=volume_meta
            ###self.vol is the clicked volume regardless of study
            images = self.get_pixels_hu(slices)
            images = images + 1024
            decomp_img,spacing = self.resample(scan=volume_meta,image=images)
            spacing = np.round(spacing,3)
            num_rows = np.round(decomp_img.shape[0],3)
            num_cols = np.round(decomp_img.shape[1],3)
            num_z = np.round(decomp_img.shape[2],3)
            
#             print(num_rows)
#             print(num_cols)
#             print(num_z)
#             print(decomp_img.shape)
            
            for i in range(0,len(decomp_img)):
                arr = decomp_img[i,:,:]
                if arr.shape[0]>512:
                    arr = scipy.misc.resize(arr,(512,512))
#                 print(arr.min())
#                 print(arr.max())
                #np.save('image',arr)
                array_buffer = arr.tobytes()
                img = Image.new("I", arr.T.shape)
                img.frombytes(array_buffer, 'raw', "I;16")
                img.save(str(self.vol_name)+'/images/image - '+str(i)+'.png')

#             for i,img in enumerate(decomp_img):
#                 print(img.shape)
#                 print(type(img))
#                 print(img.dtype)
#                 #png.from_array(img, 'L').save('volume'+str(self.vol_name)+'/Images/Image_'+str(i)+'.png')
# #                 with open('img - '+str(i)+'.png', 'wb') as f:
# #                     writer = png.Writer(width=img.shape[1], height=img.shape[0], bitdepth=16)
# #                     z2list = img.reshape(-1, img.shape[1]*img.shape[2]).tolist()
# #                     writer.write(f, z2list)
#                 scipy.misc.imsave('volume'+str(self.vol_name)+'/Images/Image_'+str(i)+'.png',img)
            
            with open(str(self.vol_name)+'/volume.xml','w') as f:
                f.write('<volume>\n')
                f.write('<Spacing><x>'+str(spacing[2])+'</x><y>'+str(spacing[1])+'</y><z>'+str(spacing[0])+'</z></Spacing>')
                f.write('\n<Size><x>'+str(num_z)+'</x><y>'+str(num_cols)+'</y><z>'+str(num_rows)+'</z></Size>')
                f.write('\n</volume>')
                
            #print(self.vol_name)
            #print('D:/Anaconda3/project/TAVR/annotation/text.exe --path volume'+str(self.vol_name))
            #print('anno_new/text.exe --path volume'+str(self.vol_name))
            #print(self.vol_name)
            #print(type(self.vol_name))
            subprocess.run('anno_new/test.exe --path '+str(self.vol_name))
        
        ##print("Not implemented")
        
    def load_volume(self,event):
        ### for treeview
        ### TODO create another tree view with "study" , upon clicking reveals all the "vols" in that study
        ### implement study level clustering
        
        
        self.tree_vol.delete(*self.tree_vol.get_children())
        
        #button_text = (ser_num,acq_num,ph_num,ser_des,man_name)
        #return_volume(self.df,self.group_df,acq_num,ser_num,ph_num,man_name)
        #print(self.tree_stud.selection())
        #self.vol_name = ''
        self.counter=-1
        req_tuple = self.study_to_text[int(self.tree_stud.selection()[0][1:],16)]
        #print(req_tuple)
        #req_tuple = self.val_to_tuple[int(self.tree_stud.selection()[0][1:],16)-1]
        ##print(req_tuple)
        
#         for i in req_tuple:
#             print(type(i))
#             print(i)
        
        self.study = return_volume(self.df,self.group_df,req_tuple[1],req_tuple[0],req_tuple[2],req_tuple[-1])
        #print(len(vol))
        
        num_vols = len(self.study)
        num_imgs = [len(i) for i in self.study]
        
        #self.tree_vol.insert()
        self.header = ['Volume Index','Series Description','Acquisition No.','Phase %','No. of Images']
        
        
        for i in range(0,num_vols):
            #print(str(i))
#             print(req_tuple[-2][0])
#             print(req_tuple[1][0])
#             print(req_tuple[2][0])
            #print(req_typle[-2])

            button_text1 = (str(i),req_tuple[-2].values[0],req_tuple[1],req_tuple[2],str(num_imgs[i]))
            self.tree_vol.insert("","end",values = button_text1)
            self.tree_vol.bind("<Double-1>",self.load_each_volume)
        
#         self.vol = self.vol[req_tuple[0][0]%len(self.vol)]
#         self.vol_name = req_tuple[0][0]
# #         print(len(self.vol))
# #         print(self.vol)
#         #self.v.set(0)
#         self.next_b()
        self.v.set(0)
#         self.w.configure(text = "viewing  Vol Idx %s, Series %s, Acq Num %s"%(req_tuple[0][0],req_tuple[2],req_tuple[0][3]))
#         self.w.pack()
        self.study_id = int(self.tree_stud.selection()[0][1:],16)
        #self.vol_name = 'Study'+str(int(self.tree_stud.selection()[0][1:],16))
    def load_each_volume(self,event):
        
        vol_idx = (int(self.tree_vol.selection()[0][1:],16)-1) % len(self.study)
        #print(int(self.tree_vol.selection()[0][1:],16)-1)
        #print(vol_idx)
        self.vol_name = 'study_'+str(self.study_id)+'_vol_'+str(vol_idx)
        self.vol = self.study[vol_idx]
        
        self.next_b()
        
        #print('aa')
    def next_b(self):
        #self.w.destroy()
        
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
        
        self.label.pack()
        
        
    def prev_b(self):
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
        
        self.label.pack()
    
    def load_text_box(self):
        self.root.destroy()
        
        bb = entry_gui()
        
        

if __name__ =="__main__":
    gui = GUI(file_path = '/data/gabriel/TAVR/TAVR_Sample_Study')