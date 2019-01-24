import os
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import scipy.misc
#from radiob import global_df
# def clean_name(n):
#     if n[-1] =='/':
#         return clean_name(n[:-1])
#     else:
#         return n
def hu_conv(args):
    slices = args[0]
    img = args[1]
    idx = args[2]

    intercept =args[3]
    slope = args[4]#slices[slice_number].RescaleSlope
    padding =  args[5]#slices[slice_number].PixelPaddingValue

    #img = image[slice_number]
    img[img==padding] = 0        
    # Shift 2 bits based difference 16 -> 14-bit as returned by jpeg2k_bit_depth
    #padded_pixels = np.where( img & (1 << 14))
    #image[slice_number] = np.right_shift( image[slice_number], 2)

    if slope != 1:
        img = slope * img.astype(np.float64)
        img = img.astype(np.int16)

    img += np.int16(intercept)

    return (idx,img.astype(np.int16))

def save_im(args):
    arr = args[0]
    save_path = args[1]
    vol_name = args[2]
    i = args[3]
    
    save_path = args
    if arr.shape[0]>512:
        arr = scipy.misc.imresize(arr,(512,512))
    #                 print(arr.min())
    #                 print(arr.max())
    #np.save('image',arr)
    array_buffer = arr.tobytes()
    img = Image.new("I", arr.T.shape)
    img.frombytes(array_buffer, 'raw', "I;16")
    print(save_path)
    print(str(vol_name))
    print(str(i))
    img.save(save_path+'/'+str(vol_name)+'/images/image - '+str(i)+'.png')


def read_phase(slice):
    if slice.__contains__((0x8,0x70)):
        maufacture = slice[(0x8,0x70)].repval
        if(maufacture == "'GE MEDICAL SYSTEMS'" and slice.__contains__((0x45,0x1033))):
            return slice[(0x45,0x1033)].value.decode('utf-8').replace(' ','')
        elif slice.__contains__((0x20,0x9241)):
            if isinstance(slice[(0x20,0x9241)].value,float):
                return str(slice[(0x20,0x9241)].value)
            else:
            #print(slice[(0x20,0x9241)].value)
                return slice[(0x20,0x9241)].value.decode('utf-8').replace(' ','')
    return str(-1)


def is_dicom(path):
    try:
        pydicom.dcmread(path).pixel_array
        return True
    except:
        return False

def load_all_dcm(path):
    all_dcms = []
    for i,j,k in os.walk(path):
        
        if isinstance(k,list) and len(k)>0 and is_dicom(i+'/'+k[0]) and 'DICOMDIR' not in k[0]:
            all_dcms+=[(pydicom.dcmread(i+'/'+p,stop_before_pixels=True),i+'/'+p) for p in k if 'DICOMDIR' not in p and is_dicom(i+'/'+p)]
    return all_dcms
    
def group_scans_df(global_var,path,save_cache_path,save=False):
    #path=clean_name(path)
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
    
    if save:
        df_dic.to_csv('./'+path.split('/')[-1]+'.csv')
    #global global_var
    #global_var.put(df_dic)
    #print(global_var)
    #print('group_scans_df')
    #print(global_var)
    #print(save_cache_path)
    pickle.dump(df_dic,open(save_cache_path,'wb'))
            
    #return df_dic

def return_grouped_s(global_df,pickle_file):
    #global global_var
    #global global_df
    #global_var = global_df
    df = pickle_file
    df_new = df.groupby(['Modality','SeriesNumber'])
    pickle.dump(df_new,open(cache_path+'.group.pkl','wb'))
    global_df.put(df_new)
def load_series_group(global_df,path,cache_path):
    #global global_df
    #var = global_df
    if not (os.path.isfile(cache_path)) :
        group_scans_df(global_df,path,cache_path)
        return_grouped_s(var)
        
    else:
        pickle_obj = pickle.load(open(cache_path,'rb'))
        return_grouped_s(global_df,pickle_obj)
        
        
def return_grouped(df):
    return df.groupby(['AcquisitionNumber','SeriesNumber','Phase','Manufacturer'])

def return_volume(orig_df , g,acq_num,ser_num,phase_num,man_name="'GE MEDICAL SYSTEMS'"):
    ### append - this decides whether memory should be consumed for the volumes , helpful if we just want to count
    ### the number of images in a volume to display all the counts for all volumes
    cc=0
    total_count = 0
    phase_num = str(phase_num)
    #print(g.keys())
    grouped_subset = g.groups[(acq_num, ser_num, str(phase_num), man_name)] # gives the indices of the grouping
    
    group = orig_df.loc[grouped_subset]
    #print(len(group))
    
    # if (len(group)==1):
    #     return 
    
    #for name,group in g:
    #cc+=1
    # print(cc)
    # print(name)
    # print(type(group))
    # break
    #here = 0
    indic=[]
    all_id = []
    #if(name[0]!=-1  ):

    inst_g = group.sort_values(by='InstanceNumber')
    #print(len(inst_g))
    
    if (len(inst_g)==1):
        return [[inst_g[0:1].Name_of_file.values]]
    
    ind_list = inst_g.index
    vol_idx = 1

    cur_thick = inst_g[1:2].ImagePositionPatient.values - inst_g[0:1].ImagePositionPatient.values
    volumes = []
    cur_thick = np.round(cur_thick,4)

    vol_count=0
    vol_counts = []
    
    # print(grouped_subset)
    # print(group)
    
    for vol_idx in range(1,len(inst_g)):
        ### starting from 1
        #print('vol_idx=',vol_idx)
        prev_slice = inst_g[vol_idx-1:vol_idx]
        cur_slice = inst_g[vol_idx:vol_idx+1]
        next_slice = inst_g[vol_idx+1:vol_idx+2]
        ### include all the "prev_slice"s if continuity holds, 
        ### include the first prev_slice once continuity breaks and then create a new volume
        ### compute the new expected thickness once by computing next_slice - cur_slice

        #all_id.append(cur_slice.values)


        this_thick = cur_slice.ImagePositionPatient.values - prev_slice.ImagePositionPatient.values
        this_thick = np.round(this_thick,4)
        #this_thick = this_thick*1e4//1/1e4
        if this_thick == cur_thick and len(volumes)==0:
            #print(this_thick)
            
            total_count+=1
            vol_count+=1


            volumes.append([])
            volumes[-1].append(prev_slice.Name_of_file.values)
            
            #volumes[-1].app
            #indic.append(prev_slice.index.values)

#                 volumes[-1].append(inst_g[vol_idx-1:vol_idx].Name_of_file.values)
#                 indic.append(inst_g[vol_idx-1:vol_idx].index.values)

        elif this_thick == cur_thick and len(volumes)>0 :
            total_count+=1
            #volumes[-1].append(inst_g[vol_idx-1:vol_idx].Name_of_file.values)
            #indic.append(inst_g[vol_idx-1:vol_idx].index.values)
            vol_count+=1
            
            volumes[-1].append(prev_slice.Name_of_file.values)
            #indic.append(prev_slice.index.values)


        elif this_thick != cur_thick :
            #print(vol_idx)
            total_count+=1
            #here = 1
            vol_count+=1
            #print(inst_g[vol_idx-1:vol_idx].Name_of_file.values)
            #volumes[-1].append(inst_g[vol_idx-1:vol_idx].Name_of_file.values)
            #indic.append(inst_g[vol_idx-1:vol_idx].index.values)
            volumes[-1].append(prev_slice.Name_of_file.values)
            #print(vol_idx)
            #indic.append(prev_slice.index.values)

            volumes.append([])
            vol_counts.append(vol_count)
            vol_count = 0
            cur_thick = next_slice.ImagePositionPatient.values - cur_slice.ImagePositionPatient.values
            cur_thick =  np.round(cur_thick,4)
            
        if vol_idx == len(inst_g)-1 and this_thick==cur_thick:
            volumes[-1].append(cur_slice.Name_of_file.values)
            
    
        #print('len vol = ',len(volumes[-1]))


        #print(cur_thick)
        #print(volumes)
#             if here==1:

#                 print('volumes = ',len(volumes))
#                 print([len(i) for i in volumes])
#                 break
#             print("diff = ",len(all_id)-len(indic))
    
        
    return volumes

    #return vol_count


