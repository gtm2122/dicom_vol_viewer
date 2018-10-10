import os
import pydicom
import numpy as np
import pandas as pd

def clean_name(n):
    if n[-1] =='/':
        return clean_name(n[:-1])
    else:
        return n

def read_phase(slice):
    if slice.__contains__((0x8,0x70)):
        maufacture = slice[(0x8,0x70)].repval
        if(maufacture == "'GE MEDICAL SYSTEMS'" and slice.__contains__((0x45,0x1033))):
            return slice[(0x45,0x1033)].value.decode('utf-8').replace(' ','')
    return str(-1)

def group_scans_df(path,save=False):
    path=clean_name(path)
    slices = [(pydicom.read_file(path+'/'+s,stop_before_pixels=True),path+'/'+s) for s in os.listdir(path)]
    #print(slices[0][1])
    dic = {'Name_of_file':[i[1] for i in slices]\
           ,'SeriesNumber':[i[0].SeriesNumber if i[0].__contains__('SeriesNumber') else -1 for i in slices],\
           'AcquisitionNumber':[i[0].AcquisitionNumber if i[0].__contains__('AcquisitionNumber') else -1 for i in slices]\
           ,'Manufacturer':[i[0][(0x8,0x70)].repval if i[0].__contains__((0x8,0x70)) else -1 for i in slices]\
           ,'Phase':[read_phase(i[0]) for i in slices]\
           ,'InstanceNumber':[i[0].InstanceNumber if i[0].__contains__('InstanceNumber') else -1 for i in slices]\
           ,'ImagePositionPatient':[i[0].ImagePositionPatient[2] if i[0].__contains__('ImagePositionPatient') else -1 for i in slices]}
    
    df_dic = pd.DataFrame(dic)
    
    if save:
        df_dic.to_csv('./'+path.split('/')[-1]+'.csv')

    return df_dic

def return_grouped(df):
    return df.groupby(['AcquisitionNumber','SeriesNumber','Phase','Manufacturer'])

def return_volume(orig_df , g,acq_num,ser_num,phase_num,man_name="'GE MEDICAL SYSTEMS'"):
    ### append - this decides whether memory should be consumed for the volumes , helpful if we just want to count
    ### the number of images in a volume to display all the counts for all volumes
    cc=0
    total_count = 0
    phase_num = str(phase_num)
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



    
