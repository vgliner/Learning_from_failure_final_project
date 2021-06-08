from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
import configparser
import os
from os import listdir
from os.path import isfile, join
import time
import pandas as pd
import csv
import psutil
import urllib.request, json 
import ssl
import ECG_segmentation
import torch


class NY_Dataset(Dataset):
    # Convention   [n , height, width, color channel] 
    def __init__(self, root_dir=None,uploading_mode='HDD', classification_category='Atrial fibrillation', to_cut_image= False,
                maximal_RAM_occupation= 92, use_stored_data = False, stored_data_last_entries = 0, bool_is_from_NY_DB_class = False,
                to_equalize_hist= False, negative_class = False, to_use_hdf5 = False, dual_class = False):
        super().__init__()
        self.dual_class = dual_class
        self.to_use_hdf5 = to_use_hdf5   
        self.to_equalize_hist = to_equalize_hist
        self.all_possible_categories= []
        self.uploading_mode = uploading_mode
        config = configparser.ConfigParser()
        path_ = os.path.dirname(os.path.abspath(__file__))
        configuration = config.read(path_+'/config.ini')
        if to_use_hdf5:
            self.database_path = config['Image_database_path']['Images_Path_hdf5'] 
            with h5py.File(self.database_path, "r") as hdf:
                self.files_in_database = list(hdf.keys())                                         
        else:
            self.database_path = config['Image_database_path']['Images_Path']
            self.files_in_database = os.listdir(self.database_path)
            self.files_in_database = [i for i in self.files_in_database if i.endswith('.png')]              
        self.classification_path = config['Image_database_path']['Classification_Path']
        self.server_db_path = config['Server_storage_path']['Server_Path']

        self.classification_category = classification_category
        self.is_from_NY_DB_class = bool_is_from_NY_DB_class
        if self.is_from_NY_DB_class == False:
            self.classifications = self.upload_classifications()
        self.all_categories_set = set(self.all_possible_categories)
        self.image_ids=[file[:-4] for file in self.files_in_database]
        self.to_cut_image = to_cut_image
        self.maximal_RAM_occupation= maximal_RAM_occupation
        self.loaded_data = []
        self.use_stored_data = use_stored_data
        self.stored_data_last_entries = stored_data_last_entries
        if self.is_from_NY_DB_class:
            self.use_stored_data = True
            self.list_of_captured_from_mobile_files= None
        if self.use_stored_data:
            if self.is_from_NY_DB_class:
                if self.to_use_hdf5:
                    self.Adversarial_path = config['Adversarial_path']['Adversarial_path_hdf5']
                    with h5py.File(self.Adversarial_path, "r") as hdf:
                        self.list_of_captured_from_mobile_files = list(hdf.keys())
                else:
                    self.Adversarial_path = config['Adversarial_path']['Adversarial_path']
                    self.list_of_captured_from_mobile_files = [f for f in listdir(self.Adversarial_path) if isfile(join(self.Adversarial_path, f))]
                # print(f'Trues: {len(self.files_in_database)}, Falses: {len(self.list_of_captured_from_mobile_files)}')
            else:
                ssl._create_default_https_context = ssl._create_unverified_context
                urllib.request.urlopen("https://132.68.36.204/getAnalyzeData")
                with urllib.request.urlopen("https://132.68.36.204/getAnalyzeData") as url:
                    self.stored_data = json.loads(url.read().decode())
                # print(self.stored_data)
        self.in_w = 1650
        self.in_h = 880
        self.in_channels = 3
        self.dim = (self.in_w, self.in_h)
        self.stat_only = False
        self.negative_class = negative_class
        self.loaded_data = [0]*self.__len__()
        k= psutil.virtual_memory().percent
        if (k > 0.7 * self.maximal_RAM_occupation) and (self.uploading_mode=='Cache'):
            print(f'Caching will not be effective. Occupied % : {k}')
            self.uploading_mode = 'HDD'                 


    def set_statistics_only(self,stat_only = False):
        self.stat_only = stat_only


    def __len__(self):
        if self.use_stored_data:
            if self.is_from_NY_DB_class:
                # return len(self.files_in_database)+len(self.list_of_captured_from_mobile_files)
                return 2 * len(self.list_of_captured_from_mobile_files)
            else:
                return len(self.files_in_database)+self.stored_data_last_entries
        else:        
            return len(self.files_in_database)

    def __getitem__(self, idx):
        # if self.use_stored_data:
        #     print(f'Loading index {idx} out of {len(self.files_in_database)+len(self.stored_data[self.stored_data_min_entry:])}')
        if self.loaded_data[idx]== 0:
            try:
                if self.use_stored_data:
                    if self.is_from_NY_DB_class:  # Discriminator
                        try:
                            """
                            if idx >= len(self.list_of_captured_from_mobile_files):  # NY DB
                                img = self.NY_db_image_upload(idx-len(self.list_of_captured_from_mobile_files))
                                classification_ = True
                            else: # STORED DB- Adversarial
                                server_filename = self.list_of_captured_from_mobile_files[idx]
                                img = cv2.imread(os.path.join(self.Adversarial_path, server_filename))
                                img = self.Scale_Adversarial_DB(img)
                                classification_ = False
                            """
                            if (idx % 2 )==0 and idx//2 < (len(self.list_of_captured_from_mobile_files)):
                                if self.stat_only == False:
                                # print(f'Uploading from Adversarial {idx//2}')
                                    if self.to_use_hdf5:
                                        with h5py.File(self.Adversarial_path, "r") as hdf:
                                            img = np.array(hdf[self.list_of_captured_from_mobile_files[idx//2]]).astype("uint8")
                                    else:
                                        server_filename = self.list_of_captured_from_mobile_files[idx//2]
                                        img = cv2.imread(os.path.join(self.Adversarial_path, server_filename))
                                        img = self.Scale_Adversarial_DB(img)
                                        if self.to_equalize_hist:
                                            img = self.Normalize_NY_image(img, from_NY_db=False,img_id = idx)                                    
                                classification_ = False                                
                            elif ((idx-1) % 2 )==0 and (idx-1)//2 < (len(self.list_of_captured_from_mobile_files)):
                                # print(f'Uploading from NY DB {(idx-1)//2}')
                                if self.stat_only == False:
                                    if self.to_use_hdf5:
                                        with h5py.File(self.database_path, "r") as hdf:
                                            img = np.array(hdf[self.files_in_database[idx//2]]).astype("uint8")
                                    else:                                                                            
                                        img = self.NY_db_image_upload((idx-1)//2)
                                        if self.to_equalize_hist:
                                            img = self.Normalize_NY_image(img, from_NY_db=True,img_id = idx)                                        
                                classification_ = True                                
                            else:
                                # print(f'Uploading from NY DB {idx-len(self.list_of_captured_from_mobile_files)}')
                                if self.stat_only == False:
                                    if self.to_use_hdf5:
                                        with h5py.File(self.database_path, "r") as hdf:
                                            img = np.array(hdf[self.files_in_database[idx-len(self.list_of_captured_from_mobile_files)]]).astype("uint8")
                                    else:
                                        img = self.NY_db_image_upload(idx-len(self.list_of_captured_from_mobile_files))
                                        if self.to_equalize_hist:
                                            img = self.Normalize_NY_image(img, from_NY_db=True,img_id = idx)                                        
                                classification_ = True    

                        except:
                            print(f'Problem with index {idx} at Adversarial DB')
                    else:
                        if idx >= self.stored_data_last_entries:  # NY DB Diseases
                            if self.stat_only == False:
                                if self.to_use_hdf5:
                                    with h5py.File(self.database_path, "r") as hdf:
                                        img = np.array(hdf[self.files_in_database[idx]]).astype("uint8")
                                else:
                                    img = self.NY_db_image_upload(idx-self.stored_data_last_entries)
                                    if self.to_equalize_hist:
                                        img = self.Normalize_NY_image(img, from_NY_db=True,img_id = idx)                                
                                    classification_ = self.classifications[self.image_ids[idx-self.stored_data_last_entries]]
                        else: # STORED DB
                            stored_db_idx = self.stored_data[idx - self.stored_data_last_entries + len(self.stored_data[:-self.stored_data_last_entries])]
                            if self.stat_only == False:
                                server_filename = stored_db_idx['File_Name'][:-4]+'.jpg'
                                img = cv2.imread(os.path.join(self.server_db_path, server_filename))
                                img = self.cloud_image_upload(img)
                                if self.to_equalize_hist:
                                    img = self.Normalize_NY_image(img, from_NY_db=False,img_id = idx)                                    
                            classification_ = [True if self.classification_category.replace(" ","") in stored_db_idx['Class'] else False][0]
                else:
                    if self.stat_only == False:  
                        if self.to_use_hdf5:
                            with h5py.File(self.database_path, "r") as hdf:
                                img = np.array(hdf[self.files_in_database[idx]]).astype("uint8")
                        else:
                            img = self.NY_db_image_upload(idx)
                            if self.to_equalize_hist:
                                img = self.Normalize_NY_image(img, from_NY_db=True,img_id = idx)    
                    if self.to_use_hdf5:
                        classification_ = self.classifications[self.files_in_database[idx][:-4]]
                    else:
                        classification_ = self.classifications[self.image_ids[idx]]
            except:
                print(f'Problematic index : {idx}')
            if self.stat_only == False:
                sample = (img,classification_)
            else:
                sample = (0,classification_)
            if self.uploading_mode == 'Cache' :
                k= psutil.virtual_memory().percent
                if self.loaded_data[idx]==0 and k <= self.maximal_RAM_occupation and self.stat_only == False:
                    self.loaded_data[idx] = sample                
        else:
            sample = self.loaded_data[idx]
        if self.negative_class:
            cl_ = False if sample[1] else True
            sample= (sample[0],cl_)
        return sample
    
    def Normalize_NY_image(self, img, from_NY_db=True,img_id = 0):
        img_out = np.zeros_like(img,dtype = float)
        # if from_NY_db:
        #     bias= [249.6448053, 232.5919378, 235.52369368, 31.49782145, 38.72928989,36.40867263]
        # else:
        #     bias= [84.50752542, 78.20612408, 72.69077064, 50.65068953, 50.08393256, 50.81933268]
        #     # bias= [0.0,0.0,0.0,1.0,1.0,1.0]
        # img_out[0]=(img[0]-bias[0])/bias[3]
        # img_out[1]=(img[1]-bias[1])/bias[4]
        # img_out[2]=(img[2]-bias[2])/bias[5]
        d0 = np.max(img[0])-np.min(img[0])*1.0
        d1 = np.max(img[1])-np.min(img[1])*1.0
        d2 = np.max(img[2])-np.min(img[2])*1.0
        if d0 == 0:
            d0 =1.0
        if d1 == 0:
            d1 =1.0
        if d2 == 0:
            d2 =1.0                    
        img_out[0]=(img[0]-np.min(img[0]))/d0
        img_out[1]=(img[1]-np.min(img[1]))/d1
        img_out[2]=(img[2]-np.min(img[2]))/d2
        return img_out


    def Scale_Adversarial_DB(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        width = self.in_w # Like the cut of NY DB
        height = self.in_h  
        dsize = (width, height)
        # cv2.imshow('image window',img)
        # cv2.waitKey(0)
        # img = ECG_segmentation.cut_image_based_on_Gabor(img)
        if self.to_cut_image:
            img = cv2.resize(img, dsize)         
        img = np.transpose(img,(2,0,1))    
        return img    


    def NY_db_image_upload(self,idx):
        img = cv2.imread(os.path.join(self.database_path, self.files_in_database[idx]))
        # print(f'Uploaded: {self.files_in_database[idx]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        if self.to_cut_image:
            img = img[270:1150,:,:]                    
        img = np.transpose(img,(2,0,1))    
        return img    

    def cloud_image_upload(self, img, to_show = False, to_rotate = True, to_remove_frame = True):
        if to_rotate:
            img = cv2.rotate(img,cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        if to_remove_frame:
            img = self.remove_black_frame(img)
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        if to_show:
            self.plot(img) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        img = np.transpose(img,(2,0,1))
        return img                

    def remove_black_frame(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axis_0_indxs =  np.nonzero(np.sum(gray,axis=0))
        axis_1_indxs =  np.nonzero(np.sum(gray,axis=1))
        T = [img[:,i,:] for i in axis_0_indxs]
        T=np.squeeze(T)
        T1 = [T[i,:,:] for i in axis_1_indxs]
        T1=np.squeeze(T1)
        img = T1
        return img

    def plot(self, idx):
        item_to_show = self.__getitem__(idx)
        item_dims = np.shape(item_to_show[0])
        print(f'Showing item {idx},  size : {item_dims}')
        plt.imshow(item_to_show[0])
        plt.show()
        return

    def upload_classifications(self):
        classifications = {}
        Trues = 0
        Falses = 0
        with open(self.classification_path, newline='\n') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_idx, row in enumerate(reader):
                if row_idx > -1:
                    cl = [row['Dx'+str(indx)] for indx in range(1,11)]
                    for c in cl:
                        if c is not None:
                            self.all_possible_categories.append(c)
                    if isinstance(self.classification_category, list)==False:
                        classification = self.classification_category in cl
                    else:
                        if self.dual_class == False:
                            E = [i for i in self.classification_category if i in cl]
                            if len(E)>0:
                                classification= True
                            else:
                                classification= False
                        else:
                            classification = torch.tensor([int(self.classification_category[0] in cl),int(self.classification_category[1] in cl)])
                            if classification[0].item()==0 and classification[1].item()==0:
                                classification = 0
                            elif classification[0].item()==1 and classification[1].item()==0:
                                classification = 1
                            elif classification[0].item()==0 and classification[1].item()==1:
                                classification = 2                 
                            else:
                                classification = 3                 
                    classifications[row['id']] = classification
                    if self.dual_class == False:                    
                        if classification:
                            Trues+= 1
                        else:
                            Falses+= 1
                    else:
                        pass
                        # if classification[0]:
                        #     Trues+= 1
                        # else:
                        #     Falses+= 1                        
            print(f'Trues : {Trues}, Falses: {Falses}')
            self.stats = [Trues , Falses]
        return classifications

def find_corrupted_image_in_adversarial_db():
    PATH = '/home/vadimgl/Data_May_and_Nitzan/Adversarial'
    import glob
    file_list = glob.glob(os.path.join(PATH,'*.jpg'))
    cntr = 0    
    for file in file_list:
        cntr+=1
        if cntr %100 == 0:
            print(f'I am at {cntr}')
        with open(os.path.join(PATH, file), 'rb') as f:
            check_chars = f.read()[-2:]
        if check_chars != b'\xff\xd9':
            print(f'Not complete image: {file}')
        else:
            imrgb = cv2.imread(os.path.join(PATH, file), 1)    


def Find_mean_and_std_of_NY_db():
    logger = []
    sums_ = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    ds = NY_Dataset(uploading_mode='HDD', classification_category='Atrial fibrillation', to_cut_image= True,
            maximal_RAM_occupation= 92, use_stored_data = True, stored_data_last_entries = 0, bool_is_from_NY_DB_class = True, to_equalize_hist=True)
    for img_cntr in range(len(ds)):
        if img_cntr+1 %2 ==0:
            continue
        img,cls = ds[img_cntr]
        k= [np.mean(img[0]),np.mean(img[1]),np.mean(img[2]),np.std(img[0]),np.std(img[1]),np.std(img[2])]
        logger.append(k)
        sums_ += np.array(k)
        print(f'Image # {img_cntr},So far:: mean 1: {sums_/(img_cntr+1)}')
        print(f'Image # {img_cntr},This specific:: {k}')

    print('Finished')

if __name__=="__main__":
    Find_mean_and_std_of_NY_db()
    # find_corrupted_image_in_adversarial_db()
    # ds = NY_Dataset(uploading_mode='HDD', classification_category='Atrial fibrillation', to_cut_image= True,
    #             maximal_RAM_occupation= 92, use_stored_data = True, stored_data_last_entries = 0, bool_is_from_NY_DB_class = True)
    # start = time.time()
    # Trues = 0
    # print(f'Length of database: {len(ds)}')
    # for loops in range(2):
    #     for item_idx in range(100000):
    #         item=ds[item_idx]
    #         # print(f'Item: {item_idx}, {item[1]}')
    #         Trues+= item[1]
    #         if item_idx %100 ==0:
    #             print(f'Currently at : {item_idx}')
    #             end = time.time()
    #             print(f'Elapsed time: {end-start}, loaded: {len(ds)}. Average : {(end-start)/len(ds)}, Trues : {Trues}')
    #             end = start

    # # ds.plot(0)
    # print('Testing NY database dataloader')