from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
class WFDataset(Dataset):
    def __init__(self,nparray_list,delay=30,seq_len=200,outdim=3,transform=None,begin_index=48):
        self.nparray_list=nparray_list
        self.transform=transform
        self.delay=delay
        self.sequence_len=seq_len
        self.outdim=outdim
        self.begin_index=begin_index
        self.len_per_array=self.nparray_list[0].shape[0]-self.sequence_len-self.delay-begin_index
        self.atten_indices=[3,6,4,7,5,8]

    def __len__(self):
        return self.len_per_array*len(self.nparray_list)

    def __getitem__(self,idx):
        list_idx=idx//self.len_per_array
        array_idx=idx%self.len_per_array

        sample={'X':self.nparray_list[list_idx][array_idx+self.begin_index:array_idx+self.begin_index+self.sequence_len,:],
                'y':self.nparray_list[list_idx][array_idx+self.begin_index+self.delay:array_idx+self.delay+self.begin_index+self.sequence_len,:self.outdim],
                #'decoder_inputs':self.nparray_list[list_idx][array_idx+self.begin_index:array_idx+self.begin_index+self.sequence_len,:13],
                'decoder_inputs':self.nparray_list[list_idx][array_idx+self.begin_index+self.delay:array_idx+self.delay+self.begin_index+self.sequence_len,3:13],
                'atten_features':self.nparray_list[list_idx][array_idx+self.begin_index+self.delay:array_idx+self.delay+self.begin_index+self.sequence_len,self.atten_indices],
                }
        #print(sample)
        if self.transform is not None:
            sample=self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self,sample):
        X,y,decoder_inupts,atten_features=sample['X'],sample['y'],sample['decoder_inputs'],sample['atten_features']
        return {'X':torch.from_numpy(X.astype(np.float32)),
                'y':torch.from_numpy(y.astype(np.float32)),
                'decoder_inputs':torch.from_numpy(decoder_inupts.astype(np.float32)),
                'atten_features':torch.from_numpy(atten_features.astype(np.float32))
               }



