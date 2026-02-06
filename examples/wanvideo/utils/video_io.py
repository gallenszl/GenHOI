import os
import skvideo.io
import cv2
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import decord

# def to_np(x):
#     if x.max()<=1 and x.min()>=-1:
#         x = 
#     elif x.max()<=1 and x.min()>=0:
#         x = x * 255
#     x = x.float()
    
#     x = torch.clamp(x,0,255).cpu().numpy()
#     if n_dim==5:
#         x = rearrange(x,'b f c h w -> (b f) c h w')
#     elif n_dim==4:
#         x = x.transpose(0,2,3,1).astype(np.uint8)
#     else:
#         x = x.transpose(1,2,0).astype(np.uint8)
#     return x

def to_np(x):
    return x.cpu().numpy()

def auto_scale(x):
    max_v = x.max()
    min_v = x.min()
    if max_v <=1 and min_v < 0 and min_v >=-1 :
        x =  (x * 0.5 + 0.5) * 255
    elif max_v <=1 and min_v>=0 :
        x =  x * 255
    return x.astype(np.uint8)

def auto_reshape(x):
    n_dim = len(x.shape)
    if isinstance(x,torch.Tensor):
        if n_dim==3:
            if x.shape[0]==3:
                return rearrange(x,'c h w -> 1 h w c' )
        elif n_dim==4:
            if x.shape[1]==3:
                return rearrange(x,'f c h w -> f h w c')
        elif n_dim==5:
            if x.shpae[1]==3:
                return rearrange(x,'b c f h w -> (b f) h w c')
            elif x.shape[2]==3:
                return rearrange(x,'b f c h w -> (b f ) h w c')
    elif isinstance(x,np.ndarray):
        if n_dim==3:
            if x.shape[0]==3:
                return  x.transpose(1,2,0)
        elif n_dim==4:
            if x.shape[1]==3:
                return x.transpose(0,2,3,1)
        elif n_dim==5:
            print("x.shape:",x.shape)
            if x.shape[1]==3:
                x = x.transpose(0,2,3,4,1)
                b,f,h,w,c = x.shape
                x = x.reshape((b*f,h,w,c))
                return x
            elif x.shape[2]==3:
                x = x.transpose(0,1,3,4,2)
                b,f,h,w,c = x.shape
                x = x.reshape((b*f,h,w,c))
                return x
    return x


def save_image(image,out_path):
    if isinstance(image,torch.Tensor):
        # image = to_np(image)
        image = auto_scale(to_np(auto_reshape(image)))
    else:
        image = auto_scale(auto_reshape(image))
    image = image[0]
    Image.fromarray(image).save(out_path)
    


def save_video(frames,video_out_path,fid_vis=False):
    if isinstance(frames,torch.Tensor):
        frames = to_np(frames)
        frames = auto_scale(auto_reshape(frames))
        frames = np.ascontiguousarray(frames)
    basedir = os.path.dirname(video_out_path)
    if basedir!="":
        os.makedirs(basedir,exist_ok=True)
    output_dict = {'-r':'25', '-pix_fmt':'yuv420p', '-crf':'18'}
    writer = skvideo.io.FFmpegWriter(video_out_path, outputdict=output_dict, verbosity=1)
    fid = 0
    for frame in frames:
        if fid_vis:
            frame = frame.astype(np.uint8)
            H,W = frame.shape[:2]
            # print("frame:", frame.shape)
            cv2.putText(frame, str(fid).zfill(4), (int(H * 0.1),int(W * 0.1) ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            fid+=1
            # frame = cv2.putText(frame,str(fid).zfill(4),(int(W-1),int(0.1 * H) ),cv2.FONT_HERSHEY_COMPLEX,2.0,(100,200,200),5 )
        writer.writeFrame(frame)
    writer.close()

def load_image(image_path):
    img = Image.open(image_path)
    return np.array(img)

def load_video(vid_path,inds=None,return_array=True):
    vr = decord.VideoReader(vid_path)
    if inds is None:
        inds = range(0,len(vr))
    out = []
    for i in inds:
        frame =vr[i].asnumpy()
        out.append(frame)
    if return_array:
        return np.array(out)
    else:
        return out 


