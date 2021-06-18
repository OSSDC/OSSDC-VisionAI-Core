import traceback
import sys
import os

# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
# https://github.com/NVlabs/SegFormer

# Install steps:
# git clone https://github.com/NVlabs/SegFormer
# cd SegFormer
# python setup.py install
# mkdir checkpoints
# cd checkpoints
# download checkpoints
# gdown --id 1z3eFf-xVMkcb1Nmcibv6Ut-lTh81RLgO
# gdown --id 1MZhqvWDOKdo5rBPC2sL6kWL25JpxOg38
# gdown --id 1PNaxIg3gAqtxrqTNsYPriR2c9j68umuj
# gdown --id 16ILNDrZrQRJrXsIcSjUC56ueR72Rlant
# gdown --id 11F7GHP6F8S9nUOf_KDvg8pouDEFEBGYz

# it will download these checkpoints, use this format SegFormr.b1-512-ade (for first one) in -t parameter to select them
# segformer.b1.512x512.ade.160k.pth
# segformer.b2.1024x1024.city.160k.pth
# segformer.b3.512x512.ade.160k.pth
# segformer.b5.1024x1024.city.160k.pth
# segformer.b5.640x640.ade.160k.pth


# Status: not working

pathToProject='../SegFormer/'
sys.path.insert(0, pathToProject)
os.chdir(pathToProject)

from mmseg.apis import init_segmentor, inference_segmentor

def init_model(transform):
    
    # use this format SegFormr.b1-512-ade  in -t parameter to select a certain config/checkpoint
    (version,resolution,datasetType) =  transform.split('-')
    config = "local_configs/segformer/"+version.upper()+"/segformer."+version+"."+resolution+"x"+resolution+"."+datasetType+".160k.py"
    checkpoint= "checkpoints/segformer."+version+"."+resolution+"x"+resolution+"."+datasetType+".160k.pth"

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, checkpoint, device='cuda:0')
     
    return (model),None


def process_image(transform,processing_model,img):
    tracks = []
    try:
        (model) = processing_model

        result = inference_segmentor(model, img)
        img = model.show_result(img, result, palette=None, show=False)

    except Exception as e:
        track = traceback.format_exc()
        print(track)
        print("SegFormer Exception",e)
        pass
                
    return tracks,img

