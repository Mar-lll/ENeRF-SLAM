import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.common import as_intrinsics_matrix
from torch.utils.data import Dataset
from torchvision import transforms, datasets
#import networks
#from layers import disp_to_depth
import PIL.Image as pil

def readEXR_onlydepth(filename):
    """
    Read depth data from EXR image file.

    Args:
        filename (str): File path.

    Returns:
        Y (numpy.array): Depth buffer in float32 format.
    """
    # move the import here since only CoFusion needs these package
    # sometimes installation of openexr is hard, you can run all other datasets
    # even without openexr
    import Imath
    import OpenEXR as exr

    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

    channelData = dict()

    for c in header['channels']:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)

        channelData[c] = C

    Y = None if 'R' not in header['channels'] else channelData['R']

    return Y


def get_dataset(cfg, args, scale, device='cuda:0'):
    return dataset_dict[cfg['dataset']](cfg, args, scale, device=device)


class BaseDataset(Dataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(BaseDataset, self).__init__()
        self.name = cfg['dataset']
        self.device = device
        self.scale = scale
        self.png_depth_scale = cfg['cam']['png_depth_scale']

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']

        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        if args.input_folder is None:
            self.input_folder = cfg['data']['input_folder']
        else:
            self.input_folder = args.input_folder

        self.crop_edge = cfg['cam']['crop_edge']
        self.dataset = cfg['dataset']

    def __len__(self):
        return self.n_img

    def readEXR_onlydepth(self,filename):
        """
        Read depth data from EXR image file.

        Args:
            filename (str): File path.

        Returns:
            Y (numpy.array): Depth buffer in float32 format.
        """
        # move the import here since only CoFusion needs these package
        # sometimes installation of openexr is hard, you can run all other datasets
        # even without openexr
        import Imath
        import OpenEXR as exr

        exrfile = exr.InputFile(filename)
        header = exrfile.header()
        dw = header['dataWindow']
        isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)

        channelData = dict()

        for c in header['channels']:
            C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
            C = np.frombuffer(C, dtype=np.float32)
            C = np.reshape(C, isize)

            channelData[c] = C

        Y = None if 'R' not in header['channels'] else channelData['R']
        if self.dataset == 'endomapper':
            far_=4.
            near_=0.01
            x=1.0-far_/near_
            y=far_/near_
            z=x/far_
            w=y/far_
            for i in range(dw.max.y - dw.min.y + 1):
                for j in range(dw.max.x - dw.min.x + 1):
                    Y[i][j]= 1./(z*(1-Y[i][j])+w)

        return Y

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        self.use_estimated_depth = False
        if self.use_estimated_depth:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            model_path = os.path.join(self.input_folder,"Model_MIA")

            print("-> Loading model from ", model_path)
            encoder_path = os.path.join(model_path, "encoder.pth")
            depth_decoder_path = os.path.join(model_path, "depth.pth")

            # LOADING PRETRAINED MODEL
            print("   Loading pretrained encoder")
            encoder = networks.ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(encoder_path, map_location=device)

            # extract the height and width of image that this model was trained with
            feed_height = loaded_dict_enc['height']
            feed_width = loaded_dict_enc['width']
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}

            encoder.load_state_dict(filtered_dict_enc)
            encoder.to(device)
            encoder.eval()

            print("   Loading pretrained decoder")
            depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

            loaded_dict = torch.load(depth_decoder_path, map_location=device)
            depth_decoder.load_state_dict(loaded_dict)

            depth_decoder.to(device)
            depth_decoder.eval()

            # FINDING INPUT IMAGES
            if os.path.isfile(color_path):
                # Only testing on a single image
                paths = [color_path]
                #output_directory = os.path.dirname(args.image_path)

            else:
                raise Exception("Can not find args.image_path: {}".format(color_path))

            print("-> Predicting on {:d} test images".format(len(paths)))

            # PREDICTING ON EACH IMAGE IN TURN
            with torch.no_grad():
                for idx, image_path in enumerate(paths):

                    if image_path.endswith("_disp.jpg"):
                        # don't try to predict disparity for a disparity image!
                        continue

                    # Load image and preprocess
                    input_image = pil.open(image_path).convert('RGB')

                    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                    # PREDICTION
                    input_image = input_image.to(device)
                    features = encoder(input_image)
                    outputs = depth_decoder(features)

                    disp = outputs[("disp", 0)]
                   
                    # Saving numpy file
                    #name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                    scaled_disp, _ = disp_to_depth(disp, 0.1, 150)
                    pred_disp = cv2.resize(scaled_disp.numpy(), (1280, 1024))
                    pred_depth = 1/pred_disp
                    depth_data = pred_depth * 19378.1
                    
        else:
            depth_path = self.depth_paths[index]
            if '.png' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            elif '.npy' in depth_path:
                pred_disp = np.load(depth_path).squeeze()
                pred_disp = cv2.resize(pred_disp, (1280, 1024))
                pred_depth = 1/pred_disp
                #depth_data = pred_depth * 19378.1
                depth_data = pred_depth * 20000
            elif '.exr' in depth_path:
                depth_data = self.readEXR_onlydepth(depth_path)
            elif 'tiff' in depth_path:
                depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        color_data = cv2.imread(color_path)
        
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
            # undistortion is only applied on color image, not depth!
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale
        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))
        color_data = torch.from_numpy(color_data)
        depth_data = torch.from_numpy(depth_data)*self.scale

        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.crop_edge
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        pose = self.poses[index]
        pose[:3, 3] *= self.scale

        return index, color_data.to(self.device), depth_data.to(self.device), pose.to(self.device)


class SYN(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(SYN, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.exr'))

        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = [] 
   
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
            for i in range(self.n_img):
                line = lines[i]

                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1

                c2w[1,3] *= -1
                c2w[0,1] *= -1
                c2w[1,0] *= -1
                c2w[1,2] *= -1
                c2w[2,1] *= -1

                #c2w[3,1] *= -1
                c2w[:3,3] /= 10
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
        else:
 
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)

class Scared_EST(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Scared_EST, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.npy'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = [] 
        
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
            for i in range(self.n_img):
                line = lines[i]

                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                c2w[:3,2]*=-1
                c2w[:3,3]*=-1

                c2w[:3,2] *= -1
                c2w[0,3]*=-1
                c2w[1,3]*=-1
                c2w[:3,3]/=34.133333
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)


class Hamlyn(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(Hamlyn, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/results/frame*.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/results/depth*.png'))
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/traj.txt')

    def load_poses(self, path):
        self.poses = []
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
            for i in range(self.n_img):
                line = lines[i]

                c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
                c2w[2,3] = -c2w[2,3]
                c2w[1,3] = -c2w[1,3]
                c2w[0,2]= -c2w[0,2]

                c2w[2,0]= -c2w[2,0]

                c2w[0,1] *= -1
                c2w[1,0] *= -1

                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
        else:
            for i in range(self.n_img):
                c2w = np.eye(4)
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)

class C3VD(BaseDataset):
    def __init__(self, cfg, args, scale, device='cuda:0'
                 ):
        super(C3VD, self).__init__(cfg, args, scale, device)
        self.color_paths = sorted(
            glob.glob(f'{self.input_folder}/color_undistorted/*.png'))
        self.depth_paths = sorted(
            glob.glob(f'{self.input_folder}/depth_undistorted/*_depth.tiff'))
        #self.color_paths = []
        #for i in range(len(self.depth_paths)):
        #    self.color_paths.append(self.input_folder+'/color_undistorted/'+str(i)+'_color.png')      
        self.n_img = len(self.color_paths)
        self.load_poses(f'{self.input_folder}/pose.txt')

    def load_poses(self, path):
        self.poses = [] 
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
            for i in range(self.n_img):
                line = lines[i]
                c2w = np.array(list(map(float, line.split(',')))).reshape(4, 4).T 
                c2w[:3, 1] *= -1
                c2w[:3, 2] *= -1
                c2w[:3,3] /= 10
                c2w = torch.from_numpy(c2w).float()
                self.poses.append(c2w)
        
dataset_dict = {
    "Hamlyn": Hamlyn,
    'SCARED_EST': Scared_EST,
    "syn": SYN,
    "c3vd": C3VD,
} 
 