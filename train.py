import torch
import numpy as np
import cv2
import warnings
import torch.nn as nn
import torch.optim as optim

from utils import read_image, preprocess
from dkm.models.model_zoo.DKMv3 import DKMv3
from CustomDataset import CustomDataset
from torch.utils.data import DataLoader
from utils import  compute_geom
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything.dpt import DepthAnything
from loss import SSIM
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch_size = 50

    ######### Pose model
    # weights path
    checkpoints_path = "/home/wdkang/code/CustomUnsupervised/weights/pose/gim_dkm_100h.ckpt"
    pose_model = DKMv3(weights=None, h=672, w=896)
    # load state dict
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        if 'encoder.net.fc' in k:
            state_dict.pop(k)
    # load state dict
    pose_model.load_state_dict(state_dict)
    pose_model.to(device)

    ######### Depth model
    transform = Compose([
        Resize(
            width=512,
            height=512,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    vit_version = "vits"
    depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(vit_version)).to(device)

    csv_path = "/home/wdkang/code/CustomUnsupervised/test.csv"
    train_dataset = CustomDataset(csv_path)
    test_dataset = CustomDataset(csv_path, False)

    train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)


    criterion1 = SSIM()
    criterion2 = nn.L1Loss()
    optimizer = optim.AdamW(depth_model.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch_size)
    

    for epoch in tqdm(range(epoch_size)):
        for train_idx, (train0, train1) in enumerate(train_loader):
            
            ########################################################################
            ########################################################################
            ########################################################################
            # Pose Estimation
            train0_pose = train0.squeeze(0).detach().cpu().numpy()
            train1_pose = train1.squeeze(0).detach().cpu().numpy()
            image0, scale0 = preprocess(train0_pose)
            image1, scale1 = preprocess(train1_pose)
            image0 = image0.to(device)[None]
            image1 = image1.to(device)[None]
            data = dict(color0=image0, color1=image1, image0=image0, image1=image1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dense_matches, dense_certainty = pose_model.match(image0, image1)
                sparse_matches, mconf = pose_model.sample(dense_matches, dense_certainty, 5000)

            height0, width0 = image0.shape[-2:]
            height1, width1 = image1.shape[-2:]
            kpts0 = sparse_matches[:, :2]
            kpts0 = torch.stack((
                width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
            kpts1 = sparse_matches[:, 2:]
            kpts1 = torch.stack((
                width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)
            # robust fitting
            _, mask = cv2.findFundamentalMat(kpts0.cpu().numpy(),
                                            kpts1.cpu().numpy(),
                                            cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                            confidence=0.999999, maxIters=10000)
            mask = mask.ravel() > 0

            b_ids = torch.where(mconf[None])[0]

            data.update({
                'hw0_i': image0.shape[-2:],
                'hw1_i': image1.shape[-2:],
                'mkpts0_f': kpts0,
                'mkpts1_f': kpts1,
                'm_bids': b_ids,
                'mconf': mconf,
                'inliers': mask,
            })

        
            geom_info = compute_geom(data)
            fundamental = np.array(geom_info["Fundamental"])
            homography = np.array(geom_info["Homography"])
            H1 = np.array(geom_info["H1"])
            H2 = np.array(geom_info["H2"])
            K = np.load("/home/wdkang/code/unsupervised_depth/gim/camera_front_K.npy")
            
            E = np.dot(np.dot(K.T, fundamental), K)

            # Essential Matrix를 특이값 분해하여 회전과 변환 행렬 추출
            U, _, Vt = np.linalg.svd(E)
            W = np.array([[0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])

            # 회전 행렬과 변환 벡터 계산
            R1 = np.dot(np.dot(U, W), Vt)
            R2 = np.dot(np.dot(U, W.T), Vt)

            if np.trace(R1) < 0:
                R = R2
            else:
                R = R1
            
            t = U[:, 2]

            t = np.expand_dims(t, axis =1)
            P = np.hstack([R, t])
            ########################################################################
            ########################################################################
            ########################################################################
            # 2. Depth estimation
            ref_img_processing = transform({"image" : train0_pose})["image"]
            tar_img_processing = transform({"image" : train1_pose})["image"]
            
            ref_img_processing = torch.from_numpy(ref_img_processing).to(device)
            tar_img_processing = torch.from_numpy(tar_img_processing).to(device)
            ref_img_processing = ref_img_processing.unsqueeze(0)
            tar_img_processing = tar_img_processing.unsqueeze(0)
            # Depth prediction
            ref_depth = depth_model(ref_img_processing)
            #TODO : depth to pointcouds function
            batch_size, width, height = ref_depth.size()
            batch_size, width, height = ref_depth.size()
            ref_depth = ref_depth.squeeze(0)
            u_coord = torch.arange(width, dtype=torch.float32, device=ref_depth.device)
            v_coord = torch.arange(height, dtype=torch.float32, device=ref_depth.device)
            u, v = torch.meshgrid(u_coord, v_coord)  # (H, W)
            x_normalized = (u - K[0, 2]) / K[0, 0]
            y_normalized = (v - K[1, 2]) / K[1, 1]
            # Compute 3D coordinates
            x = x_normalized * ref_depth
            y = y_normalized * ref_depth
            z = ref_depth
            point_cloud = torch.stack((x, y, z), dim=-1).view(batch_size, -1, 3)

            ####rotation + projection
            K_tensor = torch.from_numpy(K)
            P_transpose = P.T
            P_tensor = torch.from_numpy(P_transpose)
            K_tensor = K_tensor.to(device).unsqueeze(0).to(torch.float)
            P_tensor = P_tensor.to(device).unsqueeze(0).to(torch.float)
            
            rotation_points = torch.bmm(point_cloud, P_tensor[:, :3, :])
            
            projected_points = torch.bmm(point_cloud, K_tensor) # batch X N X3 | batch X 3 X 3
            
            # 투영된 좌표를 이미지의 크기에 맞게 조정
            projected_points[:, 0] /= projected_points[:, 2]
            projected_points[:, 1] /= projected_points[:, 2]
            # 이미지의 크기에 맞게 좌표를 조정
            projected_points[:, 0] = (projected_points[:, 0] + 0.5) * width
            projected_points[:, 1] = (projected_points[:, 1] + 0.5) * height


            # Move color information from reference image to target image
            # Extract color information from reference image
            ref_img_colors = ref_img_processing.reshape(1, -1, 3)

            # Add color information to the point cloud
            point_cloud_with_color = torch.cat((point_cloud, ref_img_colors), dim=-1)
            tar_img_with_color = train1_pose.copy()
            
            # Move color information from reference image to target image
            tar_img_with_color = tar_img_processing
            for i in range(batch_size):
                for j in range(projected_points.shape[1]):
                    xyz = projected_points[i, j]  # Get 3D coordinates
                    x_float, y_float = xyz[0].item(), xyz[1].item()
                    # Check if x and y values are NaN
                    if not (np.isnan(x_float) or np.isnan(y_float)):
                        x, y = int(round(x_float)), int(round(y_float))  # Convert tensor to integer and round
                        if 0 <= x < tar_img_with_color.shape[3] and 0 <= y < tar_img_with_color.shape[2]:
                            color_info = point_cloud_with_color[i, j, 3:].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                            tar_img_with_color[0, :, y, x] = color_info  # Assign color information

            
            optimizer.zero_grad()
            ssim_loss = criterion1(tar_img_with_color, tar_img_processing)
            ssim_loss = 1 / ssim_loss
            l1_loss = criterion2(tar_img_with_color, tar_img_processing)
            total_loss = ssim_loss * 0.5 + l1_loss * 0.5
            total_loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            for test_idx, (test0, test1) in enumerate(test_loader):
                ########################################################################
                ########################################################################
                ########################################################################
                # Pose Estimation
                test0_pose = test0.squeeze(0).detach().cpu().numpy()
                test1_pose = test1.squeeze(0).detach().cpu().numpy()
                image0, scale0 = preprocess(test0_pose)
                image1, scale1 = preprocess(test1_pose)
                image0 = image0.to(device)[None]
                image1 = image1.to(device)[None]
                data = dict(color0=image0, color1=image1, image0=image0, image1=image1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dense_matches, dense_certainty = pose_model.match(image0, image1)
                    sparse_matches, mconf = pose_model.sample(dense_matches, dense_certainty, 5000)

                height0, width0 = image0.shape[-2:]
                height1, width1 = image1.shape[-2:]

                kpts0 = sparse_matches[:, :2]
                kpts0 = torch.stack((
                    width0 * (kpts0[:, 0] + 1) / 2, height0 * (kpts0[:, 1] + 1) / 2), dim=-1,)
                kpts1 = sparse_matches[:, 2:]
                kpts1 = torch.stack((
                    width1 * (kpts1[:, 0] + 1) / 2, height1 * (kpts1[:, 1] + 1) / 2), dim=-1,)

                # robust fitting
                _, mask = cv2.findFundamentalMat(kpts0.cpu().numpy(),
                                                kpts1.cpu().numpy(),
                                                cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                                confidence=0.999999, maxIters=10000)
                mask = mask.ravel() > 0

                b_ids = torch.where(mconf[None])[0]

                data.update({
                    'hw0_i': image0.shape[-2:],
                    'hw1_i': image1.shape[-2:],
                    'mkpts0_f': kpts0,
                    'mkpts1_f': kpts1,
                    'm_bids': b_ids,
                    'mconf': mconf,
                    'inliers': mask,
                })

            
                geom_info = compute_geom(data)
                fundamental = np.array(geom_info["Fundamental"])
                homography = np.array(geom_info["Homography"])
                H1 = np.array(geom_info["H1"])
                H2 = np.array(geom_info["H2"])
                K = np.load("/home/wdkang/code/unsupervised_depth/gim/camera_front_K.npy")
                
                E = np.dot(np.dot(K.T, fundamental), K)

                # Essential Matrix를 특이값 분해하여 회전과 변환 행렬 추출
                U, _, Vt = np.linalg.svd(E)
                W = np.array([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])

                # 회전 행렬과 변환 벡터 계산
                R1 = np.dot(np.dot(U, W), Vt)
                R2 = np.dot(np.dot(U, W.T), Vt)

                if np.trace(R1) < 0:
                    R = R2
                else:
                    R = R1
                
                t = U[:, 2]

                t = np.expand_dims(t, axis =1)
                P = np.hstack([R, t])
                ########################################################################
                ########################################################################
                ########################################################################
                # 2. Depth estimation
                ref_img_processing = transform({"image" : test0_pose})["image"]
                tar_img_processing = transform({"image" : test1_pose})["image"]

                ref_img_processing = torch.from_numpy(ref_img_processing).to(device)
                tar_img_processing = torch.from_numpy(tar_img_processing).to(device)
                ref_img_processing = ref_img_processing.unsqueeze(0)
                tar_img_processing = tar_img_processing.unsqueeze(0)


                # Depth prediction
                ref_depth = depth_model(ref_img_processing)
                #TODO : depth to pointcouds function
                batch_size, width, height = ref_depth.size()
                batch_size, width, height = ref_depth.size()
                ref_depth = ref_depth.squeeze(0)
                u_coord = torch.arange(width, dtype=torch.float32, device=ref_depth.device)
                v_coord = torch.arange(height, dtype=torch.float32, device=ref_depth.device)
                u, v = torch.meshgrid(u_coord, v_coord)  # (H, W)
                x_normalized = (u - K[0, 2]) / K[0, 0]
                y_normalized = (v - K[1, 2]) / K[1, 1]
                # Compute 3D coordinates
                x = x_normalized * ref_depth
                y = y_normalized * ref_depth
                z = ref_depth
                point_cloud = torch.stack((x, y, z), dim=-1).view(batch_size, -1, 3)

                ####rotation + projection
                K_tensor = torch.from_numpy(K)
                P_transpose = P.T
                P_tensor = torch.from_numpy(P_transpose)
                K_tensor = K_tensor.to(device).unsqueeze(0).to(torch.float)
                P_tensor = P_tensor.to(device).unsqueeze(0).to(torch.float)
                
                rotation_points = torch.bmm(point_cloud, P_tensor[:, :3, :])
                
                projected_points = torch.bmm(point_cloud, K_tensor) # batch X N X3 | batch X 3 X 3
                
                # 투영된 좌표를 이미지의 크기에 맞게 조정
                projected_points[:, 0] /= projected_points[:, 2]
                projected_points[:, 1] /= projected_points[:, 2]
                # 이미지의 크기에 맞게 좌표를 조정
                projected_points[:, 0] = (projected_points[:, 0] + 0.5) * width
                projected_points[:, 1] = (projected_points[:, 1] + 0.5) * height


                # Move color information from reference image to target image
                # Extract color information from reference image
                ref_img_colors = ref_img_processing.reshape(1, -1, 3)

                # Add color information to the point cloud
                point_cloud_with_color = torch.cat((point_cloud, ref_img_colors), dim=-1)
                tar_img_with_color = test1_pose.copy()
                
                # Move color information from reference image to target image
                tar_img_with_color = tar_img_processing
                for i in range(batch_size):
                    for j in range(projected_points.shape[1]):
                        xyz = projected_points[i, j]  # Get 3D coordinates
                        x_float, y_float = xyz[0].item(), xyz[1].item()
                        # Check if x and y values are NaN
                        if not (np.isnan(x_float) or np.isnan(y_float)):
                            x, y = int(round(x_float)), int(round(y_float))  # Convert tensor to integer and round
                            if 0 <= x < tar_img_with_color.shape[3] and 0 <= y < tar_img_with_color.shape[2]:
                                color_info = point_cloud_with_color[i, j, 3:].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                                tar_img_with_color[0, :, y, x] = color_info  # Assign color information

                    
                ssim_loss = criterion1(tar_img_with_color, tar_img_processing)
                ssim_loss = 1/ssim_loss
                l1_loss = criterion2(tar_img_with_color, tar_img_processing)
                total_loss = ssim_loss* 0.5 + l1_loss* 0.5
        scheduler.step()