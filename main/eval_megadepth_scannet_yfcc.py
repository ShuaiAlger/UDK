
import os
from tokenize import Double
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from megadepth import MegaDepthDataset
from scannet import ScanNetDataset
from match_metrics import compute_pose_errors, compute_symmetrical_epipolar_errors, convert_points_to_homogeneous, aggregate_metrics

from sg_eval_utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

from recm_tools import add_ones


from pathlib import Path




from yfcc import YFCCDataset, YFCCDatasetRot90
def get_yfcc_loader():
    datas = DataLoader(YFCCDataset(), batch_size=1, shuffle=True, drop_last=False, num_workers=2)
    print(datas.__len__())
    return datas

def get_yfcc_rot_loader():
    datas = DataLoader(YFCCDatasetRot90(), batch_size=1, shuffle=True, drop_last=False, num_workers=2)
    print(datas.__len__())
    return datas



def get_scannet_loader():
    data_root_dir = './scannet/test'
    TEST_BASE_PATH = "assets/scannet_test_1500"
    scene_list_path = f"{TEST_BASE_PATH}/scannet_test.txt"
    npz_dir = f"{TEST_BASE_PATH}"
    intrinsic_path = f"{TEST_BASE_PATH}/intrinsics.npz"
    
    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]

    sets = []
    for npz_name in tqdm(npz_names):
        # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
        npz_path = os.path.join(npz_dir, npz_name)

        sets.append(ScanNetDataset(root_dir=data_root_dir, npz_path=npz_path, intrinsic_path=intrinsic_path, mode='test', min_overlap_score=0.0))

    datas = DataLoader(sets[0], batch_size=1, shuffle=True, drop_last=False, num_workers=2)
    return datas


def get_megadepth_loader():
    data_root_dir = '/media/shuai/Correspondence/DATASETS/megadepth/test'
    TEST_BASE_PATH = "/media/shuai/Correspondence/DATASETS/assets/megadepth_test_1500_scene_info"
    scene_list_path = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"
    npz_dir = f"{TEST_BASE_PATH}"
    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
        npz_names = [f'{n}.npz' for n in npz_names]
    
    sets = []
    for npz_name in tqdm(npz_names):
        # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
        npz_path = os.path.join(npz_dir, npz_name)
        # sets.append(MegaDepthDataset(root_dir=data_root_dir, npz_path=npz_path, mode='test', min_overlap_score=0.0, img_resize=840, img_padding=True, depth_padding=True, df=8))
        sets.append(MegaDepthDataset(root_dir=data_root_dir, npz_path=npz_path, mode='test', min_overlap_score=0.0, img_resize=1200, img_padding=True, depth_padding=True, df=8))

    datas = DataLoader(sets[0]+sets[1]+sets[2]+sets[3]+sets[4], batch_size=1, shuffle=False, drop_last=False, num_workers=2)
    return datas



def compute_metrics(batch):
    with torch.no_grad():
        compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
        compute_pose_errors(batch)  # compute R_errs, t_errs, pose_errs for each pair

        rel_pair_names = list(zip(*batch['pair_names']))
        bs = batch['image0'].size(0)
        metrics = {
            # to filter duplicate pairs caused by DistributedSampler
            'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
            'epi_errs': [batch['epi_errs'].cpu().numpy() for b in range(bs)],
            'R_errs': batch['R_errs'],
            't_errs': batch['t_errs'],
            'inliers': batch['inliers']}
        ret_dict = {'metrics': metrics}
    return ret_dict, rel_pair_names



from itertools import chain
def flattenList(x):
    return list(chain(*x))
from comm import gather



def test_epoch_end(outputs):
    # metrics: dict of list, numpy
    _metrics = [o['metrics'] for o in outputs]
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4) # TRAINER.EPI_ERR_THR
    
    # print(val_metrics_4tb)
    return str(val_metrics_4tb)



import numpy as np
import cv2


from sg_eval_utils import rotate_intrinsics, rotate_pose_inplane
from recm_tools import rotate_image_bound_with_M



import copy

def draw_green_red(_img1, _img2, _m_pts1_src, _m_pts2_src, _correct, _thinkness):

    canvas = 255*np.ones((max(_img1.shape[0],_img2.shape[0]), _img1.shape[1]+_img2.shape[1], 3), dtype=np.uint8)

    _m_pts1 = copy.deepcopy(_m_pts1_src.copy())
    _m_pts2 = copy.deepcopy(_m_pts2_src.copy())

    canvas[0:_img1.shape[0], 0:_img1.shape[1], :] = _img1
    canvas[0:_img2.shape[0], _img1.shape[1]:_img1.shape[1]+_img2.shape[1], :] = _img2
    
    N_ = len(_m_pts1)

    for i in range(N_):
        color = (0, 0, 255)
        if _correct[i]:
            color = (0, 255, 0)
        cv2.line(canvas, (int(_m_pts1[i, 0]), int(_m_pts1[i, 1])), (int(_m_pts2[i, 0]+_img1.shape[1]), int(_m_pts2[i, 1])), color, _thinkness)
    return canvas



def evaluate_megadepth(matcher=None, USE_ROT90_DATASET = 0, USE_CONTINUE_ROT_DATASET = 0):
    np.random.seed(0)
    datas = get_megadepth_loader()
    print("Get Data Loader")

    ret_dicts = []

    pose_errors = []
    precisions = []
    matching_scores = []


    with tqdm(total=len(datas)) as _tqdm:
        for i, data in enumerate(tqdm(datas)):

            img0 = data['image0']
            img1 = data['image1']

            img0 = img0[0].cpu().numpy().transpose((1, 2, 0))
            img1 = img1[0].cpu().numpy().transpose((1, 2, 0))



            rot0 = 0
            rot1 = 0
            
            if USE_ROT90_DATASET:
                random_rotation = np.random.random()
                if 0 < random_rotation and random_rotation <= 0.25:
                    rot1 = 0
                if 0.25 < random_rotation and random_rotation <= 0.5:
                    rot1 = 1 
                if 0.5 < random_rotation and random_rotation <= 0.75:
                    rot1 = 2
                if 0.75 < random_rotation and random_rotation <= 1.0:
                    rot1 = 3

                if rot1 == 1:
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                if rot1 == 2:
                    img1 = cv2.rotate(img1, cv2.ROTATE_180)
                if rot1 == 3:
                    img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)


            M = None
            M_inv = None
            if USE_CONTINUE_ROT_DATASET:
                random_rotation = np.random.random()
                random_rotation = 360 * random_rotation
                img1, M = rotate_image_bound_with_M(img1, random_rotation)
                M = np.row_stack((M, np.array([0, 0, 1])))
                M_inv = np.mat(np.linalg.inv(M))

            
            mkpts0, mkpts1 = matcher(img0, img1)
            mkpts0 = torch.from_numpy(mkpts0).float()
            mkpts1 = torch.from_numpy(mkpts1).float()
            data.update({
                'expec_f': torch.empty(0, 3, device='cuda'),
                'mkpts0_f': mkpts0,
                'mkpts1_f': mkpts1,
            })



            T_0to1 = data['T_0to1'][0].cpu().numpy()
            scales0 = data['scale0'][0].cpu().numpy()
            scales1 = data['scale1'][0].cpu().numpy()
            K0 = data['K0'][0].cpu().numpy()
            K1 = data['K1'][0].cpu().numpy()
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)





            if USE_ROT90_DATASET:
                if rot0 != 0 or rot1 != 0:
                    cam0_T_w = np.eye(4)
                    cam1_T_w = T_0to1
                    if rot0 != 0:
                        K0 = rotate_intrinsics(K0, img0.shape, rot0)
                        cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                    if rot1 != 0:
                        K1 = rotate_intrinsics(K1, img1.shape, rot1)
                        cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                    cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                    T_0to1 = cam1_T_cam0


            mkpts0 = mkpts0.cpu().numpy()
            mkpts1 = mkpts1.cpu().numpy()

            img0 = np.asarray(img0, dtype=np.uint8)
            img1 = np.asarray(img1, dtype=np.uint8)

            mkpts1_copy = copy.deepcopy(mkpts1.copy())


            if USE_CONTINUE_ROT_DATASET:
                hmkpts1 = add_ones(mkpts1)
                mkpts1 = (M_inv * hmkpts1.T).A.T[:, 0:2]


            kpts0 = mkpts0
            kpts1 = mkpts1

            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4



            cv2.imwrite("canvas_res/"+str(i)+".png", draw_green_red(img0, img1, mkpts0, mkpts1_copy, correct, 1))



            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            thresh = 1.  # In pixels relative to resized image size.
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            # eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
            # np.savez(str(eval_path), **out_eval)

            pose_error = np.maximum(out_eval['error_t'], out_eval['error_R'])
            pose_errors.append(pose_error)
            precisions.append(out_eval['precision'])

            # for pair in pairs:
            #     name0, name1 = pair[:2]
            #     stem0, stem1 = Path(name0).stem, Path(name1).stem
            #     eval_path = output_dir / \
            #         '{}_{}_evaluation.npz'.format(stem0, stem1)
            #     results = np.load(eval_path)
            #     pose_error = np.maximum(results['error_t'], results['error_R'])
            #     pose_errors.append(pose_error)
            #     precisions.append(results['precision'])
            #     matching_scores.append(results['matching_score'])


            thresholds = [5, 10, 20]
            aucs = pose_auc(pose_errors, thresholds)
            aucs = [100.*yy for yy in aucs]
            prec = 100.*np.mean(precisions)
            ms = 100.*np.mean(matching_scores)

            res = "EvalResults over {} pairs):".format(len(datas))
            res = res + "AUC@5 AUC@10 AUC@20 Prec MScore   "
            res = res + "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(
                aucs[0], aucs[1], aucs[2], prec, ms)

            _tqdm.set_postfix(results=str(res))



def evaluate_scannet(matcher=None):
    datas = get_scannet_loader()
    print("Get Data Loader")

    ret_dicts = []
    with tqdm(total=len(datas)) as _tqdm:
        for data in tqdm(datas):
            img0 = data['image0'][0].cpu().numpy()
            img1 = data['image1'][0].cpu().numpy()

            img0 = np.asarray(img0, dtype=np.uint8)
            img1 = np.asarray(img1, dtype=np.uint8)

            cv2.imshow("img0", img0)
            cv2.imshow("img1", img1)
            cv2.waitKey(10)

            mkpts0, mkpts1 = matcher(img0, img1)

            data.update({
                'expec_f': torch.empty(0, 3, device='cuda'),
                'mkpts0_f': torch.from_numpy(mkpts0).float(),
                'mkpts1_f': torch.from_numpy(mkpts1).float(),
            })
            ret_dict, rel_pair_names = compute_metrics(data)
            # print(ret_dict['metrics']['R_errs'], ret_dict['metrics']['t_errs'])  # R_errs t_errs inliers
            ret_dicts.append(ret_dict)

            res = test_epoch_end(ret_dicts)

            _tqdm.set_postfix(results=str(res))


def evaluate_yfcc(matcher=None, rot90=True):
    if not rot90:
        datas = get_yfcc_loader()
    if rot90:
        datas = get_yfcc_rot_loader()

    ret_dicts = []
    with tqdm(total=len(datas)) as _tqdm:
        for data in tqdm(datas):
            img0 = data['image0'][0].cpu().numpy()
            img1 = data['image1'][0].cpu().numpy()

            img0 = np.asarray(img0, dtype=np.uint8)
            img1 = np.asarray(img1, dtype=np.uint8)

            cv2.imshow("img0", img0)
            cv2.imshow("img1", img1)
            cv2.waitKey(10)

            mkpts0, mkpts1 = matcher(img0, img1)


            mkpts0 = torch.from_numpy(mkpts0).float()
            mkpts1 = torch.from_numpy(mkpts1).float()


            data.update({
                'expec_f': torch.empty(0, 3, device='cuda'),
                'mkpts0_f': mkpts0,
                'mkpts1_f': mkpts1,
            })
            ret_dict, rel_pair_names = compute_metrics(data)

            # print(ret_dict['metrics']['R_errs'], ret_dict['metrics']['t_errs'])  # R_errs t_errs inliers

            ret_dicts.append(ret_dict)

            res = test_epoch_end(ret_dicts)

            _tqdm.set_postfix(results=res)

    test_epoch_end(ret_dicts)






def evaluate_yfcc_rot_rand(matcher=None, USE_CONTINUE_ROT_DATASET=True):
    datas = get_yfcc_loader()

    # datas = get_yfcc_rot_loader()

    ret_dicts = []
    with tqdm(total=len(datas)) as _tqdm:
        for data in tqdm(datas):
            img0 = data['image0'][0].cpu().numpy()
            img1 = data['image1'][0].cpu().numpy()

            img0 = np.asarray(img0, dtype=np.uint8)
            img1 = np.asarray(img1, dtype=np.uint8)


            M = None
            M_inv = None
            if USE_CONTINUE_ROT_DATASET:
                random_rotation = np.random.random()
                random_rotation = 360 * random_rotation
                img1, M = rotate_image_bound_with_M(img1, random_rotation)
                M = np.row_stack((M, np.array([0, 0, 1])))
                M_inv = np.mat(np.linalg.inv(M))





            mkpts0, mkpts1 = matcher(img0, img1)


            if USE_CONTINUE_ROT_DATASET:
                hmkpts1 = add_ones(mkpts1)
                mkpts1 = (M_inv * hmkpts1.T).A.T[:, 0:2]




            mkpts0 = torch.from_numpy(mkpts0).float()
            mkpts1 = torch.from_numpy(mkpts1).float()




            data.update({
                'expec_f': torch.empty(0, 3, device='cuda'),
                'mkpts0_f': mkpts0,
                'mkpts1_f': mkpts1,
            })
            ret_dict, rel_pair_names = compute_metrics(data)

            # print(ret_dict['metrics']['R_errs'], ret_dict['metrics']['t_errs'])  # R_errs t_errs inliers

            ret_dicts.append(ret_dict)

            res = test_epoch_end(ret_dicts)

            _tqdm.set_postfix(results=res)

    test_epoch_end(ret_dicts)






if __name__ == '__main__':

    pass

    # evaluate_megadepth()

    # evaluate_scannet()



