"""A simple, flexible implementation of a face analysis tool."""

import math
import os

import onnx
import torch
import torch.nn.functional as F
import torchvision.ops as ops
from onnx2torch import convert
from skimage import transform as trans
from torchvision.transforms.functional import resize

arcface_dst = torch.tensor(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]]
).float()


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, axis=-1)


def face_transform(data, center, output_size, scale, rotation, device):
    def to_homogeneous(mat):
        return torch.vstack([mat, torch.tensor([0.0, 0.0, 1.0])])

    scale_ratio = scale
    rot = float(rotation) * math.pi / 180.0
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio

    C, H, W = data.shape

    t1 = to_homogeneous(torch.tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0]])).float()
    t2 = to_homogeneous(torch.tensor([[1, 0, -cx], [0, 1, -cy]])).float()
    cos_theta = math.cos(rot)
    sin_theta = math.sin(rot)
    t3 = to_homogeneous(torch.tensor([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0]])).float()
    t4 = to_homogeneous(torch.tensor([[1, 0, output_size / 2], [0, 1, output_size / 2]])).float()
    M_homogeneous = t4 @ t3 @ t2 @ t1
    M = M_homogeneous[:2, :]
    T = torch.tensor([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    theta = torch.inverse(T @ M_homogeneous @ torch.inverse(T))
    theta = theta[:2, :].unsqueeze(0).to(device)
    grid = F.affine_grid(theta, data.unsqueeze(0).size(), align_corners=True)
    transformed = F.grid_sample(data.unsqueeze(0), grid, align_corners=True)
    cropped = transformed[0]
    cropped = cropped[:, :output_size, :output_size]
    return cropped.unsqueeze(0), M


def trans_points2d(pts, M):
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)
    points_hom = torch.cat([pts, ones], dim=1)
    points_hom = points_hom.unsqueeze(-1)
    transformed_hom = torch.matmul(M, points_hom)
    transformed = transformed_hom[:, :2, :].squeeze(-1)
    return transformed


def estimate_norm(lmk, image_size=112, mode="arcface"):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = torch.from_numpy(tform.params).float()
    return M


def norm_crop(img, landmark, image_size=112, mode="arcface"):
    """Align an image into ArcFace canonical 112x112 using a similarity transform.

    Differentiable in ``img`` — the only non-grad path is ``landmark``, which
    comes from the no-grad SCRFD detector. This is what carries the REFL
    gradient from cosine reward all the way back to the generated pixels.
    """
    M_homogeneous = estimate_norm(landmark, image_size, mode)
    C, H, W = img.shape
    img = img.unsqueeze(0)
    T = torch.tensor([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
    T_inv = torch.inverse(T)
    theta = torch.inverse(T @ M_homogeneous @ T_inv)
    theta = theta[:2, :].unsqueeze(0).to(img.device)
    grid = F.affine_grid(theta, img.size(), align_corners=True)
    transformed = F.grid_sample(img, grid, align_corners=True)
    cropped = transformed[0]
    warped = cropped[:, :image_size, :image_size]
    return warped


def invert_affine_transform(matrix):
    L = matrix[..., :2]
    T = matrix[..., 2:]
    a, b = L[..., 0, 0], L[..., 0, 1]
    c, d = L[..., 1, 0], L[..., 1, 1]
    det = a * d - b * c
    inv_det = 1.0 / det
    inv_L = torch.stack(
        [torch.stack([d * inv_det, -b * inv_det], dim=-1), torch.stack([-c * inv_det, a * inv_det], dim=-1)], dim=-2
    )
    inv_T = -torch.matmul(inv_L, T)
    inv_matrix = torch.cat([inv_L, inv_T], dim=-1)
    return inv_matrix


class Face(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return torch.norm(self.embedding)

    @property
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm


class SCRFD:
    def __init__(self, model_file=None, device="cuda"):
        self.model_file = model_file
        self.device = device
        self.center_cache = {}
        model = onnx.load(self.model_file)
        self.torch_model = convert(model)
        self.torch_model.eval()
        self.torch_model.requires_grad_(False)
        self.torch_model.to(self.device)
        self.use_kps = True
        self.fmc = 3
        self._num_anchors = 2
        self._feat_stride_fpn = [8, 16, 32]
        self.input_size = (640, 640)

    def forward(self, det_img, threshold=0.5):
        input_height = det_img.shape[2]
        input_width = det_img.shape[3]
        scores_list = []
        bboxes_list = []
        kpss_list = []
        net_outs = self.torch_model(det_img.float())

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx].cpu()
            bbox_preds = net_outs[idx + self.fmc].cpu()
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + self.fmc * 2].cpu() * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                rows = torch.arange(height)
                cols = torch.arange(width)
                grid_y, grid_x = torch.meshgrid(rows, cols, indexing="ij")
                anchor_centers = torch.stack([grid_x, grid_y], dim=-1).float()
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = torch.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            scores = scores.reshape(-1, 1)
            bbox_preds = bbox_preds.reshape(-1, 4)
            if self.use_kps:
                kps_preds = kps_preds.reshape(-1, kps_preds.shape[-1])
            pos_mask = scores[:, 0] >= threshold
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_mask]
            pos_bboxes = bboxes[pos_mask]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_mask]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    @torch.no_grad()
    def detect(self, image, input_size=None, max_num=0, metric="default", nms_thresh=0.4, det_thresh=0.5):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(image.shape[1]) / image.shape[2]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / image.shape[1]
        resized_img = resize(image, (new_height, new_width), antialias=False)
        det_img = torch.zeros((3, input_size[1], input_size[0]), device=self.device)
        det_img[:, :new_height, :new_width] = resized_img
        det_img = det_img.unsqueeze(0)
        scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh)

        scores = torch.vstack(scores_list)
        scores_ravel = scores.flatten()
        order = torch.argsort(scores_ravel, descending=True)
        bboxes = torch.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = torch.vstack(kpss_list) / det_scale

        pre_det = torch.cat((bboxes, scores), dim=1).float()
        pre_det = pre_det[order]
        keep = self.nms(pre_det, nms_thresh)
        det = pre_det[keep, :]

        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        return det, kpss

    def nms(self, dets, nms_thresh):
        boxes = dets[:, :4]
        scores = dets[:, 4]
        keep = ops.nms(boxes, scores, iou_threshold=nms_thresh)
        return keep.tolist()


class ArcFace:
    def __init__(self, model_file=None, device="cuda"):
        self.model_file = model_file
        self.device = device
        model = onnx.load(self.model_file)
        self.torch_model = convert(model)
        self.torch_model.eval()
        self.torch_model.to(self.device)
        self.torch_model.requires_grad_(False)
        self.taskname = "recognition"
        self.input_size = (112, 112)

    def get(self, img, face, input_size=(112, 112)):
        aimg = norm_crop(img, landmark=face.kps, image_size=self.input_size[0])
        im_ratio = float(aimg.shape[1]) / aimg.shape[2]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        resized_img = resize(aimg, (new_height, new_width), antialias=False)
        face.embedding = self.get_feat(resized_img.unsqueeze(0)).flatten()
        return face.embedding

    def compute_sim(self, feat1, feat2):
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = torch.dot(feat1, feat2) / (torch.norm(feat1) * torch.norm(feat2))
        return sim

    def get_feat(self, imgs):
        imgs = imgs[:, [2, 1, 0], :, :]
        net_out = self.torch_model(imgs)
        return net_out


class Landmark:
    def __init__(self, model_file=None, device="cuda"):
        self.model_file = model_file
        self.device = device
        model = onnx.load(self.model_file)
        self.torch_model = convert(model)
        self.torch_model.eval()
        self.torch_model.to(device)
        self.torch_model.requires_grad_(False)
        self.lmk_dim = 2
        self.lmk_num = 106
        self.taskname = "landmark_%dd_%d" % (self.lmk_dim, self.lmk_num)
        self.input_size = (192, 192)

    def get(self, img, face, input_size=(192, 192)):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0] / (max(w, h) * 1.5)
        aimg, M = face_transform(img, center, self.input_size[0], _scale, rotate, img.device)
        aimg = (aimg + 1) / 2 * 255.0
        aimg = aimg[:, [2, 1, 0], :, :]

        input_size = self.input_size if input_size is None else input_size
        im_ratio = float(aimg.shape[2]) / aimg.shape[3]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        resized_img = resize(aimg, (new_height, new_width), antialias=False)
        det_img = torch.zeros((aimg.shape[0], 3, input_size[1], input_size[0]), device=self.device)
        det_img[:, :, :new_height, :new_width] = resized_img

        pred = self.torch_model(det_img)[0]
        pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1 :, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= self.input_size[0] // 2

        IM = invert_affine_transform(M).to(img.device)
        pred = trans_points2d(pred, IM)
        face[self.taskname] = pred
        return pred


class FaceAnalysis:
    def __init__(self, root="~/.insightface", device="cuda"):
        self.root = root
        self.device = device
        self.detection_root = os.path.join(root, "scrfd_10g_bnkps.onnx")
        self.landmark_root = os.path.join(root, "2d106det.onnx")
        self.arcface_root = os.path.join(root, "glintr100.onnx")
        self.detection_model = SCRFD(self.detection_root, self.device)
        self.landmark_model = Landmark(self.landmark_root, self.device)
        self.arcface_model = ArcFace(self.arcface_root, self.device)

    def landmark_loss(self, id_landmark=None, gt_landmark=None, mask=None):
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        error = torch.abs(id_landmark - gt_landmark) * mask
        valid_frame_count = mask.sum() + 1e-8
        loss = error.sum() / valid_frame_count / id_landmark.shape[-2]
        return loss

    def embedding_loss(self, id_embedding=None, gt_embedding=None, mask=None):
        cos_sim = F.cosine_similarity(id_embedding, gt_embedding, dim=2)
        cos_loss = (1 - cos_sim) * mask
        valid_frame_count = mask.sum() + 1e-8
        loss = cos_loss.sum() / valid_frame_count
        return loss

    def pool_embedding_loss(self, id_embedding=None, gt_embedding=None, id_mask=None):
        """Pool-style cosine similarity between gen frames and any ref frame.

        Returns a scalar reward (per-call). The scorer calls this once per
        sample and stacks the result into a per-sample reward tensor.
        """
        id_emb_expanded = id_embedding.unsqueeze(2)
        gt_emb_expanded = gt_embedding.unsqueeze(1)
        gt_mask = torch.ones(gt_embedding.shape[0], gt_embedding.shape[1]).to(id_mask.device)
        if gt_mask.shape[1] > 1:
            gt_mask[:, 0] = 0
        is_all_zero = (gt_embedding == 0).all(dim=-1)
        gt_mask[is_all_zero] = 0

        cos_sim_all = F.cosine_similarity(id_emb_expanded, gt_emb_expanded, dim=3)
        valid_mask = id_mask.unsqueeze(2) * gt_mask.unsqueeze(1)

        gt_valid_count = gt_mask.sum(dim=1) + 1e-8
        weight_matrix = valid_mask / (gt_valid_count.unsqueeze(1).unsqueeze(2) + 1e-8)
        mean_similarities = (cos_sim_all * weight_matrix).sum(dim=2)
        cos_loss = mean_similarities * id_mask
        valid_frame_count = id_mask.sum() + 1e-8
        loss = cos_loss.sum() / valid_frame_count
        return loss
