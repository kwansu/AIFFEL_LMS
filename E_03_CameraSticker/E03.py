import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import dlib
import math
import os

from numpy.core.numeric import cross


def calc_perspective_image(image, rect, uv):
    rect -= rect[0]
    src_rect =np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]],
                        [image.shape[1], image.shape[0]]])

    transform = cv2.getPerspectiveTransform(src_rect, rect)
    _top_left = np.min(rect, axis=0)

    translate = np.eye(3)
    translate[0, 2] = -_top_left[0]
    translate[1, 2] = -_top_left[1]
    transform = np.dot(translate, transform)
    return cv2.warpPerspective(image, transform, uv.astype(np.int32))


def blend_sticker(img_source, img_sticker, x, y, uv, alpha=1):
    x_si = max(0, x)
    x_ei = min(img_source.shape[1], x+uv[0])
    y_si = max(0, y)
    y_ei = min(img_source.shape[0], y+uv[1])

    y_e = y_ei-y-uv[1] if y_ei < y+uv[1] else None
    x_e = x_ei-x-uv[0] if x_ei < x+uv[0] else None

    alpha_mask = img_sticker[y_si-y:y_e, x_si-x:x_e, -1]/255
    alpha_mask = alpha_mask.reshape(alpha_mask.shape + (1,))
    img_sticker = img_sticker[y_si-y:y_e, x_si-x:x_e, :-1]

    sticker_area = img_source[y_si:y_ei, x_si:x_ei]
    img_source[y_si:y_ei, x_si:x_ei] = sticker_area*(1-alpha_mask*alpha) + img_sticker*alpha_mask*alpha


#######################################################################

my_image_path = r'C:\Users\kwansu\Desktop\AIFFEL_LMS\E_03_CameraSticker\data\face5.png'
img_bgr = cv2.imread(my_image_path) 

detector_hog = dlib.get_frontal_face_detector()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)

print(f"얼굴 영역 좌표 {dlib_rects}")   # 찾은 얼굴영역 좌표


# 68 랜드마크
model_path = r'C:\Users\kwansu\Desktop\AIFFEL_LMS\E_03_CameraSticker\data\shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)

list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)
    landmarks = list_landmarks[0]

print(f"인식한 랜드마크 개수 : {len(list_landmarks[0])}")




# for idx, point in enumerate(list_points):
#     cv2.circle(img_bgr, point, 2, (0, 255, 255), -1) # yellow

# img_show_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)



sticker_path = r'C:\Users\kwansu\Desktop\AIFFEL_LMS\E_03_CameraSticker\data\cat2.png'
img_cat = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
print(img_cat.shape)
img_cat = cv2.resize(img_cat, (700,200))
half = img_cat.shape[1] // 2
img_cat_left = img_cat[:,:half]
img_cat_right = img_cat[:,half:]



left_rect = np.zeros([4,2], dtype=np.float32)
left_rect[0] = landmarks[1]
left_rect[1] = landmarks[30]
left_rect[2] = landmarks[3]
left_rect[3] = landmarks[51]

top_left = np.min(left_rect, axis=0).astype(np.int32)
uv = np.max(left_rect, axis=0).astype(np.int32) - top_left

img_cat_left = calc_perspective_image(img_cat_left, left_rect, uv)
blend_sticker(img_bgr, img_cat_left, int(top_left[0]), int(top_left[1]), uv, alpha=0.6)


right_rect = np.zeros([4,2], dtype=np.float32)
right_rect[0] = landmarks[30]
right_rect[1] = landmarks[15]
right_rect[2] = landmarks[51]
right_rect[3] = landmarks[13]

top_left = np.min(right_rect, axis=0).astype(np.int32)
uv = np.max(right_rect, axis=0).astype(np.int32) - top_left

img_cat_right = calc_perspective_image(img_cat_right, right_rect, uv)
blend_sticker(img_bgr, img_cat_right, top_left[0], top_left[1], uv, alpha=0.6)


sticker_path = r'C:\Users\kwansu\Desktop\AIFFEL_LMS\E_03_CameraSticker\data\king.png'
img_crown = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)

crown_rect = np.zeros([4,2], dtype=np.float32)
crown_rect[2] = landmarks[0]
crown_rect[3] = landmarks[16]
cross_v = crown_rect[3] - crown_rect[2]
cross_h = np.array((cross_v[1], -cross_v[0]))
#cross_h *= 1 if cross_v[0]>1 else -1
# width = float(np.sqrt(np.sum(cross_v**2)))
# height = (width/img_crown.shape[0] * img_crown.shape[1])
#cross_h = cross_v * height / width

crown_rect[0] = landmarks[27]
crown_rect[1] = landmarks[28]
crown_rect[1] = crown_rect[0] - crown_rect[1]
ori = crown_rect[0] + 2*crown_rect[1]

crown_rect[2] = ori - cross_v/2
crown_rect[3] = ori + cross_v/2
crown_rect[0] = crown_rect[2] + cross_h
crown_rect[1] = crown_rect[3] + cross_h
top_left = np.min(crown_rect, axis=0).astype(np.int32)
uv = np.max(crown_rect, axis=0).astype(np.int32) - top_left

img_crown = calc_perspective_image(img_crown, crown_rect, uv)
blend_sticker(img_bgr, img_crown, top_left[0], top_left[1], uv, alpha=0.6)

plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()