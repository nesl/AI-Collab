import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import math
import logging
class BEV:
    def __init__(self, f, img_H, img_W, cx, cy, z):
        self.focal_length = f
        self.image_height = img_H
        self.image_width = img_W
        self.image_cx = cx
        self.image_cy = cy
        self.z = z
        self.x = None
        self.y = None
        self.roll = None
        self.pitch = None
        self.yaw = None
        self.camera_matrix = None
    def updata_camera_matrix(self, x, y, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.camera_matrix = get_camera_matrix(self.focal_length, self.image_cx, self.image_cy,
                                               self.roll, self.pitch, self.yaw, self.x, self.y, self.z)
    def get_BEV(self, x_coord, y_coord):
        return image_to_real_world(self.camera_matrix, np.array([x_coord, y_coord]), self.image_height, self.
                                   image_width)
    def on_image(self, x, y, z):
        x_img, y_img = world_to_camera(self.camera_matrix, np.array([[x, y, z]]).T)
        if x_img >= self.image_width / 2 or x_img < -self.image_width / 2 or y_img > 0 or y_img < -self.image_height / 2:
            return False
        return True
    def get_visibility_map(self, xl, xh, yl, yh):
        map = np.zeros((xh - xl, yh - yl))
        for i in range(xl, xh):
            for j in range(yl, yh):
                if self.on_image(i, j, 0) == False:
                    map[i - xl, j - yl] = -2
        return map

def to_homogeneous(heter_coord: np.array) -> np.array:
    homo_coord = np.vstack((heter_coord, np.ones((1, heter_coord.shape[1]))))
    return homo_coord

def to_heterogeneous(homo_coord: np.array) -> np.array:
    heter_coord = homo_coord[:-1] / homo_coord[-1]
    return heter_coord

def get_intrinsic_matrix(f: float, cx: float, cy: float) -> np.array:
    intrinsic_matrix = np.zeros((3, 3))
    intrinsic_matrix[0, 0] = f
    intrinsic_matrix[1, 1] = f
    intrinsic_matrix[0, 2] = cx
    intrinsic_matrix[1, 2] = cy
    intrinsic_matrix[2, 2] = 1
    return intrinsic_matrix

def get_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.array:
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    R_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])
    R_y = np.array([
        [cos_pitch, 0, sin_pitch],
        [0, 1, 0],
        [-sin_pitch, 0, cos_pitch]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, cos_roll, -sin_roll],
        [0, sin_roll, cos_roll]
    ])
    R = R_z @ (R_y @ R_x)
    return R

def get_translation_matrix(R: np.array, x: float, y: float, z: float) -> np.array:
    return -R @ np.array([[x], [y], [z]])

def get_extrinsic_matrix(R: np.array, T: np.array) -> np.array:
    return np.hstack((R, T))

def get_camera_matrix(f, cx, cy, roll, pitch, yaw, x, y, z):
    in_matrix = get_intrinsic_matrix(f, cx, cy)
    R = get_rotation_matrix(roll, pitch, yaw)
    ex_matrix = get_extrinsic_matrix(R, get_translation_matrix(R, x, y, z))
    return in_matrix @ ex_matrix

def world_to_camera(camera_matrix: np.array, real_world_coord: np.array) -> np.array:
    homo_coord = to_homogeneous(real_world_coord)
    camera_coord = camera_matrix @ homo_coord
    return to_heterogeneous(camera_coord)

def camera_to_world_x(camera_matrix: np.array, camera_coord: np.array, x = 0) -> np.array:
    cx, cy = camera_coord
    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = camera_matrix.flatten()
    A = np.array([[p12, p13, -cx], [p22, p23, -cy], [p32, p33, -1]])
    b = np.array([-x * p11 - p14, -x * p21 - p24, -x * p31 - p34])
    s = np.linalg.solve(A, b)
    return np.array([x, s[0], s[1]])

def camera_to_world_y(camera_matrix: np.array, camera_coord: np.array, y = 0) -> np.array:
    cx, cy = camera_coord
    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = camera_matrix.flatten()
    A = np.array([[p11, p13, -cx], [p21, p23, -cy], [p31, p33, -1]])
    b = np.array([-y * p12 - p14, -y * p22 - p24, -y * p32 - p34])
    s = np.linalg.solve(A, b)
    return np.array([s[0], y, s[1]])

def camera_to_world_z(camera_matrix: np.array, camera_coord: np.array, z = 0) -> np.array:
    cx, cy = camera_coord
    p11, p12, p13, p14, p21, p22, p23, p24, p31, p32, p33, p34 = camera_matrix.flatten()
    A = np.array([[p11, p12, -cx], [p21, p22, -cy], [p31, p32, -1]])
    b = np.array([-z * p13 - p14, -z * p23 - p24, -z * p33 - p34])
    s = np.linalg.solve(A, b)
    return np.array([s[0], s[1], z])

def image_to_real_world(camera_matrix: np.array, image_coord: np.array, img_H: int, img_W: int) -> np.array:
    return camera_to_world_z(camera_matrix, image_coord - np.array([img_W / 2, img_H / 2]))

def test_equal(p1, p2, eps):
    result = np.absolute(p1 - p2) < eps
    return np.all(result)

def get_homography(src, dst, img_H, img_W):
    s11, s12, s21, s22, s31, s32, s41, s42 = src.flatten()
    d11, d12, d21, d22, d31, d32, d41, d42 = dst.flatten()
    cb = np.float32([[s11, s12], [s21, s22], [s31, s32], [s41, s42]]) + np.float32([img_W / 2, img_H / 2])
    rb = np.float32([[d11, d12], [d21, d22], [d31, d32], [d41, d42]]) + np.float32([img_W / 2, img_H / 2])
    return cv2.getPerspectiveTransform(cb, rb)

def get_BEV_of_simple_camera(x_coord, y_coord, img_W, img_H):
    cx, cy = 0, 0
    f = 175
    x, y, z = 0, 0, 0.6
    yaw, pitch, roll = math.radians(180), math.radians(0), math.radians(-90)
    camera_matrix = get_camera_matrix(f, cx, cy, roll, pitch, yaw, x, y, z)
    return image_to_real_world(camera_matrix, np.array([x_coord, y_coord]), img_H, img_W)

def get_BEV_of_complex_camera(x_coord, y_coord, x, y, yaw, pitch, roll):
    f = 150
    img_H, img_W = 240, 360
    cx, cy = 0, 0
    z = 0.7
    camera_matrix = get_camera_matrix(f, cx, cy, roll, pitch, yaw, x, y, z)
    return image_to_real_world(camera_matrix, np.array([x_coord, y_coord]), img_H, img_W)

def non_max_suppression(x, y, score, threshold):
    x = np.array(x)
    y = np.array(y)
    exclude_candidates = set()
    for i in range(len(score)):
        if i in exclude_candidates:
            continue
        for j in range(i + 1, len(score)):
            if (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 < threshold:
                exclude_candidates.add(j)
    exclude_list = list(exclude_candidates)
    all_indices = np.arange(len(score))
    mask = ~np.isin(all_indices, exclude_list)
    new_x = x[mask].tolist()
    new_y = y[mask].tolist()
    return new_x, new_y

def apply_offset(obj_loc: tuple, me_loc: tuple, offset = 0.2) -> tuple:
    obj_x, obj_y = obj_loc
    me_x, me_y = me_loc
    diff_x = obj_x - me_x
    diff_y = obj_y - me_y
    diff_x_norm = diff_x / (abs(diff_x) + abs(diff_y))
    diff_y_norm = diff_y / (abs(diff_x) + abs(diff_y))
    new_x = obj_x + diff_x_norm * offset
    new_y = obj_y + diff_y_norm * offset
    return (new_x, new_y)

def add_walls(occupancy_map, ground_truth):
    row, col = occupancy_map.shape
    for r in range(row):
        for c in range(col):
            if ground_truth[r, c] == 1:
                occupancy_map[r, c] = 1

def create_BEV_image(x_box, y_box, x_robot, y_robot, x_me = None, y_me = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.scatter(x_box, y_box, color='yellow', label='box')
    plt.scatter(x_robot, y_robot, color='red', label='robot')
    if x_me is not None and y_me is not None:
        plt.scatter(x_me, y_me, color='green', label='me')
    plt.legend()
    plt.title('BEV')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().invert_yaxis()
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.int8)
    buf.close()
    plt.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def create_visibility_image(visibility_map):
    logging.getLogger('PIL').setLevel(logging.WARNING)
    flipped_map = np.flip(visibility_map)
    plt.figure()
    plt.imshow(flipped_map, interpolation='none')

    plt.tight_layout(pad=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.int8)
    buf.close()
    plt.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def IOU(pred: np, truth: np):
    total = 0
    correct = 0
    wrong = 0
    row, col = pred.shape
    for r in range(row):
        for c in range(col):
            if pred[r, c] != -2:
                if pred[r, c] == 2 and truth[r, c] == 2:
                    total += 1
                    correct += 1
                elif pred[r, c] == 2 and truth[r, c] != 2:
                    wrong += 1
                elif pred[r, c] != 2 and truth[r, c] == 2:
                    total += 1
    print("total visible boxes:", total, "\tcorrectly labeled boxes:", correct, "\tincorrecly labeled boxes:", wrong)

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 
   
def onSegment(p, q, r): 
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
        return True
    return False
  
def orientation(p, q, r):  
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
    if (val > 0): 
        return 1
    elif (val < 0): 
        return 2
    else: 
        return 0
  
def doIntersect(p1,q1,p2,q2): 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
    if ((o1 != o2) and (o3 != o4)): 
        return True
    if ((o1 == 0) and onSegment(p1, p2, q1)): 
        return True
    if ((o2 == 0) and onSegment(p1, q2, q1)): 
        return True
    if ((o3 == 0) and onSegment(p2, p1, q2)): 
        return True
    if ((o4 == 0) and onSegment(p2, q1, q2)): 
        return True
    return False

def visionBlocked(occupancy_map):
    walls = [(Point(0, 6), Point(4, 6)), 
             (Point(0, 14), Point(4, 14)), 
             (Point(6, 0), Point(6, 4)),
             (Point(6, 16), Point(6, 19)),
             (Point(14, 0), Point(14, 4)),
             (Point(14, 16), Point(14, 19)),
             (Point(16, 6), Point(19, 6)),
             (Point(16, 14), Point(19, 14))]
    me = None
    row, col = occupancy_map.shape
    for r in range(row):
        for c in range(col):
            if occupancy_map[r, c] == 5:
                me = Point(r, c)

    for r in range(row):
        for c in range(col):
            if occupancy_map[r, c] == 0:
                for wall in walls:
                    if doIntersect(wall[0], wall[1], me, Point(r, c)):
                        occupancy_map[r, c] = -2

