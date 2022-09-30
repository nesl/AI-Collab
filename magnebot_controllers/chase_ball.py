from enum import Enum
import numpy as np
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.robot import Robot
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.ui import UI
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH
from magnebot import Magnebot, Arm, ActionStatus, ImageFrequency
from magnebot.util import get_default_post_processing_commands
from tdw.add_ons.first_person_avatar import FirstPersonAvatar
import matplotlib.pyplot as plt
import pdb
from tdw.output_data import OutputData, Images, ScreenPosition, Transforms
import cv2
from tdw.add_ons.keyboard import Keyboard
from tdw.add_ons.embodied_avatar import EmbodiedAvatar
from tdw.add_ons.avatar_body import AvatarBody
from tdw.output_data import Keyboard as KBoard
import pyvirtualcam
from magnebot import ArmJoint
import time
import cupy as cp
from tdw.quaternion_utils import QuaternionUtils

width = 640 #640 #256
height = 480 #480 #256

cam0 = pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video0')
cam1 = pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video1')
cam2 = pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video2')


class State(Enum):
    initializing = 1
    swinging = 2
    moving_to_ball = 3
    grasping = 4
    resetting_success = 5
    resetting_failure = 6
    moving_to_robot = 7
    dropping = 8
    backing_away = 9
    backing_away_from_wall = 10
    backing_away_from_wall_with_ball = 12


class Enchanced_Magnebot(Magnebot):

    def __init__(self,robot_id, position, controlled_by, key_set=None,image_frequency=ImageFrequency.never,pass_masks=['_img'],strength=1):
        super().__init__(robot_id=robot_id, position=position,image_frequency=image_frequency,pass_masks=pass_masks)
        self.key_set = key_set
        self.ui = []
        self.ui_elements = {}
        self.strength = strength
        self.danger_estimates = []
        self.company = []
        self.controlled_by = controlled_by

"""
class Enhanced_Object():
    self.required_strength = 0
    self.danger_level = 0
    self.id = 0
"""        


class ChaseBall(Controller):
    """
    Add a UR5 robot, a Magnebot and a ball.
    The robot will swing at the ball. The Magnebot will chase the ball and return it to the robot.

    This is a "promo" controller rather than an "example" controller for several reasons:

    - It uses a very state machine to manage behavior that is probably too simple for actual use-cases.
    - It includes per-frame image capture which is very slow.
    """
    change_position_called = False

    def change_position(self):
        self.change_position_called = True

    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True):
        super().__init__(port=port, check_version=check_version, launch_build=launch_build)

        self.user_magnebots = []
        self.ai_magnebots = []
        self.graspable_objects = []
        self.keys_set = []
        self.uis = []
        self.timer = 1000.0
        self.terminate = False
        #self.ui_elements = []
        #self.strength = {}
        #self.grasped_object = []

        # Add the robot, the Magnebot, and an object manager.
        #self.robot: Robot = Robot(robot_id=self.get_unique_id(), name="ur5", position={"x": -1.4, "y": 0, "z": 2.6})#{"x": 1.88, "y": 0, "z": 0.37})#{"x": -1.4, "y": 0, "z": 2.6})
        self.magnebot: Magnebot = Enchanced_Magnebot(robot_id=self.get_unique_id(), position={"x": -1.4, "y": 0, "z": -1.1},#position={"x": -1.97, "y": 0, "z": 3.11},  #{"x": -1.4, "y": 0, "z": -1.1},
                                           image_frequency=ImageFrequency.never, controlled_by='ai')
                                           
        self.ai_magnebots.append(self.magnebot)

        self.ai_magnebots.append(Enchanced_Magnebot(robot_id=self.get_unique_id(), position={"x": 0, "y": 0, "z": -1.1},
                                           image_frequency=ImageFrequency.never, controlled_by='ai'))

        self.ai_magnebots.append(Enchanced_Magnebot(robot_id=self.get_unique_id(), position={"x": 0, "y": 0, "z": -2.1},
                                           image_frequency=ImageFrequency.never, controlled_by='ai'))
        #self.strength[self.magnebot.robot_id] = 1

        self.user_magnebots.append(Enchanced_Magnebot(robot_id=self.get_unique_id(), position={"x": 0, "y": 0, "z": 1.1}, #{"x": 2, "y": 0, "z": 2},
                                           image_frequency=ImageFrequency.always, pass_masks=['_img'],key_set=["UpArrow","DownArrow","RightArrow","LeftArrow","Z","X","C","V","B"], controlled_by='human'))
                                           
        #self.strength[self.user_magnebots[0].robot_id] = 1

        #self.keys_set = [["UpArrow"],["DownArrow"],["RightArrow"],["LeftArrow"],["Z"],["X"],["C"],["V"],["B"]]

        
        self.user_magnebots.append(Enchanced_Magnebot(robot_id=self.get_unique_id(), position={"x": 0, "y": 0, "z": 2.1},
                                           image_frequency=ImageFrequency.always, pass_masks=['_img'], key_set=["W","S","D","A","H","J","K","L","G"], controlled_by='human'))

        #self.user_magnebots.append(Enchanced_Magnebot(robot_id=self.get_unique_id(), position={"x": 4, "y": 0, "z": 1.6},
         #                                  image_frequency=ImageFrequency.always, pass_masks=['_img'], key_set=["Alpha5","R","E","Y","U","I","O","P","Alpha0"], controlled_by='human'))
                                           
        #self.keys_set = [[*self.keys_set[0],"W"],[*self.keys_set[1],"S"],[*self.keys_set[2],"D"],[*self.keys_set[3],"A"], [*self.keys_set[4],"H"],[*self.keys_set[5],"J"],[*self.keys_set[6],"K"],[*self.keys_set[7],"L"]]
        
        '''
        self.user_magnebots.append(Magnebot(robot_id=self.get_unique_id(), position={"x": 0, "y": 0, "z": 2},
                                           image_frequency=ImageFrequency.always, pass_masks=['_img']))
                                           
        self.keys_set.append(["T","G","H","F"])
        self.user_magnebots.append(Magnebot(robot_id=self.get_unique_id(), position={"x": 3, "y": 0, "z": 1},
                                           image_frequency=ImageFrequency.always, pass_masks=['_img']))  
        self.keys_set.append(["I","K","L","J"])  
                                           
        '''
        
        
        image = "white.png"
        # Set the dimensions of the progress bar.
        self.progress_bar_position = {"x": 16, "y": -16}
        self.progress_bar_size = {"x": 16, "y": 16}
        self.progress_bar_scale = {"x": 10, "y": 2}
        self.progress_bar_anchor = {"x": 0, "y": 1}
        self.progress_bar_pivot = {"x": 0, "y": 1}
            
        for um_idx,um in enumerate(self.user_magnebots):
            um.collision_detection.objects = False
            um.collision_detection.walls = False
            ui = UI(canvas_id=um_idx)
            ui.attach_canvas_to_avatar(avatar_id=str(um.robot_id))
            
            if um_idx == 0:
                self.keys_set = [[um.key_set[0]],[um.key_set[1]],[um.key_set[2]],[um.key_set[3]],[um.key_set[4]],[um.key_set[5]],[um.key_set[6]],[um.key_set[7]],[um.key_set[8]]]
            else:
                for kidx in range(len(self.keys_set)):
                    self.keys_set[kidx].append(um.key_set[kidx])

            # Add the background sprite.
            ui.add_image(image=image,
                                 position=self.progress_bar_position,
                                 size=self.progress_bar_size,
                                 anchor=self.progress_bar_anchor,
                                 pivot=self.progress_bar_pivot,
                                 color={"r": 0, "g": 0, "b": 0, "a": 1},
                                 scale_factor=self.progress_bar_scale,
                                 rgba=False)
            
            bar_id = ui.add_image(image=image,
                                  position=self.progress_bar_position,
                                  size=self.progress_bar_size,
                                  anchor=self.progress_bar_anchor,
                                  pivot=self.progress_bar_pivot,
                                  color={"r": 1, "g": 0, "b": 0, "a": 1},
                                  scale_factor={"x": 0, "y": self.progress_bar_scale["y"]},
                                  rgba=False)
            # Add some text.
            text_id = ui.add_text(text="Strength: 1",
                                  position=self.progress_bar_position,
                                  anchor=self.progress_bar_anchor,
                                  pivot=self.progress_bar_pivot,
                                  font_size=18)
            # Add some text.
            mins, remainder = divmod(self.timer, 60)
            secs,millisecs = divmod(remainder,1)

            #Add timer
            timer_text_id = ui.add_text(text='{:02d}:{:02d}'.format(int(mins), int(secs)),
                                  position= {"x": -60, "y": -30},
                                  anchor = {"x": 1, "y": 1},
                                  font_size=35,
                                  color={"r": 0, "g": 0, "b": 1, "a": 1})
            
            self.uis.append(ui)
            um.ui = ui
            um.ui_elements = ((bar_id,text_id,timer_text_id))
            #self.ui_elements[um.robot_id] = ((bar_id,text_id))
            #self.grasped_object.append("")

        self.object_manager: ObjectManager = ObjectManager()
        # Add a ball.
        self.ball_id: int = self.get_unique_id()
        self.ball_id2: int = self.get_unique_id()
        self.box: int = self.get_unique_id()
        self.rug: int = self.get_unique_id()
        self.graspable_objects.extend([self.ball_id,self.ball_id2,self.box])
        self.required_strength = {self.ball_id:1,self.ball_id2:1,self.box:2}
        self.danger_level = {self.ball_id:1,self.ball_id2:1,self.box:2}
        self.dangerous_objects = []
        self.dangerous_objects.append(self.box)
        # Add a camera and enable image capture.
        #self.camera: ThirdPersonCamera = ThirdPersonCamera(position={"x": 0, "y": 10, "z": -1},
        #                                                   look_at={"x": 0, "y": 0, "z": 0},
        #                                                   avatar_id="a")
        #images_path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("chase_ball")
        #print(f"Images will be saved to: {images_path}")
        
        '''
        self.embodied_avatar = EmbodiedAvatar(avatar_id="b",
                                 body=AvatarBody.capsule,
                                 position={"x": 2, "y": 1, "z": 2},
                                 rotation={"x": 0, "y": 30, "z": 0},
                                 color={"r": 0.6, "g": 0.3, "b": 0, "a": 1})
        
        
        self.embodied_avatar2 = EmbodiedAvatar(avatar_id="c",
                                 body=AvatarBody.capsule,
                                 position={"x": 3, "y": 1, "z": 2},
                                 rotation={"x": 0, "y": 30, "z": 0},
                                 color={"r": 0.6, "g": 0.3, "b": 0.5, "a": 1})
        '''
        image_capture = ImageCapture(avatar_ids=["a"], path="")
        image_capture.set(avatar_ids=["a"],save=False)
        #self.keyboard = Keyboard()
        #self.keyboard.listen(key="UpArrow", function=self.change_position, events = ["press", "hold"])
        self.add_ons.extend([*self.ai_magnebots,  *self.user_magnebots, self.object_manager, *self.uis]) #, self.keyboard])#, image_capture]) #image_capture, self.person])
        #self.add_ons.extend([self.object_manager, self.camera, image_capture])

        # Create the scene.

        commands = [#{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                    TDWUtils.create_empty_room(20, 20),
                    self.get_add_material("parquet_long_horizontal_clean",
                                          library="materials_high.json"),
                    {"$type": "set_screen_size",
                     "width": width, #640,
                     "height": height}, #480},
                    {"$type": "rotate_directional_light_by",
                     "angle": 30,
                     "axis": "pitch"},
                    {"$type": "create_interior_walls", "walls": [{"x": 6, "y": 1}, {"x": 6, "y": 2},{"x": 6, "y": 3},{"x": 6, "y": 4},{"x": 6, "y": 5},{"x": 1, "y": 6},{"x": 2, "y": 6},{"x": 3, "y": 6},{"x": 4, "y": 6},{"x": 5, "y": 6}]},
                    {"$type": "create_interior_walls", "walls": [{"x": 14, "y": 1}, {"x": 14, "y": 2},{"x": 14, "y": 3},{"x": 14, "y": 4},{"x": 14, "y": 5},{"x": 19, "y": 6},{"x": 18, "y": 6},{"x": 17, "y": 6},{"x": 16, "y": 6},{"x": 15, "y": 6}]},   
                    {"$type": "create_interior_walls", "walls": [{"x": 6, "y": 19}, {"x": 6, "y": 18},{"x": 6, "y": 17},{"x": 6, "y": 16},{"x": 6, "y": 15},{"x": 1, "y": 14},{"x": 2, "y": 14},{"x": 3, "y": 14},{"x": 4, "y": 14},{"x": 5, "y": 14}]},
                    {"$type": "create_interior_walls", "walls": [{"x": 14, "y": 19}, {"x": 14, "y": 18},{"x": 14, "y": 17},{"x": 14, "y": 16},{"x": 14, "y": 15},{"x": 19, "y": 14},{"x": 18, "y": 14},{"x": 17, "y": 14},{"x": 16, "y": 14},{"x": 15, "y": 14}]}]
        
        #self.communicate(self.get_add_scene(scene_name="tdw_room"))
        #commands = []


        #Instantiate objects
        max_coord = 8
        object_models = ['iron_box','4ft_shelf_metal','trunck','lg_table_marble_green','b04_backpack','36_in_wall_cabinet_wood_beach_honey']
        coords = {}
        
        coords[object_models[0]] = [[max_coord,max_coord],[max_coord-1,max_coord-0.1],[max_coord-0.5,max_coord-0.2],[max_coord-0.4,max_coord],[max_coord,max_coord-0.5]]
        coords[object_models[1]] = [[max_coord-3,max_coord]]

        coords[object_models[2]] = [[max_coord,max_coord-3]]
        coords[object_models[3]] = [[max_coord-2,max_coord-2]]
        coords[object_models[4]] = [[max_coord-1,max_coord-2]]
        coords[object_models[5]] = [[max_coord-3,max_coord-3]]

        modifications = [[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]]

        final_coords = {}

        for objm in object_models:
            final_coords[objm] = []
        

        for fc in final_coords.keys():
            for m in modifications:
                final_coords[fc].extend(np.array(coords[fc])*m)

        for fc in final_coords.keys():
            for c in final_coords[fc]:
                object_id = self.get_unique_id()
                self.graspable_objects.append(object_id)
                self.required_strength[object_id] = np.random.choice([1,2,3],1)[0]
                self.danger_level[object_id] = np.random.choice([1,2],1,p=[0.9,0.1])[0]
                commands.extend(self.get_add_physics_object(model_name=fc,
                                                 object_id=object_id,
                                                 position={"x": c[0], "y": 0, "z": c[1]},
                                                 rotation={"x": 0, "y": 0, "z": 0},
                                                 default_physics_values=False,
                                                 mass=10,
                                                 scale_mass=False))
                if self.danger_level[object_id] == 2:
                    self.dangerous_objects.append(object_id)




        # Add post-processing.
        commands.extend(get_default_post_processing_commands())
        commands.extend(self.get_add_physics_object(model_name="prim_sphere",
                                                    library="models_special.json",
                                                    object_id=self.ball_id,
                                                    position={"x": -0.871, "y": 0.1, "z": 3.189},#{"x": 0.1, "y": 0.1, "z": -2.15}, #{"x": -0.871, "y": 0.1, "z": 3.189},
                                                    scale_factor={"x": 0.2, "y": 0.2, "z": 0.2},
                                                    default_physics_values=False,
                                                    scale_mass=False,
                                                    dynamic_friction=0.1,
                                                    static_friction=0.1,
                                                    bounciness=0.7,
                                                    mass=10))
        commands.extend(self.get_add_physics_object(model_name="prim_sphere",
                                                    library="models_special.json",
                                                    object_id=self.ball_id2,
                                                    position={"x": -0.871, "y": 0.1, "z": 3.189},#{"x": 0.1, "y": 0.1, "z": -2.15},
                                                    scale_factor={"x": 0.2, "y": 0.2, "z": 0.2},
                                                    default_physics_values=False,
                                                    scale_mass=False,
                                                    dynamic_friction=0.1,
                                                    static_friction=0.1,
                                                    bounciness=0.7,
                                                    mass=10))
        
        commands.extend(self.get_add_physics_object(model_name="iron_box",
                                         object_id=self.box,
                                         position={"x": 0, "y": 0, "z": 0},
                                         rotation={"x": 0, "y": 0, "z": 0}))

        commands.extend(self.get_add_physics_object(model_name="carpet_rug",
                                         object_id=self.rug,
                                         position={"x": 0, "y": 0, "z": 0},
                                         rotation={"x": 0, "y": 0, "z": 0}))
        
        commands.extend(TDWUtils.create_avatar(position={"x": -3.15, "y": 10, "z": 0.22},#{"x": 0, "y": 10, "z": -1},
                                                           look_at={"x": 0, "y": 0, "z": 0},
                                                           avatar_id="a"))
        commands.extend([{"$type": "set_pass_masks","pass_masks": ["_img"],"avatar_id": "a"},
                  {"$type": "send_images","frequency": "always","ids": ["a"]},
                  {"$type": "set_img_pass_encoding", "value": False},
                  {"$type": "set_render_order", "render_order": 1, "sensor_name": "SensorContainer", "avatar_id": "a"},
                  {"$type": "send_keyboard", "frequency": "always"}])

        
        self.communicate(commands)
        self.state = State.moving_to_ball
        self._frame: int = 0


    def add_ui(self,original_image, id_image):
    
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 0.5         
        # Blue color in BGR
        colorFont = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 2
        
        for um_idx,um in enumerate(self.user_magnebots):
            color = cp.asarray(um.static.joints[um.static.arm_joints[ArmJoint.torso]].segmentation_color)
            tmp_mask = cp.where((id_image == color).all(axis=2))
            if tmp_mask[0].size < 10:
                continue

            cv2.putText(original_image, 'U_'+str(um.robot_id), (int(tmp_mask[1][0].get()),int(tmp_mask[0][0].get())), font, fontScale, colorFont, thickness, cv2.LINE_AA)
                                    
        for um_idx,um in enumerate(self.ai_magnebots):
            color = cp.asarray(um.static.joints[um.static.arm_joints[ArmJoint.torso]].segmentation_color)
            tmp_mask = cp.where((id_image == color).all(axis=2))
            if tmp_mask[0].size < 10:
                continue

            cv2.putText(original_image, 'A_'+str(um.robot_id), (int(tmp_mask[1][0].get()),int(tmp_mask[0][0].get())), font, fontScale, colorFont, thickness, cv2.LINE_AA)



    def add_ui2(self, original_image, screen_positions):
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 0.5         
        # Blue color in BGR
        colorFont = screen_positions['color']
        # Line thickness of 2 px
        thickness = 2
        
        for s_idx,s in enumerate(screen_positions['coords']):
            try:
                cv2.putText(original_image, screen_positions['ids'][s_idx], (int(s[0]),int(s[1])), font, fontScale, colorFont[s_idx], thickness, cv2.LINE_AA)
            except:
                pdb.set_trace()
    
    def run(self):
        done = False
        commands = []
        counter = 0
        key = ""
        messages = []
        once = 0
        extra_commands = []
        duration = []
        
        user_magnebots_ids = [str(um.robot_id) for um in self.user_magnebots]
        ai_magnebots_ids = [str(um.robot_id) for um in self.ai_magnebots]
        keys_time_unheld = [0]*len(user_magnebots_ids)
        all_ids = [*user_magnebots_ids,*ai_magnebots_ids]
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
        
        while not done:
            start_time = time.time()
            
            user_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.user_magnebots]
            ai_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.ai_magnebots]
            
            commands.append({"$type": "send_screen_positions", "position_ids": list(range(0,len(all_ids))), "positions": [*user_magnebots_positions,*ai_magnebots_positions], "ids": ["a",*user_magnebots_ids], "frequency": "once"})
            commands_time = time.time()
            
            to_eliminate = []
            for ex_idx in range(len(extra_commands)):
                duration[ex_idx] -= 1
                if not duration[ex_idx]:
                    to_eliminate.append(ex_idx)
                commands.append(extra_commands[ex_idx])
            
            for e in to_eliminate:
                del duration[e]
                del extra_commands[e]

            resp = self.communicate(commands)

            commands.clear()
            #print('commands time', time.time()-commands_time)

            for idx in range(len(all_magnebots)):
                robot_id = all_magnebots[idx].robot_id
                all_magnebots[idx].strength = 1
                all_magnebots[idx].company = []
                for idx2 in range(len(all_magnebots)):
                    if idx == idx2:
                        continue
                    if np.linalg.norm(all_magnebots[idx].dynamic.transform.position - all_magnebots[idx2].dynamic.transform.position) < 2: #Check only two dimensions not three
                        all_magnebots[idx].strength += 1
                        all_magnebots[idx].company.append(all_magnebots[idx2].controlled_by)
                            
                if all_magnebots[idx].ui_elements:
                    #We assume self.uis and all_magnebots have the same sequence
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[1],text=f"Strength: {all_magnebots[idx].strength}")
                    all_magnebots[idx].ui.set_size(ui_id=all_magnebots[idx].ui_elements[0], size={"x": int(self.progress_bar_size["x"] * self.progress_bar_scale["x"] * (all_magnebots[idx].strength-1)/10),    "y": int(self.progress_bar_size["y"] * self.progress_bar_scale["y"])})

                    mins, remainder = divmod(self.timer, 60)
                    secs,millisecs = divmod(remainder,1)
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[2],text='{:02d}:{:02d}'.format(int(mins), int(secs)))
                for arm in [Arm.right,Arm.left]:
                    if all_magnebots[idx].dynamic.held[arm].size > 0:
                        #Drop object if strength decreases
                        if self.required_strength[all_magnebots[idx].dynamic.held[arm][0]] > all_magnebots[idx].strength:
                            all_magnebots[idx].drop(target=all_magnebots[idx].dynamic.held[arm][0], arm=arm)
                        #Terminate game if dangerous object held alone
                        if all_magnebots[idx].dynamic.held[arm][0] in self.dangerous_objects:
                            if (all_magnebots[idx].controlled_by == 'ai' and 'human' not in all_magnebots[idx].company) or (all_magnebots[idx].controlled_by == 'human' and 'ai' not in all_magnebots[idx].company):
                                for um in self.user_magnebots:
                                    txt = um.ui.add_text(text="Dangerous object picked without help!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 0, "g": 0, "b": 1, "a": 1},
                                     font_size=20
                                     )
                                    messages.append([idx,txt,0])
                                self.terminate = True
                            
                 
                        
                

            '''
            if self.person.right_button_pressed:
                done = True
            if self.person.left_button_pressed:
                if np.linalg.norm(self.object_manager.transforms[self.ball_id2].position - self.person.transform.position) < 3:
                    commands.extend([{"$type": "object_look_at_position","position": TDWUtils.array_to_vector3(self.person.transform.position),"id": self.ball_id2},
{"$type": "apply_force_magnitude_to_object", "magnitude": 1,"id": self.ball_id2}])
                if np.linalg.norm(self.object_mSegmentationColorsanager.transforms[self.ball_id].position - self.person.transform.position) < 3:
                    commands.extend([{"$type": "object_look_at_position","position": TDWUtils.array_to_vector3(self.person.transform.position),"id": self.ball_id},
{"$type": "apply_force_magnitude_to_object", "magnitude": 1,"id": self.ball_id}])
            '''
                
            if self.state == State.moving_to_ball:
                # Collided with a wall. Back away.
                if self.magnebot.action.status == ActionStatus.collision:
                    self.state = State.backing_away_from_wall
                    self.magnebot.move_by(-2)
                else:
                    self._frame += 1
                    # Every so often, course-correct the Magnebot.
                    if self._frame % 15 == 0:
                        # If the Magnebot is near the ball, try to pick it up.
                        if np.linalg.norm(self.object_manager.transforms[self.ball_id].position -
                                          self.magnebot.dynamic.transform.position) < 0.9:
                            self.state = State.grasping
                            self.magnebot.grasp(target=self.ball_id, arm=Arm.right)
                        # Course-correct.
                        else:
                            self.magnebot.move_to(target=self.ball_id, arrived_offset=0.1)
            elif self.state == State.backing_away_from_wall:
                # Finished backing away from the wall. Resume the chase.
                if self.magnebot.action.status != ActionStatus.ongoing:
                    self.magnebot.move_to(target=self.ball_id, arrived_offset=0.3)
                    self._frame = 0
                    self.state = State.moving_to_ball
            elif self.state == State.grasping:
                # Caught the object! Reset the arm.
                if self.magnebot.action.status == ActionStatus.success:
                    self.state = State.resetting_success
                    self.magnebot.reset_arm(arm=Arm.right)
                # Failed to grasp the object. Reset the arm.
                elif self.magnebot.action.status != ActionStatus.ongoing:
                    self.state = State.resetting_failure
                    self.magnebot.reset_arm(arm=Arm.right)
            elif self.state == State.resetting_failure:
                # Try moving to the object again.
                if self.magnebot.action.status != ActionStatus.ongoing:
                    self.state = State.moving_to_ball
                    self.magnebot.move_to(target=self.ball_id, arrived_offset=0.1)
                    self._frame = 0
            elif self.state == State.resetting_success:
                # The arm has been reset. Back away from the wall.
                if self.magnebot.action.status != ActionStatus.ongoing:
                    self.state = State.backing_away_from_wall_with_ball
                    self.magnebot.move_by(-3)
            elif self.state == State.backing_away_from_wall_with_ball:
                # The Magnebot has backed away from the wall. Move towards the robot.
                if self.magnebot.action.status != ActionStatus.ongoing:
                    self.state = State.moving_to_robot
                    self.magnebot.collision_detection.objects = False
                    self.magnebot.collision_detection.walls = False
                    self.magnebot.move_to(target={"x": -0.871, "y": 0, "z": 3}, arrived_offset=0.3)
                    #self.magnebot.move_to(target=self.box, arrived_offset=0.1)
            elif self.state == State.moving_to_robot:

                if np.linalg.norm([-0.871,0,3] -
                                          self.magnebot.dynamic.transform.position) > 0.9:
                    if self.magnebot.action.status == ActionStatus.tipping or counter % 15 == 0:
                        self.magnebot.move_to(target={"x": -0.871, "y": 0, "z": 3}, arrived_offset=0.3)
                        #self.magnebot.move_to(target=self.box, arrived_offset=0.1)
                    #print('hello', counter)
                    print(self.magnebot.action.status)
                    counter += 1
                else:
                    # The Magnebot has arrived at the robot. Drop the object.
                    if self.magnebot.action.status != ActionStatus.ongoing:
                        self.state = State.dropping
                        self.magnebot.collision_detection.objects = False
                        self.magnebot.collision_detection.walls = False
                        self.magnebot.drop(target=self.ball_id, arm=Arm.right)
            elif self.state == State.dropping:
                # The Magnebot has dropped the object. Move away from the robot.
                if self.magnebot.action.status != ActionStatus.ongoing:
                    self.state = State.backing_away
                    self.magnebot.move_by(-4)
                    # Swing again.
            #elif self.state == State.backing_away:
                # The Magnebot has moved away from the robot. We're done!
                #if np.linalg.norm([-0.871,0,3] - self.magnebot.dynamic.transform.position) > 2 and np.linalg.norm([-0.871,0,3]-self.object_manager.transforms[self.ball_id2].position) < 1:
                    #self.robot.set_joint_targets(targets={self.robot.static.joint_ids_by_name["shoulder_link"]: -70})

                    #if self.magnebot.action.status != ActionStatus.ongoing and not self.robot.joints_are_moving():
                    #    done = True
            #print(self.state)
            
            all_images = []

            output_loop = time.time()

            screen_data = {}
            magnebot_images = {}
 
            for i in range(len(resp) - 1):
                
                r_id = OutputData.get_data_type_id(resp[i])
                # Get Images output data.
                if r_id == "imag":
                    images = Images(resp[i])
                    # Determine which avatar captured the image.
                    if images.get_avatar_id() == "a":
                        # Iterate throught each capture pass.
                        for j in range(images.get_num_passes()):
                            # This is the _img pass.

                            if images.get_pass_mask(j) == "_img":
                                #image_arr = images.get_image(j)

                                # Get a PIL image.
                                pil_image = TDWUtils.get_pil_image(images=images, index=j)
                                all_images = np.asarray(pil_image)
                                img_image = np.asarray(pil_image)
                                magnebot_images[images.get_avatar_id()] = np.asarray(pil_image)
                                #cam0.send(img_image)
                                #cv2.imshow('frame',np.asarray(pil_image))
                                #cv2.waitKey(1)
                                
                            elif images.get_pass_mask(j) == "_id":
                                start_time = time.time()

                                pil_image_id = TDWUtils.get_pil_image(images=images, index=j)
                                image_array = cp.asarray(pil_image_id)
                                #image_array_np = np.asarray(pil_image)
                                end_time = time.time()
                                start_time2 = time.time()
                                #mask = cv2.inRange(image_array,color,color)
                                #pdb.set_trace()
                                mask = []
                                '''
                                for um_idx,um in enumerate(self.user_magnebots):
                                    color = cp.asarray(um.static.joints[um.static.arm_joints[ArmJoint.torso]].segmentation_color)
                                    tmp_mask = cp.where((image_array == color).all(axis=2),1,0)
                                    #tmp_mask = np.ma.masked_where((image_array == color).all(axis=2),image_array[:,:,0]).mask
                                    if um_idx == 0:
                                        mask = tmp_mask
                                    else:
                                        mask = cp.logical_or(mask,tmp_mask)
                                '''
                                
                                
                                #self.add_ui(img_image,image_array)
                                

                                end_time2 = time.time()
                                start_time3 = time.time()
                                #np.where((image_array == color).all(axis=2))
                                #masked = cv2.bitwise_and(image_array.get(),image_array.get(), mask=mask.get().astype(np.uint8))
                                cam0.send(img_image)
                                #cv2.imshow('frame',img_image)
                                #cv2.waitKey(1)
                                #print(end_time-start_time, end_time2-start_time2,time.time()-start_time3)
                            
                    
                    elif images.get_avatar_id() in user_magnebots_ids:
                        idx = user_magnebots_ids.index(images.get_avatar_id())
                        img_image = np.asarray(self.user_magnebots[idx].dynamic.get_pil_images()['img'])
                        magnebot_images[images.get_avatar_id()] = img_image
                    '''
                    elif images.get_avatar_id() == str(self.user_magnebots[0].robot_id):
                        img_image = np.asarray(self.user_magnebots[0].dynamic.get_pil_images()['img'])
                        magnebot_images[images.get_avatar_id()] = img_image
                        #id_image = cp.asarray(self.user_magnebots[0].dynamic.get_pil_images()['id'])
                        all_images = np.concatenate((all_images,img_image), axis=1)
                        #self.add_ui(img_image,id_image)

                        cam1.send(img_image)
                    elif images.get_avatar_id() == str(self.user_magnebots[1].robot_id):
                        img_image = np.asarray(self.user_magnebots[1].dynamic.get_pil_images()['img'])
                        magnebot_images[images.get_avatar_id()] = img_image
                        #id_image = cp.asarray(self.user_magnebots[1].dynamic.get_pil_images()['id'])
                        all_images = np.concatenate((all_images,img_image), axis=1)
                        #self.add_ui(img_image,id_image)

                        cam2.send(img_image)
                    '''
                        
    
                elif r_id == "scre":
                    scre = ScreenPosition(resp[i])
                    scre_coords = scre.get_screen()

                    scre_coords = (scre_coords[0],height-scre_coords[1],scre_coords[2])
                    
                    if not (scre_coords[0] < 0 or scre_coords[0] > width or scre_coords[1] < 0 or scre_coords[1] > height or scre_coords[2] < 0):
                    
                        temp_all_ids = all_ids + self.graspable_objects
                        mid = temp_all_ids[scre.get_id()]
                        color = (255, 255, 255)                        


                        if mid in ai_magnebots_ids:
                            mid = 'A_'+mid
                        elif mid in user_magnebots_ids:
                            mid = 'U_'+mid
                        else:
                            danger_estimate = self.user_magnebots[user_magnebots_ids.index(scre.get_avatar_id())].danger_estimates[mid]
                            mid = str(mid)
                            
              
                            if danger_estimate >= 2:
                                color = (0, 0, 255)
                            else:
                                color = (0, 255, 0)

                        if scre.get_avatar_id() not in screen_data:
                            screen_data[scre.get_avatar_id()] = {}
                            screen_data[scre.get_avatar_id()]['coords'] = [scre_coords]
                            screen_data[scre.get_avatar_id()]['ids'] = [mid]
                            screen_data[scre.get_avatar_id()]['color'] = [color]
                        else:
                            screen_data[scre.get_avatar_id()]['coords'].append(scre_coords)
                            screen_data[scre.get_avatar_id()]['ids'].append(mid)
                            screen_data[scre.get_avatar_id()]['color'].append(color)

                    '''
                    if images.get_avatar_id() == "b":
                        # Iterate throught each capture pass.
                        for j in range(images.get_num_passes()):
                            # This is the _img pass.
                            if images.get_pass_mask(j) == "_img":
                                image_arr = images.get_image(j)
                                # Get a PIL image.
                                pil_image = TDWUtils.get_pil_image(images=images, index=j)
                                all_images = np.concatenate((all_images,np.asarray(pil_image)), axis=1)
                                cv2.imshow('frame',all_images)
                                cv2.waitKey(1)
                    '''
                    
         

                elif r_id == "keyb":

                    keys = KBoard(resp[i])
                    # Listen for events where the key was first pressed on the previous frame.
                    for j in range(keys.get_num_pressed()):
                        idx = -1
                        if keys.get_pressed(j) in self.keys_set[0]: #Advance
                            idx = self.keys_set[0].index(keys.get_pressed(j))
                            #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                            
                            self.user_magnebots[idx].move_by(distance=10)
                                

                            keys_time_unheld[idx] = -20

                        elif keys.get_pressed(j) in self.keys_set[1]: #Back
                            idx = self.keys_set[1].index(keys.get_pressed(j))
                            self.user_magnebots[idx].move_by(distance=-10)
                            keys_time_unheld[idx] = -20
                        elif keys.get_pressed(j) in self.keys_set[2]: #Right
                            idx = self.keys_set[2].index(keys.get_pressed(j))
                            self.user_magnebots[idx].turn_by(179)
                            keys_time_unheld[idx] = -20
                        elif keys.get_pressed(j) in self.keys_set[3]: #Left
                            idx = self.keys_set[3].index(keys.get_pressed(j))
                            self.user_magnebots[idx].turn_by(-179)
                            keys_time_unheld[idx] = -20
                        elif keys.get_pressed(j) in self.keys_set[4] or keys.get_pressed(j) in self.keys_set[5]: #Pick up/Drop
                            if keys.get_pressed(j) in self.keys_set[4]:
                                arm = Arm.left
                                key_idx = 4
                            else:
                                arm = Arm.right
                                key_idx = 5
                                
                            idx = self.keys_set[key_idx].index(keys.get_pressed(j))
                            
                            if self.user_magnebots[idx].dynamic.held[arm].size > 0:
                                self.user_magnebots[idx].drop(target=self.user_magnebots[idx].dynamic.held[arm][0], arm=arm)
                            else:
                                angle = QuaternionUtils.quaternion_to_euler_angles(self.user_magnebots[idx].dynamic.transform.rotation)
                                if abs(angle[0]) > 90:
                                    if angle[1] > 0:
                                        angle[1] = 90+(90-angle[1])

                                    else:
                                        angle[1] = abs(angle[1])+180
                                else:
                                    if angle[1] < 0:
                                        angle[1] = 360+angle[1]
                                        
                                grasp_object = ""
                                for o in self.graspable_objects:
                                    if np.linalg.norm(self.object_manager.transforms[o].position -
                                        self.user_magnebots[idx].dynamic.transform.position) < 1:
                                        
                                        vec1 = np.array([np.cos(np.deg2rad(angle[1])),np.sin(np.deg2rad(angle[1]))])
                                        vec2 = self.user_magnebots[idx].dynamic.transform.position - self.object_manager.transforms[o].position
                                        vec2 /= np.linalg.norm(vec2)
                                        vec2 = np.array([vec2[0],vec2[2]])

                                        print(angle, vec1, vec2, np.divide((vec1*vec2).sum(0),1))
                                        if np.divide((vec1*vec2).sum(0),1) > 0:
                                            print("grabable")
                                            grasp_object = o
                                            break
                                if grasp_object:
                                    print("grasping", grasp_object, arm, idx)
                                    if self.user_magnebots[idx].strength < self.required_strength[o]:
                                        txt = self.user_magnebots[idx].ui.add_text(text="Too heavy to carry alone!!",
                                         position={"x": 0, "y": 0},
                                         color={"r": 0, "g": 0, "b": 1, "a": 1},
                                         font_size=20
                                         )
                                        messages.append([idx,txt,0])
                                    else:
                                        self.user_magnebots[idx].grasp(target=grasp_object, arm=arm)
                                        self.user_magnebots[idx].in_danger = True
                                        if o in self.dangerous_objects and 'ai' not in self.user_magnebots[idx].company:
                                            for um in self.user_magnebots:
                                                txt = um.ui.add_text(text="Dangerous object picked without help!",
                                                 position={"x": 0, "y": 0},
                                                 color={"r": 0, "g": 0, "b": 1, "a": 1},
                                                 font_size=20
                                                 )
                                                messages.append([idx,txt,0])
                                            self.terminate = True
                                        
                                    #self.user_magnebots[0].grasp(target=self.box, arm=Arm.right)
                                
                        elif keys.get_pressed(j) in self.keys_set[6]:
                            idx = self.keys_set[6].index(keys.get_pressed(j))
                            self.user_magnebots[idx].rotate_camera(pitch=10)
                        elif keys.get_pressed(j) in self.keys_set[7]:
                            idx = self.keys_set[7].index(keys.get_pressed(j))
                            self.user_magnebots[idx].rotate_camera(pitch=-10)
                        elif keys.get_pressed(j) in self.keys_set[8]:
                            idx = self.keys_set[8].index(keys.get_pressed(j))

                            near_items_pos = []
                            near_items_idx = []
                            danger_estimates = {}
                            possible_danger_levels = [1,2]
                            for o_idx,o in enumerate(self.graspable_objects):
                                if np.linalg.norm(self.object_manager.transforms[o].position -
                                        self.user_magnebots[idx].dynamic.transform.position) < 2:
                                    near_items_idx.append(len(all_ids)+o_idx)
                                    near_items_pos.append(TDWUtils.array_to_vector3(self.object_manager.transforms[o].position))
                                    actual_danger_level = self.danger_level[o]
                                    possible_danger_levels_tmp = possible_danger_levels.copy()
                                    possible_danger_levels_tmp.remove(actual_danger_level)
                                    danger_estimate = np.random.choice([actual_danger_level,*possible_danger_levels_tmp],1,p=[0.9,0.1])
                                    danger_estimates[o] = danger_estimate[0]
                            if near_items_pos:
                                extra_commands.append({"$type": "send_screen_positions", "position_ids": near_items_idx, "positions": near_items_pos, "ids": [self.user_magnebots[idx].robot_id], "frequency": "once"})                
                                duration.append(100)
                                self.user_magnebots[idx].danger_estimates = danger_estimates
                                #actual_danger_level = self.danger_level[temp_all_ids[scre.get_id()]]
                            
                                
                                
                                #extra_positions.append(o)

                        #elif keys.get_pressed(j) == "E":
                        #    self.user_magnebots[0].drop(target=self.ball_id2, arm=Arm.right)
                       
                   
                        #if idx >= 0:
                        #    keys_time_unheld[idx] = 0
                    # Listen for keys currently held down.


                    for j in range(keys.get_num_held()):
                        print(keys.get_held(j))
                        idx = -1
                        
                        if keys.get_held(j) in self.keys_set[0]:
                            idx = self.keys_set[0].index(keys.get_held(j))
                            #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                            #print(self.user_magnebots[idx].action.status)
                            if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                                self.user_magnebots[idx].move_by(distance=10)
                            
                        elif keys.get_held(j) in self.keys_set[1]:
                            idx = self.keys_set[1].index(keys.get_held(j))
                            if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                                self.user_magnebots[idx].move_by(distance=-10)
                        elif keys.get_held(j) in self.keys_set[2]:
                            idx = self.keys_set[2].index(keys.get_held(j))
                            if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                                self.user_magnebots[idx].turn_by(179)
                        elif keys.get_held(j) in self.keys_set[3]:
                            idx = self.keys_set[3].index(keys.get_held(j))
                            if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                                self.user_magnebots[idx].turn_by(-179)
                     
                   
                        if idx >= 0:
                            keys_time_unheld[idx] = 0
                        '''
                        elif keys.get_held(j) == 'W':
                            #new_pos = TDWUtils.array_to_vector3(self.embodied_avatar.transform.position + [0.1,0,0])
                            #commands.extend([{"$type": "move_avatar_towards_position", "position": new_pos, "speed": 0.1, "avatar_id": "b"}])
                            commands.extend([{"$type": "move_avatar_forward_by", "magnitude": 50, "avatar_id": "c"}])
                        elif keys.get_held(j) == 'S':
                         
                            commands.extend([{"$type": "move_avatar_forward_by", "magnitude": -50, "avatar_id": "c"}])
                        elif keys.get_held(j) == 'D':
                            commands.extend([{"$type": "turn_avatar_by", "torque": 50, "avatar_id": "c"}])
                        elif keys.get_held(j) == 'A':
                            commands.extend([{"$type": "turn_avatar_by", "torque": -50, "avatar_id": "c"}])
                        '''

                    # Listen for keys that were released.
                    for j in range(keys.get_num_released()):
                        pdb.set_trace()
                        if keys.get_released(j) == 'UpArrow':
                            #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                            print('stop')
                            self.user_magnebots[0].stop()

                        elif keys.get_released(j) == 'DownArrow':
                         
                            self.user_magnebots[0].stop()
                        elif keys.get_released(j) == 'RightArrow':
                            self.user_magnebots[0].stop()
                        elif keys.get_released(j) == 'LeftArrow':
                            self.user_magnebots[0].stop()

                    if keys.get_num_held() == 0:

                        for um_idx in range(len(self.user_magnebots)):
                            keys_time_unheld[um_idx] += 1
                            #print(keys_time_unheld[um_idx])
                            if keys_time_unheld[um_idx] == 3: #3
                                print("aqui")
                                self.user_magnebots[um_idx].stop()
                                
                                
            
            #print(pil_image)
            #cv2.imshow('frame',all_images)
            #cv2.waitKey(1)
            
            #print(self.user_magnebots[0].action.status)
            '''
            if self.embodied_avatar.action.status != ActionStatus.ongoing:
                print(key)
                if key == 'UpArrow':
                    self.embodied_avatar.move_by(distance=0.2)
                key = ''
            '''
            #Destroy messages after some time
            to_eliminate = []
            for m_idx in range(len(messages)):
                messages[m_idx][2] += 1
                if messages[m_idx][2] == 100:
                    self.uis[messages[m_idx][0]].destroy(messages[m_idx][1])
                    to_eliminate.append(m_idx)
                    if self.terminate:
                        done = True
                
            for te in to_eliminate:
                del messages[te]

            #Draw ui objects
            for key in magnebot_images.keys():
                if key in screen_data:
                    self.add_ui2(magnebot_images[key], screen_data[key])

            #Game ends when all dangerous objects are left in the rug
            goal_counter = 0
            for sd in self.dangerous_objects:
                if np.linalg.norm(self.object_manager.transforms[sd].position-self.object_manager.transforms[self.rug].position) < 1:
                    goal_counter += 1
            if goal_counter == len(self.dangerous_objects):
                txt = um.ui.add_text(text="Success!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 0, "g": 0, "b": 1, "a": 1},
                                     font_size=20
                                     )
                messages.append([idx,txt,0])
                self.terminate = True
                    
            #Show view of magnebot
            cv2.imshow('frame',magnebot_images[str(self.user_magnebots[0].robot_id)])
            cv2.waitKey(1)
            #print('output_loop',time.time()-output_loop)
            #print('all',time.time()-commands_time)
            #print(time.time()-start_time)

            #If timer expires end game, else keep going
            if self.timer <= 0:
                txt = um.ui.add_text(text="Failure!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 0, "g": 0, "b": 1, "a": 1},
                                     font_size=20
                                     )
                messages.append([idx,txt,0])
                self.terminate = True
            else:
                self.timer -= 0.1

            
            
        self.communicate({"$type": "terminate"})


    def _get_midpoint(self) -> np.array:
        """
        :return: The midpoint between the Magenbot and the ball.
        """

        return np.array([(self.magnebot.dynamic.transform.position[0] +
                          self.object_manager.transforms[self.ball_id].position[0]) / 2,
                         0.5,
                         (self.magnebot.dynamic.transform.position[2] +
                          self.object_manager.transforms[self.ball_id].position[2]) / 2])


if __name__ == "__main__":
    c = ChaseBall()
    print('hello')
    c.run()
