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

    def __init__(self,robot_id, position,image_frequency):
        super().__init__(robot_id=robot_id, position=position,image_frequency=image_frequency)
        self.id = robot_id
        

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
        self.ui_elements = {}
        self.strength = {}
        self.grasped_object = []

        # Add the robot, the Magnebot, and an object manager.
        self.robot: Robot = Robot(robot_id=self.get_unique_id(), name="ur5", position={"x": -1.4, "y": 0, "z": 2.6})#{"x": 1.88, "y": 0, "z": 0.37})#{"x": -1.4, "y": 0, "z": 2.6})
        self.magnebot: Magnebot = Magnebot(robot_id=self.get_unique_id(), position={"x": -1.4, "y": 0, "z": -1.1},#position={"x": -1.97, "y": 0, "z": 3.11},  #{"x": -1.4, "y": 0, "z": -1.1},
                                           image_frequency=ImageFrequency.never)
                                           
        self.ai_magnebots.append(self.magnebot)
        self.strength[self.magnebot.robot_id] = 1

        self.user_magnebots.append(Magnebot(robot_id=self.get_unique_id(), position={"x": -3.3, "y": 0, "z": 1.6}, #{"x": 2, "y": 0, "z": 2},
                                           image_frequency=ImageFrequency.always, pass_masks=['_img']))
                                           
        self.strength[self.user_magnebots[0].robot_id] = 1

        self.keys_set = [["UpArrow"],["DownArrow"],["RightArrow"],["LeftArrow"],["Z"],["X"],["C"],["V"]]

        
        #self.user_magnebots.append(Magnebot(robot_id=self.get_unique_id(), position={"x": 3, "y": 0, "z": 1.6},
        #                                   image_frequency=ImageFrequency.always, pass_masks=['_img']))
                                           
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
            
            self.uis.append(ui)
            self.ui_elements[um.robot_id] = ((bar_id,text_id))
            self.grasped_object.append("")

        self.object_manager: ObjectManager = ObjectManager()
        # Add a ball.
        self.ball_id: int = self.get_unique_id()
        self.ball_id2: int = self.get_unique_id()
        self.box: int = self.get_unique_id()
        self.graspable_objects.extend([self.ball_id,self.ball_id2,self.box])
        self.required_strength = {self.ball_id:1,self.ball_id2:1,self.box:2}
        
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
        self.add_ons.extend([self.robot, self.magnebot,  *self.user_magnebots, self.object_manager, *self.uis]) #, self.keyboard])#, image_capture]) #image_capture, self.person])
        #self.add_ons.extend([self.object_manager, self.camera, image_capture])

        # Create the scene.

        commands = [{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                    #TDWUtils.create_empty_room(9, 9),
                    self.get_add_material("parquet_long_horizontal_clean",
                                          library="materials_high.json"),
                    {"$type": "set_screen_size",
                     "width": width, #640,
                     "height": height}, #480},
                    {"$type": "rotate_directional_light_by",
                     "angle": 30,
                     "axis": "pitch"}]
        
        #self.communicate(self.get_add_scene(scene_name="tdw_room"))
        #commands = []
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
        
        commands.extend(TDWUtils.create_avatar(position={"x": -3.15, "y": 10, "z": 0.22},#{"x": 0, "y": 10, "z": -1},
                                                           look_at={"x": 0, "y": 0, "z": 0},
                                                           avatar_id="a"))
        commands.extend([{"$type": "set_pass_masks","pass_masks": ["_img"],"avatar_id": "a"},
                  {"$type": "send_images","frequency": "always","ids": ["a"]},
                  {"$type": "set_img_pass_encoding", "value": False},
                  {"$type": "set_render_order", "render_order": 1, "sensor_name": "SensorContainer", "avatar_id": "a"},
                  {"$type": "send_keyboard", "frequency": "always"}])

        
        self.communicate(commands)
        self.state = State.initializing
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
        colorFont = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 2
        
        for s_idx,s in enumerate(screen_positions['coords']):

            cv2.putText(original_image, screen_positions['ids'][s_idx], (int(s[0]),int(s[1])), font, fontScale, colorFont, thickness, cv2.LINE_AA)
    
    def run(self):
        done = False
        commands = []
        counter = 0
        key = ""
        messages = []
        once = 0
        
        user_magnebots_ids = [str(um.robot_id) for um in self.user_magnebots]
        ai_magnebots_ids = [str(um.robot_id) for um in self.ai_magnebots]
        keys_time_unheld = [0]*len(user_magnebots_ids)
        all_ids = [*user_magnebots_ids,*ai_magnebots_ids]
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
        
        while not done:
            start_time = time.time()
            
            user_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.user_magnebots]
            ai_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.ai_magnebots]
            
            commands = [{"$type": "send_screen_positions", "position_ids": list(range(0,len(all_ids))), "positions": [*user_magnebots_positions,*ai_magnebots_positions], "ids": ["a",*user_magnebots_ids], "frequency": "once"}]
            commands_time = time.time()

            resp = self.communicate(commands)

            commands.clear()
            #print('commands time', time.time()-commands_time)

            for idx in range(len(all_magnebots)):
                robot_id = all_magnebots[idx].robot_id
                self.strength[robot_id] = 1
                for idx2 in range(len(all_magnebots)):
                    if idx == idx2:
                        continue
                    if np.linalg.norm(all_magnebots[idx].dynamic.transform.position - all_magnebots[idx2].dynamic.transform.position) < 2: #Check only two dimensions not three
                        self.strength[robot_id] += 1
                if robot_id in self.ui_elements:
                    #We assume self.uis and all_magnebots have the same sequence
                    self.uis[idx].set_text(ui_id=self.ui_elements[robot_id][1],text=f"Strength: {self.strength[robot_id]}")
                    self.uis[idx].set_size(ui_id=self.ui_elements[robot_id][0], size={"x": int(self.progress_bar_size["x"] * self.progress_bar_scale["x"] * (self.strength[robot_id]-1)/10),    "y": int(self.progress_bar_size["y"] * self.progress_bar_scale["y"])})

                for arm in [Arm.right,Arm.left]:
                    if all_magnebots[idx].dynamic.held[arm].size > 0:
                        if self.required_strength[all_magnebots[idx].dynamic.held[arm][0]] > self.strength[robot_id]:
                            all_magnebots[idx].drop(target=all_magnebots[idx].dynamic.held[arm][0], arm=arm)
                        
                

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
            # Initialize the robots.
            if self.state == State.initializing:
                # Stop initializing and start swinging.
                if not self.robot.joints_are_moving() and self.magnebot.action.status != ActionStatus.ongoing:
                    # Now that the robot isn't intersecting with the floor, it is safe to request collision data.
                    commands.append({"$type": "send_collisions",
                                     "enter": True,
                                     "stay": False,
                                     "exit": False,
                                     "collision_types": ["obj"]})
                    # Rotate the shoulder to swing at the ball.
                    self.robot.set_joint_targets(targets={self.robot.static.joint_ids_by_name["shoulder_link"]: -70})
                    self.state = State.swinging
            elif self.state == State.swinging:
                for collision in self.robot.dynamic.collisions_with_objects:
                    # The first element in `collision` is always a body part and the second element is always an object.
                    if collision[1] == self.ball_id:
                        # Start moving the Magnebot towards the ball.
                        self.state = State.moving_to_ball
                        self.magnebot.move_to(target=self.ball_id, arrived_offset=0.3)
                        # Reset the robot.
                        self.robot.set_joint_targets(targets={self.robot.static.joint_ids_by_name["shoulder_link"]: 0})
            elif self.state == State.moving_to_ball:
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
            elif self.state == State.backing_away:
                # The Magnebot has moved away from the robot. We're done!
                if np.linalg.norm([-0.871,0,3] - self.magnebot.dynamic.transform.position) > 2 and np.linalg.norm([-0.871,0,3]-self.object_manager.transforms[self.ball_id2].position) < 1:
                    self.robot.set_joint_targets(targets={self.robot.static.joint_ids_by_name["shoulder_link"]: -70})

                    if self.magnebot.action.status != ActionStatus.ongoing and not self.robot.joints_are_moving():
                        done = True
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
                    
                        mid = all_ids[scre.get_id()]
                        
                        if mid in ai_magnebots_ids:
                            mid = 'A_'+mid
                        else:
                            mid = 'U_'+mid
                        if scre.get_avatar_id() not in screen_data:
                            screen_data[scre.get_avatar_id()] = {}
                            screen_data[scre.get_avatar_id()]['coords'] = [scre_coords]
                            screen_data[scre.get_avatar_id()]['ids'] = [mid]
                        else:
                            screen_data[scre.get_avatar_id()]['coords'].append(scre_coords)
                            screen_data[scre.get_avatar_id()]['ids'].append(mid)

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
                                

                            keys_time_unheld[idx] = 0

                        elif keys.get_pressed(j) in self.keys_set[1]: #Back
                            idx = self.keys_set[1].index(keys.get_pressed(j))
                            self.user_magnebots[idx].move_by(distance=-10)
                            keys_time_unheld[idx] = 0
                        elif keys.get_pressed(j) in self.keys_set[2]: #Right
                            idx = self.keys_set[2].index(keys.get_pressed(j))
                            self.user_magnebots[idx].turn_by(179)
                            keys_time_unheld[idx] = 0
                        elif keys.get_pressed(j) in self.keys_set[3]: #Left
                            idx = self.keys_set[3].index(keys.get_pressed(j))
                            self.user_magnebots[idx].turn_by(-179)
                            keys_time_unheld[idx] = 0
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
                                    if self.strength[self.user_magnebots[idx].robot_id] < self.required_strength[o]:
                                        txt = self.uis[idx].add_text(text="Too heavy to carry alone!!",
                                         position={"x": 0, "y": 0},
                                         color={"r": 0, "g": 0, "b": 1, "a": 1},
                                         font_size=20
                                         )
                                        messages.append([idx,txt,0])
                                    else:
                                        self.user_magnebots[idx].grasp(target=grasp_object, arm=arm)
                                    #self.user_magnebots[0].grasp(target=self.box, arm=Arm.right)
                                
                        elif keys.get_pressed(j) in self.keys_set[6]:
                            idx = self.keys_set[6].index(keys.get_pressed(j))
                            self.user_magnebots[idx].rotate_camera(pitch=10)
                        elif keys.get_pressed(j) in self.keys_set[7]:
                            idx = self.keys_set[7].index(keys.get_pressed(j))
                            self.user_magnebots[idx].rotate_camera(pitch=-10)

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
                            print(keys_time_unheld[um_idx])
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
            
            for m_idx in range(len(messages)):
                messages[m_idx][2] += 1
                if messages[m_idx][2] == 100:
                    self.uis[messages[m_idx][0]].destroy(messages[m_idx][1])
                    del messages[m_idx]
                

            for key in magnebot_images.keys():
                if key in screen_data:
                    self.add_ui2(magnebot_images[key], screen_data[key])
            
            cv2.imshow('frame',magnebot_images[str(self.user_magnebots[0].robot_id)])
            cv2.waitKey(1)
            #print('output_loop',time.time()-output_loop)
            #print('all',time.time()-commands_time)
            #print(time.time()-start_time)
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
