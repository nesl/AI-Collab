import numpy as np
import pdb
import cv2
import time
import socketio
import argparse
import pyvirtualcam
import csv
from scipy.spatial.transform import Rotation
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.ui import UI
from tdw.quaternion_utils import QuaternionUtils
from tdw.output_data import OutputData, Images, ScreenPosition, Transforms, Raycast, Keyboard as KBoard
from tdw.add_ons.keyboard import Keyboard
from magnebot import Magnebot, Arm, ActionStatus, ImageFrequency
from magnebot.util import get_default_post_processing_commands

from PIL import Image

from ai_magnebot_controller import AI_Magnebot_Controller



#Dimension of our camera view
width = 640 
height = 480 

num_users = 2
num_ais = 1

cams = []
global_refresh_sensor = 100

address = ''





#This class inherits the magnebot class, we just add a number of attributes over it
class Enhanced_Magnebot(Magnebot):

    def __init__(self,robot_id, position, controlled_by, key_set=None,image_frequency=ImageFrequency.never,pass_masks=['_img'],strength=1):
        super().__init__(robot_id=robot_id, position=position,image_frequency=image_frequency,pass_masks=pass_masks)
        self.key_set = key_set
        self.ui = []
        self.ui_elements = {}
        self.strength = strength
        self.danger_estimates = []
        self.company = {}
        self.controlled_by = controlled_by
        self.focus_object = ""
        self.item_info = {}
        self.estimate_confidence = 0.9
        self.screen_positions = {"position_ids":[],"positions":[],"duration":[]}
        self.refresh_sensor = global_refresh_sensor
        self.messages = []
        self.grasping = False

    


class Simulation(Controller):
  

    def __init__(self, args, port: int = 1071, check_version: bool = True, launch_build: bool = True):
        super().__init__(port=port, check_version=check_version, launch_build=launch_build)

        self.user_magnebots = []
        self.ai_magnebots = []
        self.graspable_objects = []
        self.keys_set = []
        self.uis = []
        self.timer = 1000.0
        self.terminate = False
        self.local = args.local

        ai_spawn_positions = [{"x": -1.4, "y": 0, "z": -1.1},{"x": 0, "y": 0, "z": -1.1}, {"x": 0, "y": 0, "z": -2.1}]
        user_spawn_positions = [{"x": 0, "y": 0, "z": 1.1},{"x": 0, "y": 0, "z": 2.1}, {"x": 4, "y": 0, "z": 1.6}]

        #Functionality of keys according to order of appearance: [Advance, Back, Right, Left, Grab with left arm, Grab with right arm, Camera down, Camera up, Activate sensor, Focus on object]
        proposed_key_sets = []
        with open('keysets.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for r_idx, row in enumerate(csv_reader):
                if r_idx > 0:      
                    proposed_key_sets.append(row)

        #proposed_key_sets = [["UpArrow","DownArrow","RightArrow","LeftArrow","Z","X","C","V","B","N"],["W","S","D","A","H","J","K","L","G","F"],["Alpha5","R","E","Y","U","I","O","P","Alpha0","Alpha9"]]

        #Create ai magnebots
        for ai_idx in range(num_ais):                                   
            self.ai_magnebots.append(Enhanced_Magnebot(robot_id=self.get_unique_id(), position=ai_spawn_positions[ai_idx],image_frequency=ImageFrequency.never, controlled_by='ai'))

        #Create user magnebots
        for us_idx in range(num_users):
            self.user_magnebots.append(Enhanced_Magnebot(robot_id=self.get_unique_id(), position=user_spawn_positions[us_idx], image_frequency=ImageFrequency.always, pass_masks=['_img'],key_set=proposed_key_sets[us_idx], controlled_by='human'))
                                           
                                           
                  
        
        
        reticule_size = 9
        # Create a reticule.
        arr = np.zeros(shape=(reticule_size, reticule_size), dtype=np.uint8)
        x = np.arange(0, arr.shape[0])
        y = np.arange(0, arr.shape[1])
        # Define a circle on the array.
        r = reticule_size // 2
        mask = ((x[np.newaxis, :] - r) ** 2 + (y[:, np.newaxis] - r) ** 2 < r ** 2)
        # Set the color of the reticule.
        arr[mask] = 200
        arr = np.stack((arr,) * 4, axis=-1)
        # Add pointer in the middle

        Image.fromarray(arr).save('pointer.png', "PNG")




        image = "white.png"
        # Set the dimensions of the progress bar.
        self.progress_bar_position = {"x": 16, "y": -16}
        self.progress_bar_size = {"x": 16, "y": 16}
        self.progress_bar_scale = {"x": 10, "y": 2}
        self.progress_bar_anchor = {"x": 0, "y": 1}
        self.progress_bar_pivot = {"x": 0, "y": 1}

            
        #Initializing user interface objects
        for um_idx,um in enumerate(self.user_magnebots):
            um.collision_detection.objects = False
            um.collision_detection.walls = False
            ui = UI(canvas_id=um_idx)
            ui.attach_canvas_to_avatar(avatar_id=str(um.robot_id))
            
            #Create a global key_set
            if um_idx == 0:
                self.keys_set = [[um.key_set[0]],[um.key_set[1]],[um.key_set[2]],[um.key_set[3]],[um.key_set[4]],[um.key_set[5]],[um.key_set[6]],[um.key_set[7]],[um.key_set[8]], [um.key_set[9]]]
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

            
            
            ui.add_image(image='pointer.png',
                        size={"x": reticule_size, "y": reticule_size},
                        rgba=True,
                        position={"x": 0, "y": 0})

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
          
        #Needed to get objects positions
        self.object_manager: ObjectManager = ObjectManager()

        self.required_strength = {}
        self.danger_level = {} 
        self.dangerous_objects = []
        
        #Add-ons
        self.add_ons.extend([*self.ai_magnebots,  *self.user_magnebots, self.object_manager, *self.uis])


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
        
        


        #Instantiate and locate objects
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
                self.required_strength[object_id] = int(np.random.choice([1,2,3],1)[0])
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

        #Create a rug
        self.rug: int = self.get_unique_id()
        commands.extend(self.get_add_physics_object(model_name="carpet_rug",
                                         object_id=self.rug,
                                         position={"x": 0, "y": 0, "z": 0},
                                         rotation={"x": 0, "y": 0, "z": 0}))
        
        #Creating third person camera
        commands.extend(TDWUtils.create_avatar(position={"x": -3.15, "y": 10, "z": 0.22},#{"x": 0, "y": 10, "z": -1},
                                                           look_at={"x": 0, "y": 0, "z": 0},
                                                           avatar_id="a"))
        commands.extend([{"$type": "set_pass_masks","pass_masks": ["_img"],"avatar_id": "a"},
                  {"$type": "send_images","frequency": "always","ids": ["a"]},
                  {"$type": "set_img_pass_encoding", "value": False},
                  {"$type": "set_render_order", "render_order": 1, "sensor_name": "SensorContainer", "avatar_id": "a"},
                  {"$type": "send_keyboard", "frequency": "always"}])

        
        self.communicate(commands)

        

        #Initializing communication with server
        
        
        self.target = {};
        
        self.user_magnebots_ids = [str(um.robot_id) for um in self.user_magnebots]
        self.ai_magnebots_ids = [str(um.robot_id) for um in self.ai_magnebots]


        self.sio = None

        if not self.local:
            self.sio = socketio.Client(ssl_verify=False)
            
            @self.sio.event
            def connect():
                print("I'm connected!")
                
                self.sio.emit("simulator", [*self.user_magnebots_ids, *self.ai_magnebots_ids])

            @self.sio.event
            def connect_error(data):
                print("The connection failed!")

            @self.sio.event
            def disconnect():
                print("I'm disconnected!")
                
            @self.sio.event
            def set_goal(agent_id,obj_id):
                print("Received new goal")
                self.target[agent_id] = obj_id
                
            @self.sio.event
            def ai_message(message, source_agent_id, agent_id):
                ai_magnebot = self.ai_magnebots[self.ai_magnebots_ids.index(agent_id)]
                ai_magnebot.messages.append((source_agent_id,message))
                print("message", message, source_agent_id, agent_id)
                
            self.sio.connect(address)


    

    def add_ui(self, original_image, screen_positions):
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
    


    def raycast_output(self, resp, all_ids):

        raycast = Raycast(resp)
        print("raycast from ", raycast.get_raycast_id(), raycast.get_hit(), raycast.get_hit_object(), raycast.get_object_id() in self.graspable_objects, str(raycast.get_raycast_id()) in self.user_magnebots_ids, raycast.get_object_id())
             
        o_id = raycast.get_object_id()
        
        if raycast.get_hit() and raycast.get_hit_object() and o_id in self.graspable_objects and str(raycast.get_raycast_id()) in self.user_magnebots_ids: #If ray hits an object
        
            
            pos_idx = len(all_ids)+self.graspable_objects.index(o_id)
            
            
            u_idx = self.user_magnebots_ids.index(str(raycast.get_raycast_id()))
            
            if not self.user_magnebots[u_idx].grasping: #Grasping also uses raycasting but we don't want to use this code for that situation
            
                self.user_magnebots[u_idx].screen_positions["position_ids"].append(pos_idx)
                self.user_magnebots[u_idx].screen_positions["positions"].append(TDWUtils.array_to_vector3(raycast.get_point()))
                self.user_magnebots[u_idx].screen_positions["duration"].append(100)
                
                print("raycasted ", raycast.get_object_id(), raycast.get_point())
                
                self.user_magnebots[u_idx].focus_object = o_id
                
                if o_id not in self.user_magnebots[u_idx].item_info:
                    self.user_magnebots[u_idx].item_info[o_id] = {}
                    
                self.user_magnebots[u_idx].item_info[o_id]['weight'] = int(self.required_strength[o_id])

                if not self.local:
                    self.sio.emit('objects_update', (u_idx,self.user_magnebots[idx].item_info))


    def screen_output(self, resp, screen_data, all_magnebots, all_ids):

        scre = ScreenPosition(resp)
                    
        idx = self.user_magnebots_ids.index(scre.get_avatar_id())
        
        if scre.get_id() in all_magnebots[idx].screen_positions['position_ids']: #Screen coordinate was requested by particular magnebot
        
            scre_coords = scre.get_screen()

            scre_coords = (scre_coords[0],height-scre_coords[1],scre_coords[2])
            
            if not (scre_coords[0] < 0 or scre_coords[0] > width or scre_coords[1] < 0 or scre_coords[1] > height or scre_coords[2] < 0): #Screen coordinates should not surpass limits
            
                temp_all_ids = all_ids + self.graspable_objects
                mid = temp_all_ids[scre.get_id()]
                color = (255, 255, 255)                        


                #Coordinates can be for a magnebot or object
                if mid in self.ai_magnebots_ids:
                    mid = 'A_'+mid
                elif mid in self.user_magnebots_ids:
                    mid = 'U_'+mid
                else: #For object

                    avatar = self.user_magnebots[self.user_magnebots_ids.index(scre.get_avatar_id())]
                    if mid in avatar.danger_estimates:
                        danger_estimate = avatar.danger_estimates[mid]
                    else:
                        danger_estimate = 0
         
                    mid = str(mid)
                    
      
                    if danger_estimate >= 2: #Different color for different danger estimate
                        color = (0, 0, 255)
                    elif danger_estimate == 1:
                        color = (0, 255, 0)
                    else:
                        color = (255, 255, 255)

                if scre.get_avatar_id() not in screen_data:
                    screen_data[scre.get_avatar_id()] = {}
                    screen_data[scre.get_avatar_id()]['coords'] = [scre_coords]
                    screen_data[scre.get_avatar_id()]['ids'] = [mid]
                    screen_data[scre.get_avatar_id()]['color'] = [color]
                else:
                    screen_data[scre.get_avatar_id()]['coords'].append(scre_coords)
                    screen_data[scre.get_avatar_id()]['ids'].append(mid)
                    screen_data[scre.get_avatar_id()]['color'].append(color)


    def keyboard_output(self, resp, extra_commands, duration, keys_time_unheld, all_ids, messages):

        keys = KBoard(resp)

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

            elif keys.get_pressed(j) in self.keys_set[4] or keys.get_pressed(j) in self.keys_set[5]: #Pick up/Drop with one of the arms
                if keys.get_pressed(j) in self.keys_set[4]:
                    arm = Arm.left
                    key_idx = 4
                else:
                    arm = Arm.right
                    key_idx = 5
                    
                idx = self.keys_set[key_idx].index(keys.get_pressed(j))
                
                if self.user_magnebots[idx].dynamic.held[arm].size > 0: #Press once to pick up, twice to drop
                    self.user_magnebots[idx].drop(target=self.user_magnebots[idx].dynamic.held[arm][0], arm=arm)
                    self.user_magnebots[idx].grasping = False
                else: #Pick up object you have focused in
                    
                    #Object can be too heavy to carry alone, or you may have picked the wrong object (dangerous)
                    grasp_object = self.user_magnebots[idx].focus_object
                    if grasp_object:
                        print("grasping", grasp_object, arm, idx)
                        if self.user_magnebots[idx].strength < self.required_strength[grasp_object]:
                            txt = self.user_magnebots[idx].ui.add_text(text="Too heavy to carry alone!!",
                             position={"x": 0, "y": 0},
                             color={"r": 0, "g": 0, "b": 1, "a": 1},
                             font_size=20
                             )
                            messages.append([idx,txt,0])
                        else:
                            self.user_magnebots[idx].grasp(target=grasp_object, arm=arm)
                            self.user_magnebots[idx].grasping = True
                            self.user_magnebots[idx].in_danger = True
                            if grasp_object in self.dangerous_objects and 'ai' not in self.user_magnebots[idx].company.values():
                                for um in self.user_magnebots:
                                    txt = um.ui.add_text(text="Dangerous object picked without help!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 0, "g": 0, "b": 1, "a": 1},
                                     font_size=20
                                     )
                                    messages.append([idx,txt,0])
                                self.terminate = True
                            

                    
            elif keys.get_pressed(j) in self.keys_set[6]: #Move camera down
                idx = self.keys_set[6].index(keys.get_pressed(j))
                self.user_magnebots[idx].rotate_camera(pitch=10)

            elif keys.get_pressed(j) in self.keys_set[7]: #Move camera up
                idx = self.keys_set[7].index(keys.get_pressed(j))
                self.user_magnebots[idx].rotate_camera(pitch=-10)

            elif keys.get_pressed(j) in self.keys_set[8]: #Estimate danger level
                idx = self.keys_set[8].index(keys.get_pressed(j))

                near_items_pos = []
                near_items_idx = []
                danger_estimates = {}
                possible_danger_levels = [1,2]
                
                if self.user_magnebots[idx].refresh_sensor >= global_refresh_sensor: #Check if our sensor is refreshed
                    self.user_magnebots[idx].refresh_sensor = 0
                    
                    for o_idx,o in enumerate(self.graspable_objects): #Sensor only actuates over objects that are in a certain radius
                        if np.linalg.norm(self.object_manager.transforms[o].position -
                                self.user_magnebots[idx].dynamic.transform.position) < 2:
                            near_items_idx.append(len(all_ids)+o_idx)
                            near_items_pos.append(TDWUtils.array_to_vector3(self.object_manager.transforms[o].position))
                            actual_danger_level = self.danger_level[o]
                            
                            
                            if o not in self.user_magnebots[idx].item_info:
                                self.user_magnebots[idx].item_info[o] = {}
                                
                            self.user_magnebots[idx].item_info[o]['weight'] = int(self.required_strength[o])
                            
                            if 'sensor' not in self.user_magnebots[idx].item_info[o]:
                                self.user_magnebots[idx].item_info[o]['sensor'] = {}
                            
                            #Get danger estimation, value and confidence level
                            if self.user_magnebots[idx].robot_id not in self.user_magnebots[idx].item_info[o]['sensor']:
                                possible_danger_levels_tmp = possible_danger_levels.copy()
                                possible_danger_levels_tmp.remove(actual_danger_level)
                            
                                danger_estimate = np.random.choice([actual_danger_level,*possible_danger_levels_tmp],1,p=[self.user_magnebots[idx].estimate_confidence,1-self.user_magnebots[idx].estimate_confidence])
                                danger_estimates[o] = danger_estimate[0]
                                
                                self.user_magnebots[idx].item_info[o]['sensor'][self.user_magnebots[idx].robot_id] = {}
                                self.user_magnebots[idx].item_info[o]['sensor'][self.user_magnebots[idx].robot_id]['value'] = int(danger_estimate[0])
                                self.user_magnebots[idx].item_info[o]['sensor'][self.user_magnebots[idx].robot_id]['confidence'] = self.user_magnebots[idx].estimate_confidence
                                
                            else: #If we already have a danger estimation reuse that one
                                danger_estimates[o] = self.user_magnebots[idx].item_info[o]['sensor'][self.user_magnebots[idx].robot_id]['value']
                                
                    if near_items_pos:
                        
                        
                        self.user_magnebots[idx].screen_positions["position_ids"].extend(near_items_idx)
                        self.user_magnebots[idx].screen_positions["positions"].extend(near_items_pos)
                        self.user_magnebots[idx].screen_positions["duration"].extend([100]*len(near_items_idx)) 
                        
                        self.user_magnebots[idx].danger_estimates = danger_estimates
                        
                        if not self.local:
                            self.sio.emit('objects_update', (idx,self.user_magnebots[idx].item_info))
                       
                
            
            elif keys.get_pressed(j) in self.keys_set[9]: #Focus on object, use raycasting, needs adjustment
                idx = self.keys_set[9].index(keys.get_pressed(j))
                
                
                #print(angle, x_new, y_new, z_new, real_camera_position)
                camera_position_relative = np.array([-0.1838, 0.053+0.737074, 0])
                
                #print({"x": x_new, "y": y_new, "z": z_new}, self.user_magnebots[idx].dynamic.transform.position)
                r1 = Rotation.from_quat(self.user_magnebots[idx].dynamic.transform.rotation)
                r2 = Rotation.from_euler('zxy', self.user_magnebots[idx].camera_rpy, degrees=True)
                r3 = r2*r1
                print(r3.inv().apply([0,0,1])*np.array([-1,-1,1])+self.user_magnebots[idx].dynamic.transform.position,self.user_magnebots[idx].dynamic.transform.position)
                print(r1.as_euler('xyz', degrees=True))
                
                print(r2.as_euler('zyx', degrees=True))
                new_camera_position_relative = r1.inv().apply(camera_position_relative)
                source = r3.inv().apply([0,0,0])*np.array([-1,-1,1])+self.user_magnebots[idx].dynamic.transform.position+new_camera_position_relative
                destination = r3.inv().apply([0,0,1])*np.array([-1,-1,1])+self.user_magnebots[idx].dynamic.transform.position+new_camera_position_relative
                
                extra_commands.append({"$type":"send_raycast",
                   "origin": TDWUtils.array_to_vector3(source),
                   "destination": TDWUtils.array_to_vector3(destination),
                   "id": str(self.user_magnebots[idx].robot_id)}) 
                
                duration.append(1)
                
                

        # Listen for keys currently held down. This is mainly for movement keys


        for j in range(keys.get_num_held()):
            #print(keys.get_held(j))
            idx = -1
            
            if keys.get_held(j) in self.keys_set[0]: #Advance
                idx = self.keys_set[0].index(keys.get_held(j))
                #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                #print(self.user_magnebots[idx].action.status)
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].move_by(distance=10)
                
            elif keys.get_held(j) in self.keys_set[1]: #Back
                idx = self.keys_set[1].index(keys.get_held(j))
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].move_by(distance=-10)
            elif keys.get_held(j) in self.keys_set[2]: #Right
                idx = self.keys_set[2].index(keys.get_held(j))
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].turn_by(179)
            elif keys.get_held(j) in self.keys_set[3]: #Left
                idx = self.keys_set[3].index(keys.get_held(j))
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].turn_by(-179)
         
       
            if idx >= 0:
                keys_time_unheld[idx] = 0
            

        # Listen for keys that were released. DOESN'T WORK
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


        if keys.get_num_held() == 0: #After some time unheld we stop the current action

            for um_idx in range(len(self.user_magnebots)):
                keys_time_unheld[um_idx] += 1
                #print(keys_time_unheld[um_idx])
                if keys_time_unheld[um_idx] == 3: #3
                    print("aqui")
                    self.user_magnebots[um_idx].stop()


        

    def run(self):
        done = False
        commands = []
        key = ""
        messages = []
        extra_commands = []
        duration = []
        
        keys_time_unheld = [0]*len(self.user_magnebots_ids)
        all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]

        
        #Include the positions of other magnebots in the view of all user magnebots
        for um in self.user_magnebots:
            um.screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
            um.screen_positions["positions"].extend([-1]*len(all_ids))
            um.screen_positions["duration"].extend([-1]*len(all_ids))
            
        #We initialize the controller for ai agents
        ai_controllers = []
        for am in self.ai_magnebots:
            ai_controllers.append(AI_Magnebot_Controller(am))
        
        print("User ids: ", self.user_magnebots_ids, "AI ids: ", self.ai_magnebots_ids)
        

        #Loop until simulation ends
        while not done:
            start_time = time.time()
            
            screen_positions = {"position_ids":[],"positions":[]}
            
            #Track magnebots positions
            user_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.user_magnebots]
            ai_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.ai_magnebots]
            
            
                

            #Set a visual target whenever the user wants to help
            if self.target:
                temp_all_ids = all_ids + self.graspable_objects
                position_ids = []
                agent_ids = []
                positions = []
                
                for t in self.target.keys():
                
                    position_id = temp_all_ids.index(int(self.target[t]))
                    if not position_id in self.user_magnebots[t].screen_positions["position_ids"]:
                        self.user_magnebots[t].screen_positions["position_ids"].append(position_id)
                        self.user_magnebots[t].screen_positions["positions"].append(TDWUtils.array_to_vector3(self.object_manager.transforms[int(self.target[t])].position))
                        self.user_magnebots[t].screen_positions["duration"].append(-1)
                    else:
                        position_index = self.user_magnebots[t].screen_positions["position_ids"].index(position_id)
                        self.user_magnebots[t].screen_positions["positions"][position_index] = TDWUtils.array_to_vector3(self.object_manager.transforms[int(self.target[t])].position)
                
                
              
                    
                    
            commands_time = time.time()
            
            #Some extra commands to send and when to remove them
            to_eliminate = []
            for ex_idx in range(len(extra_commands)):
                duration[ex_idx] -= 1
                if not duration[ex_idx]:
                    to_eliminate.append(ex_idx)
                commands.append(extra_commands[ex_idx])
            
            for e in to_eliminate:
                del duration[e]
                del extra_commands[e]
                
                
                

            #We update timer
            mins, remainder = divmod(self.timer, 60)
            secs,millisecs = divmod(remainder,1)

            object_info_update = []
            
            #Update all stats related with closeness of magnebots, like strength factor
            #Iterate over all magnebots
            for idx in range(len(all_magnebots)):
                robot_id = all_magnebots[idx].robot_id
                all_magnebots[idx].strength = 1
                company = {}
                
                
                
                for idx2 in range(len(all_magnebots)):
                    if idx == idx2:
                        continue
                    if np.linalg.norm(all_magnebots[idx].dynamic.transform.position - all_magnebots[idx2].dynamic.transform.position) < 2: #Check only two dimensions not three
                        all_magnebots[idx].strength += 1 #Increase strength
                        company[all_magnebots[idx2].robot_id] = all_magnebots[idx2].controlled_by #Add information about neighbors
                        
                        #Update object info entries when closeby
                        if not all_magnebots[idx].item_info == all_magnebots[idx2].item_info:
                            for it_element in all_magnebots[idx2].item_info.keys():
                                if it_element not in all_magnebots[idx].item_info:
                                    all_magnebots[idx].item_info[it_element] = all_magnebots[idx2].item_info[it_element]
                                else:
                                    if 'sensor' in all_magnebots[idx2].item_info[it_element]:
                                        if 'sensor' in all_magnebots[idx].item_info[it_element]:
                                            all_magnebots[idx].item_info[it_element]['sensor'].update(all_magnebots[idx2].item_info[it_element]['sensor'])
                                        else:
                                            all_magnebots[idx].item_info[it_element]['sensor'] = all_magnebots[idx2].item_info[it_element]['sensor']
                            
                            all_magnebots[idx2].item_info = all_magnebots[idx].item_info 
                            object_info_update.extend([idx,idx2])
                            
                            
                #Transmit neighbors info
                if not all_magnebots[idx].company == company:
                    all_magnebots[idx].company = company          
                    if not self.local:      
                        self.sio.emit('neighbors_update', (idx,all_magnebots[idx].company)) 
                      
                #Refresh danger level sensor             
                if all_magnebots[idx].refresh_sensor < global_refresh_sensor:
                    all_magnebots[idx].refresh_sensor += 1              
                            
                if all_magnebots[idx].ui_elements: #For user magnebots, update user interface
                    
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[1],text=f"Strength: {all_magnebots[idx].strength}")
                    all_magnebots[idx].ui.set_size(ui_id=all_magnebots[idx].ui_elements[0], size={"x": int(self.progress_bar_size["x"] * self.progress_bar_scale["x"] * (all_magnebots[idx].strength-1)/10),    "y": int(self.progress_bar_size["y"] * self.progress_bar_scale["y"])})

                    #We modify timer
                    all_magnebots[idx].ui.set_text(ui_id=all_magnebots[idx].ui_elements[2],text='{:02d}:{:02d}'.format(int(mins), int(secs)))
                    
                    
                    #Add screen position markers requested by each particular user magnebot
                    
                    to_delete = []
                    for sc_idx in range(len(all_magnebots[idx].screen_positions['positions'])):
                        
                        if all_magnebots[idx].screen_positions['duration'][sc_idx] > 0:
                            all_magnebots[idx].screen_positions['duration'][sc_idx] -= 1
                        
                            if all_magnebots[idx].screen_positions['duration'][sc_idx] == 0:
                                to_delete.append(sc_idx)
                            elif all_magnebots[idx].screen_positions['position_ids'][sc_idx] not in screen_positions['position_ids']:

                                screen_positions["position_ids"].append(all_magnebots[idx].screen_positions['position_ids'][sc_idx])
                                screen_positions["positions"].append(all_magnebots[idx].screen_positions['positions'][sc_idx])
                        
                    for e in to_delete:
                        try:
                            del all_magnebots[idx].screen_positions['position_ids'][e]                           
                            del all_magnebots[idx].screen_positions['positions'][e]    
                            del all_magnebots[idx].screen_positions['duration'][e]    
                        except:
                            pdb.set_trace()

                for arm in [Arm.right,Arm.left]:
                    if all_magnebots[idx].dynamic.held[arm].size > 0:
                        #Drop object if strength decreases
                        if self.required_strength[all_magnebots[idx].dynamic.held[arm][0]] > all_magnebots[idx].strength:
                            all_magnebots[idx].drop(target=all_magnebots[idx].dynamic.held[arm][0], arm=arm)
                            all_magnebots[idx].grasping = False
                        #Terminate game if dangerous object held alone
                        if all_magnebots[idx].dynamic.held[arm][0] in self.dangerous_objects:
                            
                            if (all_magnebots[idx].controlled_by == 'ai' and 'human' not in all_magnebots[idx].company.values()) or (all_magnebots[idx].controlled_by == 'human' and 'ai' not in all_magnebots[idx].company.values()):
                                for um in self.user_magnebots:
                                    txt = um.ui.add_text(text="Dangerous object picked without help!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 0, "g": 0, "b": 1, "a": 1},
                                     font_size=20
                                     )
                                    messages.append([idx,txt,0])
                                self.terminate = True
                            
            #Share object info
            object_info_update = list(set(object_info_update))

            if not self.local:
                for ob_idx in object_info_update:
                    self.sio.emit('objects_update', (ob_idx,all_magnebots[ob_idx].item_info)) 
             
             
            #Ask for the given screen positions of certain objects/magnebots     
            screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
            screen_positions["positions"].extend([*user_magnebots_positions,*ai_magnebots_positions])
            
            commands.append({"$type": "send_screen_positions", "position_ids": screen_positions["position_ids"], "positions":screen_positions["positions"], "ids": [*self.user_magnebots_ids], "frequency": "once"})

            resp = self.communicate(commands)
            


            commands.clear()
            
            
            #Step through controllers for ai
            for aic in ai_controllers:
                aic.controller(self.object_manager, self.sio)

            
            
            all_images = []

            screen_data = {}
            magnebot_images = {}
 

            #Output data
            for i in range(len(resp) - 1):
                
                r_id = OutputData.get_data_type_id(resp[i])

                # Get Images output data.
                if r_id == "imag":
                    images = Images(resp[i])
                    # Determine which avatar captured the image. In this case, the third person camera
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
                                if cams:
                                    cams[0].send(img_image)
                                #cv2.imshow('frame',np.asarray(pil_image))
                                #cv2.waitKey(1)
                                
                            
                    
                    #Process images from magnebot cameras
                    elif images.get_avatar_id() in self.user_magnebots_ids:
                        idx = self.user_magnebots_ids.index(images.get_avatar_id())
                        img_image = np.asarray(self.user_magnebots[idx].dynamic.get_pil_images()['img'])
                        magnebot_images[images.get_avatar_id()] = img_image
                        
                    
                        
    
                elif r_id == "scre": #Get screen coordinates from objects
                    self.screen_output(resp[i], screen_data, all_magnebots, all_ids)

                        
                    
                elif r_id == "rayc":   #Raycast information (given ray vector, which objects are in its path) Needs adjustment. Activated when focusing on an object
      
                    self.raycast_output(resp[i], all_ids)
                    
                    
                elif r_id == "keyb":#For each keyboard key pressed
                    self.keyboard_output(resp[i], extra_commands, duration, keys_time_unheld, all_ids, messages)
                    
                                
                                


            #Destroy messages in the user interface after some time
            to_eliminate = []
            for m_idx in range(len(messages)):
                messages[m_idx][2] += 1
                if messages[m_idx][2] == 100:
                    self.uis[messages[m_idx][0]].destroy(messages[m_idx][1])
                    to_eliminate.append(m_idx)
                    if self.terminate:
                        done = True
                
            for te in to_eliminate:
                try:
                    del messages[te]
                except:
                    pdb.set_trace()


            #Draw ui objects
            for key in magnebot_images.keys():
                if key in screen_data:
                    self.add_ui(magnebot_images[key], screen_data[key])


            #Game ends when all dangerous objects are left in the rug
            goal_counter = 0
            for sd in self.dangerous_objects:
                if np.linalg.norm(self.object_manager.transforms[sd].position-self.object_manager.transforms[self.rug].position) < 1:
                    goal_counter += 1
            if goal_counter == len(self.dangerous_objects):
                for idx,um in enumerate(self.user_magnebots):
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
            
            #Send frames to virtual cameras in system
            if cams:
                for idx in range(len(self.user_magnebots_ids)):
                    cams[idx+1].send(magnebot_images[self.user_magnebots_ids[idx]])
                
                
            
            #If timer expires end simulation, else keep going
            if self.timer <= 0:
                for idx,um in enumerate(self.user_magnebots):
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


    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='run locally only')
    parser.add_argument('--no_virtual_cameras', action='store_true', help='do not stream frames to virtual cameras')
    parser.add_argument('--address', type=str, default='https://172.17.15.69:4000' ,help='adress to connect to')
    args = parser.parse_args()

    #The web interface expects to get frames from camera devices. We simulate this by using v4l2loopback to create some virtual webcams to which we forward the frames generated in here
    if not args.no_virtual_cameras:
        for user in range(num_users+1):
            cams.append(pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video'+str(user)))

    address = args.address

    c = Simulation(args)

    c.run()
