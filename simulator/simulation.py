import numpy as np
import pdb
import cv2
import time
import socketio
import argparse
import pyvirtualcam
import csv
import json_numpy
import yaml
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

from tdw.add_ons.occupancy_map import OccupancyMap

from PIL import Image



#Dimension of our camera view
width = 640 
height = 480 

num_users = 2
num_ais = 1

cams = []
global_refresh_sensor = 0

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
        self.past_status = ActionStatus.ongoing
        self.view_radius = 0
        self.centered_view = 0
        

    

#Main class
class Simulation(Controller):
  

    def __init__(self, args, cfg, port: int = 1071, check_version: bool = True, launch_build: bool = True):
        super().__init__(port=port, check_version=check_version, launch_build=launch_build)

         
        
        self.keys_set = []
        self.local = args.local
        self.options = args
        self.cfg = cfg
        self.no_debug_camera = args.no_debug_camera
        
        self.reset = False
        
        self.timer = float(self.cfg['timer'])
        self.ai_skip_frames = int(self.cfg['ai_skip_frames'])
        
        self.occupancy_map_request = []
        self.objects_held_status_request = []
        self.danger_sensor_request = []
        self.ai_status_request = []
        self.raycast_request = []
        self.queue_perception_action = []
        self.extra_keys_pressed = []
        
        

        #Functionality of keys according to order of appearance: [Advance, Back, Right, Left, Grab with left arm, Grab with right arm, Camera down, Camera up, Activate sensor, Focus on object]
        self.proposed_key_sets = []
        with open('keysets.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for r_idx, row in enumerate(csv_reader):
                if r_idx > 0:      
                    self.proposed_key_sets.append(row)

        #proposed_key_sets = [["UpArrow","DownArrow","RightArrow","LeftArrow","Z","X","C","V","B","N"],["W","S","D","A","H","J","K","L","G","F"],["Alpha5","R","E","Y","U","I","O","P","Alpha0","Alpha9"]]

        

        #Creating occupancy map
        self.static_occupancy_map = OccupancyMap(cell_size=self.cfg['cell_size'])       

        
        

        #Add-ons
        self.add_ons.extend([self.static_occupancy_map])


        # Create the scene.

        commands = self.create_scene()
        

        

        self.communicate(commands)

        self.static_occupancy_map.generate() #Get occupancy map only with walls
        
        self.communicate([])
        
        #print(self.static_occupancy_map.occupancy_map[:20,:20])
        #pdb.set_trace()

        self.user_magnebots = []
        self.ai_magnebots = []
        self.ai_spawn_positions = [{"x": -2, "y": 0, "z": 1.1},{"x": -2, "y": 0, "z": 2.1}, {"x": -2, "y": 0, "z": 3.1}, {"x": -3, "y": 0, "z": 0.1}, {"x": -2, "y": 0, "z": 0.1},{"x": -2, "y": 0, "z": -1.1}, {"x": -2, "y": 0, "z": -2.1},{"x": -2, "y": 0, "z": -3.1},{"x": -3, "y": 0, "z": -1.1},{"x": -3, "y": 0, "z": -2.1}, {"x": -3, "y": 0, "z": 1.1}, {"x": -3, "y": 0, "z": 2.1}, {"x": -3.5, "y": 0, "z": 0.5}, {"x": -3.5, "y": 0, "z": 1.5}, {"x": -3.5, "y": 0, "z": 2.5}, {"x": -3.5, "y": 0, "z": 3.5}, {"x": -3.5, "y": 0, "z": -2.5}, {"x": -3.5, "y": 0, "z": -3.5}]
        self.user_spawn_positions = [{"x": 0, "y": 0, "z": 1.1},{"x": 0, "y": 0, "z": 2.1}, {"x": 0, "y": 0, "z": 3.1}, {"x": 1, "y": 0, "z": 0.1}, {"x": 0, "y": 0, "z": 0.1},{"x": 0, "y": 0, "z": -1.1}, {"x": 0, "y": 0, "z": -2.1},{"x": 0, "y": 0, "z": -3.1},{"x": 1, "y": 0, "z": -3.1},{"x": 1, "y": 0, "z": -2.1}]
        self.uis = []

        #Create ai magnebots
        for ai_idx in range(num_ais):                                   
            self.ai_magnebots.append(Enhanced_Magnebot(robot_id=self.get_unique_id(), position=self.ai_spawn_positions[ai_idx],image_frequency=ImageFrequency.always, controlled_by='ai'))
        
        #Create user magnebots
        for us_idx in range(num_users):
            self.user_magnebots.append(Enhanced_Magnebot(robot_id=self.get_unique_id(), position=self.user_spawn_positions[us_idx], image_frequency=ImageFrequency.always, pass_masks=['_img'],key_set=self.proposed_key_sets[us_idx], controlled_by='human'))



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
            um.collision_detection.objects = True
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

        self.add_ons.extend([*self.ai_magnebots,  *self.user_magnebots, self.object_manager, *self.uis])

        self.user_magnebots_ids = [str(um.robot_id) for um in self.user_magnebots]
        self.ai_magnebots_ids = [str(um.robot_id) for um in self.ai_magnebots]
        

        
        commands = self.populate_world()
        
        

        self.communicate(commands)





        

        #pdb.set_trace()

        
        #print(self.static_occupancy_map.occupancy_map)


        #Initializing communication with server
        
        

        self.sio = None

        #Socket io event functions
        if not self.local:
            self.sio = socketio.Client(ssl_verify=False)
            
            @self.sio.event
            def connect():
                print("I'm connected!")
                
                #Occupancy map info
                extra_config = {}
        
                extra_config['edge_coordinate'] = self.static_occupancy_map.get_occupancy_position(0,0)
                extra_config['cell_size'] = self.cfg['cell_size']
                extra_config['num_cells'] = self.static_occupancy_map.occupancy_map.shape
                extra_config['num_objects'] = len(self.graspable_objects)
                extra_config['all_robots'] = [(str(um.robot_id),um.controlled_by) for um in [*self.user_magnebots,*self.ai_magnebots]]
                
                
                self.sio.emit("simulator", (self.user_magnebots_ids,self.ai_magnebots_ids, self.options.video_index, extra_config))#[*self.user_magnebots_ids, *self.ai_magnebots_ids])

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
                
            """
            @self.sio.event
            def ai_message(message, source_agent_id, agent_id):
                ai_magnebot = self.ai_magnebots[self.ai_magnebots_ids.index(agent_id)]
                ai_magnebot.messages.append((source_agent_id,message))
                print("message", message, source_agent_id, agent_id)
            """

            #Receive action for ai controlled robot
            @self.sio.event
            def ai_action(action_message, agent_id):
                print('New command:', action_message, agent_id)
                ai_agent_idx = self.ai_magnebots_ids.index(agent_id)
                ai_agent = self.ai_magnebots[ai_agent_idx]
                

                
                
                for actions in action_message:
                
                    if 'send_' in actions[0]: # or 'send_occupancy_map' in actions[0] or 'send_objects_held_status' in actions[0]:
                        eval_string = "self."
                    else:
                        eval_string = "ai_agent."
                    
                    
                    eval_string += actions[0]+"("

                    for a_idx, argument in enumerate(actions[1:]):
                        if a_idx:
                            eval_string += ','
                        eval_string += argument

                    if 'send_' in actions[0]:
                        if len(actions) > 1:
                            eval_string += ','
                        eval_string +=  '"' + agent_id + '")'
                        self.queue_perception_action.append([eval_string,self.ai_skip_frames])
                    else:
                        #print("Eval string", eval_string)

                        eval(eval_string + ")")


            #Indicate use of occupancy maps
            @self.sio.event
            def watcher_ai(agent_id, view_radius, centered):
                ai_agent_idx = self.ai_magnebots_ids.index(agent_id)

                self.ai_magnebots[ai_agent_idx].view_radius = int(view_radius)
                self.ai_magnebots[ai_agent_idx].centered_view = int(centered)
               
            #Reset environment
            @self.sio.event 
            def reset(agent_id):
                self.reset = True
                
            #Key
            @self.sio.event
            def key(key, agent_id):
                user_agent_idx = self.user_magnebots_ids.index(agent_id)
                
                if key in self.user_magnebots[0].key_set: #Check whether key is in magnebot key set
                    k_idx = self.user_magnebots[0].key_set.index(key)
                    self.extra_keys_pressed.append(self.user_magnebots[user_agent_idx].key_set[k_idx]) #Key is converted to required one
                    #print(key, self.user_magnebots[user_agent_idx].key_set[k_idx])
                
                
            self.sio.connect(address)


    #Create the scene environment
    def create_scene(self):
    
        commands = [#{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                    {"$type": "load_scene", "scene_name": "ProcGenScene"},
                    TDWUtils.create_empty_room(10, 10),
                    self.get_add_material("parquet_long_horizontal_clean",
                                          library="materials_high.json"),
                    {"$type": "set_screen_size",
                     "width": width, #640,
                     "height": height}, #480},
                    {"$type": "rotate_directional_light_by",
                     "angle": 30,
                     "axis": "pitch"}]
                     
        fps = int(self.cfg['fps'])     
        if fps:    
            commands.append({"$type": "set_target_framerate", "framerate": fps})
        '''
        commands = [#{'$type': 'add_scene','name': 'building_site','url': 'https://tdw-public.s3.amazonaws.com/scenes/linux/2019.1/building_site'}, 
                    {"$type": "load_scene", "scene_name": "ProcGenScene"},
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
        '''
        return commands

    #Used to create all objects
    def populate_world(self):
    
        
        self.graspable_objects = []
        
        
        self.timer = float(self.cfg['timer'])
        self.terminate = False
    
        

        self.required_strength = {}
        self.danger_level = {} 
        self.dangerous_objects = []

        
        #self.communicate([])

        commands = []

        #Instantiate and locate objects
        max_coord = 3#8
        object_models = ['iron_box'] #['iron_box','4ft_shelf_metal','trunck','lg_table_marble_green','b04_backpack','36_in_wall_cabinet_wood_beach_honey']
        coords = {}
        
        #coords[object_models[0]] = [[max_coord,max_coord],[max_coord-1,max_coord-0.1],[max_coord-0.5,max_coord-0.2],[max_coord-0.4,max_coord],[max_coord,max_coord-0.5]]
        coords[object_models[0]] = [[max_coord,max_coord]]
        #coords[object_models[1]] = [[max_coord-3,max_coord]]

        #coords[object_models[2]] = [[max_coord,max_coord-3]]
        #coords[object_models[3]] = [[max_coord-2,max_coord-2]]
        #coords[object_models[4]] = [[max_coord-1,max_coord-2]]
        #coords[object_models[5]] = [[max_coord-3,max_coord-3]]

        modifications = [[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]]

        final_coords = {}

        for objm in object_models:
            final_coords[objm] = []
        

        for fc in final_coords.keys():
            for m in modifications:
                final_coords[fc].extend(np.array(coords[fc])*m)

        for fc in final_coords.keys():
            for c in final_coords[fc]:

                weight = int(np.random.choice([1,2,3],1)[0])
                danger_level = np.random.choice([1,2],1,p=[0.9,0.1])[0]
                #commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},10,danger_level,weight))
                commands.extend(self.instantiate_object(fc,{"x": c[0], "y": 0, "z": c[1]},{"x": 0, "y": 0, "z": 0},10,2,1)) #Danger level 2 and weight 1
                #print("Position:", {"x": c[0], "y": 0, "z": c[1]})


        #commands.extend(self.instantiate_object('iron_box',{"x": 0, "y": 0, "z": 0},{"x": 0, "y": 0, "z": 0},10,1,1)) #Single box



        # Add post-processing.
        commands.extend(get_default_post_processing_commands())

        
        #Create a rug
        self.rug: int = self.get_unique_id()
        """
        commands.extend(self.get_add_physics_object(model_name="carpet_rug",
                                         object_id=self.rug,
                                         position={"x": 0, "y": 0, "z": 0},
                                         rotation={"x": 0, "y": 0, "z": 0}))
        """
        
        
                  
        self.target = {}
        
        #Creating third person camera
        #commands.extend(TDWUtils.create_avatar(position={"x": 0, "y": 10, "z": 0},#{"x": 0, "y": 10, "z": -1},
        #                                                   look_at={"x": 0, "y": 0, "z": 0},
        #                                                   avatar_id="a"))
                                
        if not self.no_debug_camera:       
                        
            commands.extend([{"$type": "create_avatar", "type": "A_Img_Caps_Kinematic", "id": "a"}, 
            {"$type": "teleport_avatar_to", "avatar_id": "a", "position": {"x": 0, "y": 10, "z": 0}},
            {"$type": "look_at_position", "avatar_id": "a", "position": {"x": 0, "y": 0, "z": 0}},
            {"$type": "rotate_avatar_by", "angle": 90, "axis": "yaw", "is_world": True, "avatar_id": "a"}])
            commands.extend([{"$type": "set_pass_masks","pass_masks": ["_img"],"avatar_id": "a"},
                      {"$type": "send_images","frequency": "always","ids": ["a"]},
                      {"$type": "set_img_pass_encoding", "value": False},
                      {"$type": "set_render_order", "render_order": 1, "sensor_name": "SensorContainer", "avatar_id": "a"}])
                  
        commands.append({"$type": "send_keyboard", "frequency": "always"})
        
        
        
        
        return commands

    
    #Function to instantiate objects
    def instantiate_object(self, model_name, position, rotation, mass, danger_level, required_strength):

        object_id = self.get_unique_id()
        self.graspable_objects.append(object_id)
        self.required_strength[object_id] = required_strength
        self.danger_level[object_id] = danger_level
        command = self.get_add_physics_object(model_name=model_name,
                                         object_id=object_id,
                                         position=position,
                                         rotation=rotation,
                                         default_physics_values=False,
                                         mass=mass,
                                         scale_mass=False)
        if self.danger_level[object_id] == 2:
            self.dangerous_objects.append(object_id)

        return command

    #Function to add ui to camera frames
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
    

    #Process raycasting
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
                self.user_magnebots[u_idx].item_info[o_id]['time'] = self.timer
                self.user_magnebots[u_idx].item_info[o_id]['location'] = self.object_manager.transforms[o_id].position.tolist()


                self.raycast_request.append(str(self.user_magnebots[u_idx].robot_id))
                '''
                if not self.local:
                    self.sio.emit('objects_update', (str(self.user_magnebots[u_idx].robot_id),self.user_magnebots[u_idx].item_info))
                '''

    #Get screen coordinates of objects
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


    '''
    #Process keyboard presses
    def keyboard_output(self, resp, extra_commands, duration, keys_time_unheld, all_ids, messages):

        keys = KBoard(resp)
        
        #print(keys.get_num_pressed(), keys.get_num_held(), keys.get_num_released(), self.frame_num)

        # Listen for events where the key was first pressed on the previous frame.
    '''
    
    #Process keyboard presses
    def keyboard_output(self, key_pressed, key_hold, extra_commands, duration, keys_time_unheld, all_ids, messages):
    
        #for j in range(keys.get_num_pressed()):
        for j in range(len(key_pressed)):
            idx = -1
            if key_pressed[j] in self.keys_set[0]: #Advance
                idx = self.keys_set[0].index(key_pressed[j])
                #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].move_by(distance=10)
                    

                #keys_time_unheld[idx] = -20

            elif key_pressed[j] in self.keys_set[1]: #Back
                idx = self.keys_set[1].index(key_pressed[j])
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].move_by(distance=-10)
                #keys_time_unheld[idx] = -20

            elif key_pressed[j] in self.keys_set[2]: #Right
                idx = self.keys_set[2].index(key_pressed[j])
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].turn_by(179)
                #keys_time_unheld[idx] = -20

            elif key_pressed[j] in self.keys_set[3]: #Left
                idx = self.keys_set[3].index(key_pressed[j])
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].turn_by(-179)
                #keys_time_unheld[idx] = -20

            elif key_pressed[j] in self.keys_set[4] or key_pressed[j] in self.keys_set[5]: #Pick up/Drop with one of the arms
                if key_pressed[j] in self.keys_set[4]:
                    arm = Arm.left
                    key_idx = 4
                else:
                    arm = Arm.right
                    key_idx = 5
                    
                idx = self.keys_set[key_idx].index(key_pressed[j])
                
                if self.user_magnebots[idx].dynamic.held[arm].size > 0: #Press once to pick up, twice to drop
                    self.user_magnebots[idx].drop(target=self.user_magnebots[idx].dynamic.held[arm][0], arm=arm)
                    self.user_magnebots[idx].grasping = False
                    
                    '''
                    extra_commands.append({"$type":"send_raycast",
                   "origin": TDWUtils.array_to_vector3(source),
                   "destination": TDWUtils.array_to_vector3(destination),
                   "id": str(self.user_magnebots[idx].robot_id)}) 
                
                    duration.append(1)
                    self.user_magnebots[idx].dynamic.held[arm][0]
                    '''
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
                            #If dangerous object carried without being accompanied by an ai if human or by a human if an ai, ends the simulation
                            '''
                            if grasp_object in self.dangerous_objects and 'ai' not in self.user_magnebots[idx].company.values():
                                for um in self.user_magnebots:
                                    txt = um.ui.add_text(text="Dangerous object picked without help!",
                                     position={"x": 0, "y": 0},
                                     color={"r": 0, "g": 0, "b": 1, "a": 1},
                                     font_size=20
                                     )
                                    messages.append([idx,txt,0])
                                self.terminate = True
                            '''
                            

                    
            elif key_pressed[j] in self.keys_set[6]: #Move camera down
                idx = self.keys_set[6].index(key_pressed[j])
                self.user_magnebots[idx].rotate_camera(pitch=10)

            elif key_pressed[j] in self.keys_set[7]: #Move camera up
                idx = self.keys_set[7].index(key_pressed[j])
                self.user_magnebots[idx].rotate_camera(pitch=-10)

            elif key_pressed[j] in self.keys_set[8]: #Estimate danger level
                idx = self.keys_set[8].index(key_pressed[j])
                self.danger_sensor_request.append(str(self.user_magnebots[idx].robot_id))

                '''
                idx,item_info = self.danger_sensor_reading(self.user_magnebots[idx].robot_id)
                if not self.local:
                    self.sio.emit('objects_update', (str(self.user_magnebots[idx].robot_id),item_info))
                '''
                
            
            elif key_pressed[j] in self.keys_set[9]: #Focus on object, use raycasting, needs adjustment
                idx = self.keys_set[9].index(key_pressed[j])
                
                '''
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
                '''
                #print(self.user_magnebots[idx].robot_id, idx, key_pressed, self.keys_set)
                extra_commands.append({"$type": "send_mouse_raycast",
                              "id": str(self.user_magnebots[idx].robot_id),
                              "avatar_id": str(self.user_magnebots[idx].robot_id)})
                duration.append(1)
                
            #elif key_pressed[j] == 'P':
            #    self.reset = True
            
            if idx >= 0:
                keys_time_unheld[idx] = 0            


        # Listen for keys currently held down. This is mainly for movement keys


        for j in range(len(key_hold)):
            #print(key_hold[j])
            idx = -1
            
            if key_hold[j] in self.keys_set[0]: #Advance
                idx = self.keys_set[0].index(key_hold[j])
                #if self.user_magnebots[0].action.status != ActionStatus.ongoing:
                #print(self.user_magnebots[idx].action.status)
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].move_by(distance=10)
                
            elif key_hold[j] in self.keys_set[1]: #Back
                idx = self.keys_set[1].index(key_hold[j])
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].move_by(distance=-10)
            elif key_hold[j] in self.keys_set[2]: #Right
                idx = self.keys_set[2].index(key_hold[j])
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].turn_by(179)
            elif key_hold[j] in self.keys_set[3]: #Left
                idx = self.keys_set[3].index(key_hold[j])
                if self.user_magnebots[idx].action.status != ActionStatus.ongoing:
                    self.user_magnebots[idx].turn_by(-179)
         
       
            if idx >= 0:
                keys_time_unheld[idx] = 0
            
        '''
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

        '''
        
        if len(key_hold) == 0: #After some time unheld we stop the current action

            for um_idx in range(len(self.user_magnebots)):
                keys_time_unheld[um_idx] += 1
                #print(keys_time_unheld[um_idx])
                if keys_time_unheld[um_idx] == 3: #3
                    print("aqui")
                    self.user_magnebots[um_idx].stop()


    

    def danger_sensor_reading(self, robot_id):
    
        all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]
        idx = all_ids.index(str(robot_id))
        ego_magnebot = all_magnebots[idx]
    
        near_items_pos = []
        near_items_idx = []
        danger_estimates = {}
        possible_danger_levels = [1,2]
        
        if ego_magnebot.refresh_sensor >= global_refresh_sensor: #Check if our sensor is refreshed
            ego_magnebot.refresh_sensor = 0
            
            for o_idx,o in enumerate(self.graspable_objects): #Sensor only actuates over objects that are in a certain radius
                if np.linalg.norm(self.object_manager.transforms[o].position -
                        ego_magnebot.dynamic.transform.position) < 2:
                    near_items_idx.append(len(all_ids)+o_idx)
                    near_items_pos.append(TDWUtils.array_to_vector3(self.object_manager.transforms[o].position))
                    actual_danger_level = self.danger_level[o]
                    
                    
                    if o not in ego_magnebot.item_info:
                        ego_magnebot.item_info[o] = {}
                        
                    ego_magnebot.item_info[o]['weight'] = int(self.required_strength[o])
                    
                    if 'sensor' not in ego_magnebot.item_info[o]:
                        ego_magnebot.item_info[o]['sensor'] = {}
                    
                    #Get danger estimation, value and confidence level
                    if ego_magnebot.robot_id not in ego_magnebot.item_info[o]['sensor']:
                        possible_danger_levels_tmp = possible_danger_levels.copy()
                        possible_danger_levels_tmp.remove(actual_danger_level)
                    
                        danger_estimate = np.random.choice([actual_danger_level,*possible_danger_levels_tmp],1,p=[ego_magnebot.estimate_confidence,1-ego_magnebot.estimate_confidence])
                        danger_estimates[o] = danger_estimate[0]
                        
                        ego_magnebot.item_info[o]['sensor'][ego_magnebot.robot_id] = {}
                        ego_magnebot.item_info[o]['sensor'][ego_magnebot.robot_id]['value'] = int(danger_estimate[0])
                        ego_magnebot.item_info[o]['sensor'][ego_magnebot.robot_id]['confidence'] = ego_magnebot.estimate_confidence

                        
                    else: #If we already have a danger estimation reuse that one
                        danger_estimates[o] = ego_magnebot.item_info[o]['sensor'][ego_magnebot.robot_id]['value']
                        
                        
                    ego_magnebot.item_info[o]['time'] = self.timer
                    ego_magnebot.item_info[o]['location'] = self.object_manager.transforms[o].position.tolist()
                        
            #If objects were detected
            if near_items_pos:
                
                #To have the information displayed in the screen
                ego_magnebot.screen_positions["position_ids"].extend(near_items_idx)
                ego_magnebot.screen_positions["positions"].extend(near_items_pos)
                ego_magnebot.screen_positions["duration"].extend([100]*len(near_items_idx)) 
                
                ego_magnebot.danger_estimates = danger_estimates
        '''         
                if not self.local:
                    #print("objects_update", (idx,ego_magnebot.item_info))
                    self.sio.emit('objects_update', (idx,ego_magnebot.item_info))
            else:
                if not self.local:
                    self.sio.emit('objects_update', (idx,ego_magnebot.item_info))
        else:
            self.sio.emit('objects_update', (idx,ego_magnebot.item_info))
        '''
        
        return idx, ego_magnebot.item_info
            
    def send_occupancy_map(self, magnebot_id):
    
        self.occupancy_map_request.append(magnebot_id)
        
        
    def send_objects_held_status(self,magnebot_id):
    
        self.objects_held_status_request.append(magnebot_id)
        
    def send_danger_sensor_reading(self, magnebot_id):
    
        self.danger_sensor_request.append(magnebot_id)
            
    def get_occupancy_map(self, magnebot_id):
                    
        
        m_idx = self.ai_magnebots_ids.index(magnebot_id)
        locations_magnebot_map = {}
        
        magnebots_locations = np.where(self.object_type_coords_map == 3)
        locations_magnebot_map = {str(j):[magnebots_locations[0][i],magnebots_locations[1][i]] for i in range(len(magnebots_locations[0])) for j in self.object_attributes_id[str(magnebots_locations[0][i])+'_'+str(magnebots_locations[1][i])]}

        x = locations_magnebot_map[magnebot_id][0]
        y = locations_magnebot_map[magnebot_id][1]
        
        
        if magnebot_id in self.occupancy_map_request and self.ai_magnebots[m_idx].view_radius:
            
            self.occupancy_map_request.remove(magnebot_id)
        

            view_radius = self.ai_magnebots[m_idx].view_radius
            
            

            x_min = max(0,x-view_radius)
            y_min = max(0,y-view_radius)
            x_max = min(self.object_type_coords_map.shape[0]-1,x+view_radius)
            y_max = min(self.object_type_coords_map.shape[1]-1,y+view_radius)
            #limited_map = np.zeros_like(self.static_occupancy_map.occupancy_map)
            
            #Magnebot is at the center of the occupancy  map always or not
            if self.ai_magnebots[m_idx].centered_view:
                limited_map = np.zeros((view_radius*2+1,view_radius*2+1)) #+1 as we separately count the row/column where the magnebot is currently in
                #limited_map[:,:] = self.static_occupancy_map.occupancy_map[x_min:x_max+1,y_min:y_max+1]
                limited_map[:,:] = self.object_type_coords_map[x_min:x_max+1,y_min:y_max+1]
                objects_locations = np.where(limited_map > 1)
                reduced_metadata = {}
                limited_map[x-x_min,y-y_min] = 5

                for ol in range(len(objects_locations[0])):
                    rkey = str(objects_locations[0][ol]+x_min)+'_'+str(objects_locations[1][ol]+y_min)
                    rkey2 = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                    reduced_metadata[rkey2] = self.object_attributes_id[rkey]
            else:
                limited_map = np.zeros_like(self.object_type_coords_map)
                limited_map[[0,limited_map.shape[0]-1],:] = -1
                limited_map[:,[0,limited_map.shape[1]-1]] = -1
                limited_map[x_min:x_max+1,y_min:y_max+1] = self.object_type_coords_map[x_min:x_max+1,y_min:y_max+1]
                objects_locations = np.where(limited_map > 1)
                reduced_metadata = {}
                limited_map[x,y] = 5
                
                for ol in range(len(objects_locations[0])):
                    rkey = str(objects_locations[0][ol])+'_'+str(objects_locations[1][ol])
                    reduced_metadata[rkey] = self.object_attributes_id[rkey]
                
                
            """
            for ol in range(len(objects_locations[0])):
                rkey = str(objects_locations[0][ol]+x_min)+str(objects_locations[1][ol]+y_min)
                pdb.set_trace()
                if magnebot_id in object_attributes_id[rkey]:
                    limited_map[x,y] = 5
                else:
                    limited_map[objects_locations[0][ol],objects_locations[1][ol]] = reduced_object_type_coords_map[objects_locations[0][ol],objects_locations[1][ol]]
                    rkey2 = str(objects_locations[0][ol])+str(objects_locations[1][ol])
                    
                    reduced_metadata[rkey2] = object_attributes_id[rkey]
            
            
            for om in range(len(magnebots_locations[0])):
                if magnebots_locations[0][om] >= x_min and magnebots_locations[0][om] <= x_max and magnebots_locations[1][om] >= y_min and magnebots_locations[1][om] <= y_max:
                    rkey = str(magnebots_locations[0][om])+str(magnebots_locations[1][om])
                    if not magnebot_id in object_attributes_id[rkey]:
                        limited_map[magnebots_locations[0][om]-x_min,magnebots_locations[1][om]-y_min] = 3
                        rkey2 = str(magnebots_locations[0][om]-x_min)+str(magnebots_locations[1][om]-y_min)
                        reduced_metadata[rkey2] = object_attributes_id[rkey]
                
            """
            #print(limited_map)

            #pdb.set_trace()
            #limited_map[x_min:x_max+1,y_min:y_max+1] = self.static_occupancy_map.occupancy_map[x_min:x_max+1,y_min:y_max+1]
            
            
       
            
            #self.sio.emit('occupancy_map', (all_idx, json_numpy.dumps(limited_map), reduced_metadata, objects_held))
            
        else:
            limited_map = np.zeros_like(self.object_type_coords_map)
            limited_map[x,y] = 5
            reduced_metadata = {}

            
        return limited_map, reduced_metadata
        
    def get_objects_held_state(self, magnebot_id):
    
        #Check the objects held in each arm
        objects_held = [0,0]
        m_idx = self.ai_magnebots_ids.index(magnebot_id)
        
        if magnebot_id in self.objects_held_status_request:
        
            self.objects_held_status_request.remove(magnebot_id)
            for arm_idx, arm in enumerate([Arm.right,Arm.left]):
                
                if self.ai_magnebots[m_idx].dynamic.held[arm].size > 0:
                    objects_held[arm_idx] = int(self.ai_magnebots[m_idx].dynamic.held[arm][0])
                
        return objects_held
    
    
    #### Main Loop
    def run(self):
        done = False
        commands = []
        key = ""
        messages = []
        extra_commands = []
        duration = []
        estimated_fps = 0
        past_time = time.time()
        self.frame_num = 0
        
        
        keys_time_unheld = [0]*len(self.user_magnebots_ids)
        all_ids = [*self.user_magnebots_ids,*self.ai_magnebots_ids]
        all_magnebots = [*self.user_magnebots,*self.ai_magnebots]

        
        #Include the positions of other magnebots in the view of all user magnebots
        for um in self.user_magnebots:
            um.screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
            um.screen_positions["positions"].extend([-1]*len(all_ids))
            um.screen_positions["duration"].extend([-1]*len(all_ids))
            
        time_gone = time.time()
        
        print("User ids: ", self.user_magnebots_ids, "AI ids: ", self.ai_magnebots_ids)
        

        #Loop until simulation ends
        while not done:
            start_time = time.time()
            
            screen_positions = {"position_ids":[],"positions":[]}
            
            #Track magnebots positions
            user_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.user_magnebots]
            ai_magnebots_positions = [TDWUtils.array_to_vector3(um.dynamic.transform.position + np.array([0,0.5,0])) for um in self.ai_magnebots]
            
            #Prepare occupancy maps and associated metadata
            #object_attributes_id stores the ids of the objects and magnebots
            #object_type_coords_map creates a second occupancy map with objects and magnebots

            self.object_type_coords_map = np.copy(self.static_occupancy_map.occupancy_map)
            min_pos = self.static_occupancy_map.get_occupancy_position(0,0)[0]
            multiple = self.cfg['cell_size']
            self.object_attributes_id = {}
            
            for o in self.graspable_objects:
                pos = self.object_manager.transforms[o].position
                pos_new = [round((pos[0]+abs(min_pos))/multiple), round((pos[2]+abs(min_pos))/multiple)]
                #2 is for objects
                self.object_type_coords_map[pos_new[0],pos_new[1]] = 2
                if str(pos_new[0])+'_'+str(pos_new[1]) not in self.object_attributes_id:
                    self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])] = []
                self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])].append((o,self.required_strength[o]))
            #pdb.set_trace()
            for o in [*self.user_magnebots,*self.ai_magnebots]:
                pos = o.dynamic.transform.position
                pos_new = [round((pos[0]+abs(min_pos))/multiple), round((pos[2]+abs(min_pos))/multiple)]
                #3 is for other magnebots
                self.object_type_coords_map[pos_new[0],pos_new[1]] = 3
                if str(pos_new[0])+'_'+str(pos_new[1]) not in self.object_attributes_id:
                    self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])] = []
                self.object_attributes_id[str(pos_new[0])+'_'+str(pos_new[1])].append((str(o.robot_id)))

            #pdb.set_trace()
            

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
                    if np.linalg.norm(all_magnebots[idx].dynamic.transform.position - all_magnebots[idx2].dynamic.transform.position) < 2: #TODO Check only two dimensions not three
                        all_magnebots[idx].strength += 1 #Increase strength
                        
                        
                    if np.linalg.norm(all_magnebots[idx].dynamic.transform.position - all_magnebots[idx2].dynamic.transform.position) < 5: #TODO Check only two dimensions not three, view radius
                    
                        company[all_magnebots[idx2].robot_id] = (all_magnebots[idx2].controlled_by, all_magnebots[idx2].dynamic.transform.position.tolist()) #Add information about neighbors
                        
                all_magnebots[idx].company = company 
                        
                        
                '''
                        #Update object info entries when nearby
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
                                            
                                #Newest information based on time
                                if all_magnebots[idx].item_info[it_element]['time'] > all_magnebots[idx2].item_info[it_element]['time']:
                                    all_magnebots[idx].item_info[it_element]['time'] = all_magnebots[idx2].item_info[it_element]['time']
                                    all_magnebots[idx].item_info[it_element]['location'] = all_magnebots[idx2].item_info[it_element]['location']
                                
                            all_magnebots[idx2].item_info = all_magnebots[idx].item_info 
                            object_info_update.extend([idx,idx2])
                '''
                        
                '''      
                #Transmit neighbors info
                if not all_magnebots[idx].company == company:
                    all_magnebots[idx].company = company          
                    if not self.local:      
                        self.sio.emit('neighbors_update', (idx,all_magnebots[idx].company)) 
                '''
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
                            print("Error deleting")
                            #pdb.set_trace()

                for arm in [Arm.right,Arm.left]:
                    if all_magnebots[idx].dynamic.held[arm].size > 0:
                        #Drop object if strength decreases
                        if self.required_strength[all_magnebots[idx].dynamic.held[arm][0]] > all_magnebots[idx].strength:
                            all_magnebots[idx].drop(target=all_magnebots[idx].dynamic.held[arm][0], arm=arm)
                            all_magnebots[idx].grasping = False
                        #Terminate game if dangerous object held alone
                        '''
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
                        '''

                #Transmit ai controlled robots status
                if not self.local and all_magnebots[idx] in self.ai_magnebots:
                    if all_magnebots[idx].action.status != all_magnebots[idx].past_status:
                        all_magnebots[idx].past_status = all_magnebots[idx].action.status
                        #self.sio.emit("ai_status", (idx,all_magnebots[idx].action.status.value))


            '''
                            
            #Share object info
            object_info_update = list(set(object_info_update))

            if not self.local:
                for ob_idx in object_info_update:
                    self.sio.emit('objects_update', (ob_idx,all_magnebots[ob_idx].item_info)) 
            '''
             
            #Ask for the given screen positions of certain objects/magnebots     
            screen_positions["position_ids"].extend(list(range(0,len(all_ids))))
            screen_positions["positions"].extend([*user_magnebots_positions,*ai_magnebots_positions])
            
            commands.append({"$type": "send_screen_positions", "position_ids": screen_positions["position_ids"], "positions":screen_positions["positions"], "ids": [*self.user_magnebots_ids], "frequency": "once"})

            
            
            resp = self.communicate(commands)
            duration_fps = time.time()-past_time
            estimated_fps = (estimated_fps + 1/duration_fps)/2
            #print(estimated_fps)
            past_time = time.time()


            commands.clear()

            
            
            all_images = []

            screen_data = {}
            magnebot_images = {}
 
            key_pressed = []
            key_hold = []

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
                                
                            
                    
                    #Process images from user magnebot cameras
                    elif images.get_avatar_id() in self.user_magnebots_ids:
                        idx = self.user_magnebots_ids.index(images.get_avatar_id())
                        img_image = np.asarray(self.user_magnebots[idx].dynamic.get_pil_images()['img'])
                        magnebot_images[images.get_avatar_id()] = img_image
                    #Process images from ai magnebot cameras
                    elif images.get_avatar_id() in self.ai_magnebots_ids:
                        idx = self.ai_magnebots_ids.index(images.get_avatar_id())
                        img_image = np.asarray(self.ai_magnebots[idx].dynamic.get_pil_images()['img'])
                        magnebot_images[images.get_avatar_id()] = img_image
                        
                    
                        
    
                elif r_id == "scre": #Get screen coordinates from objects
                    self.screen_output(resp[i], screen_data, all_magnebots, all_ids)

                        
                    
                elif r_id == "rayc":   #Raycast information (given ray vector, which objects are in its path) Needs adjustment. Activated when focusing on an object
      
                    self.raycast_output(resp[i], all_ids)
                    
                    

                elif r_id == "keyb":#For each keyboard key pressed
                    keys = KBoard(resp[i])
                    key_pressed_tmp = [keys.get_pressed(j) for j in range(keys.get_num_pressed())]
                    key_hold_tmp = [keys.get_held(j) for j in range(keys.get_num_held())]
                    
                    key_pressed.extend(key_pressed_tmp)
                    key_hold.extend(key_hold_tmp)
                    
                    
                                
                                
            #Process keyboard output
            key_pressed.extend(self.extra_keys_pressed)
            self.extra_keys_pressed = []
            if key_pressed:
                print(key_pressed)    
            self.keyboard_output(key_pressed, key_hold, extra_commands, duration, keys_time_unheld, all_ids, messages)
            
            
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
            
            """"
            for sd in self.dangerous_objects:
                if np.linalg.norm(self.object_manager.transforms[sd].position-self.object_manager.transforms[self.rug].position) < 1:
                    goal_counter += 1
            """
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
            if str(self.user_magnebots[0].robot_id) in magnebot_images:
                cv2.imshow('frame',magnebot_images[str(self.user_magnebots[0].robot_id)])
                cv2.waitKey(1)
            
            
            to_remove = []
            #Execute delayed actions
            for qa_idx in range(len(self.queue_perception_action)):
                if not self.queue_perception_action[qa_idx][1]:
                    self.queue_perception_action[qa_idx][1] -= 1
                else:
                    eval(self.queue_perception_action[qa_idx][0])
                    to_remove.append(qa_idx)
                    
            for tr in to_remove:
                del self.queue_perception_action[tr]
            
            
            
            #Send frames to virtual cameras in system and occupancy maps if required, and all outputs needed

            idx = 0

            for magnebot_id in self.user_magnebots_ids:
            
                if cams and magnebot_id in magnebot_images:
                    cams[idx+1].send(magnebot_images[magnebot_id])
                    
                item_info = {}
                all_idx = all_ids.index(str(magnebot_id))
                
                if magnebot_id in self.danger_sensor_request:
                    _,item_info = self.danger_sensor_reading(magnebot_id)
                    self.danger_sensor_request.remove(magnebot_id)
                    
                if magnebot_id in self.raycast_request:
                    item_info = all_magnebots[all_idx].item_info
                    self.raycast_request.remove(magnebot_id)
                
                
                if not self.local:
                    self.sio.emit('human_output', (all_idx, all_magnebots[idx].dynamic.transform.position.tolist(), item_info, all_magnebots[all_idx].company, self.timer))
                    
                idx += 1

            
            for m_idx, magnebot_id in enumerate(self.ai_magnebots_ids):
                if cams and magnebot_id in magnebot_images:
                    cams[idx+1].send(magnebot_images[magnebot_id])

                #Occupancy maps

                extra_status = [0]*3
                
                if magnebot_id in self.occupancy_map_request:
                     extra_status[0] = 1
               
                limited_map, reduced_metadata = self.get_occupancy_map(magnebot_id)
                
                
                if magnebot_id in self.objects_held_status_request:
                    extra_status[1] = 1
                
                objects_held = self.get_objects_held_state(magnebot_id)
                
                
                
                if magnebot_id in self.danger_sensor_request:
                    _,item_info = self.danger_sensor_reading(magnebot_id)
                    self.danger_sensor_request.remove(magnebot_id)
                    extra_status[2] = 1
                else:
                    item_info = {}
                    
                    
                
                    
                all_idx = all_ids.index(str(magnebot_id))
                
                
                #if all_idx in self.ai_status_request:
                ai_status = all_magnebots[all_idx].past_status.value
                    
                if not self.local:
                    self.sio.emit('ai_output', (all_idx, json_numpy.dumps(limited_map), reduced_metadata, objects_held, item_info, ai_status, extra_status, all_magnebots[all_idx].strength, self.timer) )

                idx += 1
                
                
            
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
                self.timer -= time.time() - time_gone
                time_gone = time.time()


            #Reset world
            if self.reset:
                print("Resetting...")
                
                #resp = self.communicate({"$type": "destroy_all_objects"})
                self.reset_world()
                
                for am in all_magnebots:
                    self.sio.emit("agent_reset", str(am.robot_id))
                
                self.reset = False
                #pdb.set_trace()
                print("Reset complete")
                
   
            self.frame_num +=1
            
        self.communicate({"$type": "terminate"})
        
        return 0


    def reset_world(self):
        
        commands = []
        
        for go in self.graspable_objects:
            commands.append({"$type": "destroy_object", "id": go})
            
        commands.append({"$type": "destroy_avatar", "avatar_id": 'a'})   
        self.communicate(commands)
        
        commands = []
        
        for u_idx in range(len(self.user_magnebots)):
            self.user_magnebots[u_idx].reset(position=self.user_spawn_positions[u_idx])
            #self.user_magnebots[u_idx].ui.initialized = False
            
        for u_idx in range(len(self.ai_magnebots)):
            self.ai_magnebots[u_idx].reset(position=self.ai_spawn_positions[u_idx])
        
        
        '''
        #Reset user magnebots
        for u_idx in range(len(self.user_magnebots)):
                print("Destroying: ", u_idx)
                self.user_magnebots[u_idx].ui.destroy_all(destroy_canvas=True)
                self.communicate([])
        for u_idx in range(len(self.user_magnebots)):    
                print("Destroying: ", self.user_magnebots[u_idx].robot_id)
                self.communicate({"$type": "destroy_avatar", "id": str(self.user_magnebots[u_idx].robot_id)})
                self.communicate({"$type": "set_render_order", "render_order": 100, "sensor_name": "SensorContainer", "avatar_id": "a"})
        print("Fin")
        '''
                
        #Reset ai magnebots
        #for u_idx in range(len(self.ai_magnebots)):
        #        self.communicate({"$type": "destroy_avatar", "id": str(self.ai_magnebots[u_idx].robot_id)})
        # self.user_magnebots[u_idx].reset(position=user_spawn_positions[u_idx])
        #self.user_magnebots[0].reset(position=user_spawn_positions[0])
            
        #self.user_magnebots[1].ui.destroy_all(destroy_canvas=True)
        #self.communicate([])

        self.object_manager.initialized = False
        commands = []
        commands.extend(self.create_scene())
            
        commands.extend(self.populate_world())
        self.communicate(commands)

        #Reattach canvas
        for um in self.user_magnebots:
            um.ui.attach_canvas_to_avatar(avatar_id=str(um.robot_id))
        self.communicate([])
        
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='run locally only')
    parser.add_argument('--no_virtual_cameras', action='store_true', help='do not stream frames to virtual cameras')
    parser.add_argument('--address', type=str, default='https://172.17.15.69:4000' ,help='adress to connect to')
    parser.add_argument('--video-index', type=int, default=0 ,help='index of the first /dev/video device to start streaming to')
    parser.add_argument('--no-debug-camera', action='store_true', help='do not instantiate debug top down camera')
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    num_users = cfg['num_humans']
    num_ais = cfg['num_ais']
    
    width = cfg['width']
    height = cfg['height']
    
    global_refresh_sensor = cfg['sensor_waiting_time']

    #The web interface expects to get frames from camera devices. We simulate this by using v4l2loopback to create some virtual webcams to which we forward the frames generated in here
    if not args.no_virtual_cameras:
        for user in range(args.video_index,args.video_index+num_users+1): #One extra camera for the debug video
            cams.append(pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video'+str(user)))
        for ai in range(args.video_index+num_users+1,args.video_index+num_users+1+num_ais):
            cams.append(pyvirtualcam.Camera(width=width, height=height, fps=20, device='/dev/video'+str(ai)))

    address = args.address

    c = Simulation(args, cfg)

    result = c.run()
    print("Simulation ended with result: ", result)

    
