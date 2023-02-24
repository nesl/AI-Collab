import numpy as np
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from magnebot import Magnebot, ImageFrequency
from magnebot.util import get_default_post_processing_commands
from tdw.add_ons.first_person_avatar import FirstPersonAvatar
import pdb
#import cupy as cp
from tdw.quaternion_utils import QuaternionUtils

width = 256
height = 256


class ChaseBall(Controller):


    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True):
        super().__init__(port=port, check_version=check_version, launch_build=launch_build)



        # Add the robot, the Magnebot, and an object manager.

        
        self.av = FirstPersonAvatar(position={"x": -1.4, "y": 0, "z": -3.1})

        self.magnebot: Magnebot = Magnebot(robot_id=self.get_unique_id(),   position={"x": -1.4, "y": 0, "z": -1.1},
                                           image_frequency=ImageFrequency.never)

        self.add_ons.extend([self.magnebot])
        #self.add_ons.extend([self.av, self.magnebot])
        #self.add_ons.extend([self.object_manager, self.camera, image_capture])

        # Create the scene.

        commands = [#self.get_add_scene(scene_name="empty_room"), 
                    TDWUtils.create_empty_room(20, 20),
                    
                    {"$type": "set_screen_size",
                     "width": width, #640,
                     "height": height}, #480},
                    {"$type": "create_interior_walls", "walls": [{"x": 6, "y": 1}, {"x": 6, "y": 2},{"x": 6, "y": 3},{"x": 6, "y": 4},{"x": 6, "y": 5},{"x": 1, "y": 6},{"x": 2, "y": 6},{"x": 3, "y": 6},{"x": 4, "y": 6},{"x": 5, "y": 6}]},
                    {"$type": "create_interior_walls", "walls": [{"x": 14, "y": 1}, {"x": 14, "y": 2},{"x": 14, "y": 3},{"x": 14, "y": 4},{"x": 14, "y": 5},{"x": 19, "y": 6},{"x": 18, "y": 6},{"x": 17, "y": 6},{"x": 16, "y": 6},{"x": 15, "y": 6}]},   
                    {"$type": "create_interior_walls", "walls": [{"x": 6, "y": 19}, {"x": 6, "y": 18},{"x": 6, "y": 17},{"x": 6, "y": 16},{"x": 6, "y": 15},{"x": 1, "y": 14},{"x": 2, "y": 14},{"x": 3, "y": 14},{"x": 4, "y": 14},{"x": 5, "y": 14}]},
                    {"$type": "create_interior_walls", "walls": [{"x": 14, "y": 19}, {"x": 14, "y": 18},{"x": 14, "y": 17},{"x": 14, "y": 16},{"x": 14, "y": 15},{"x": 19, "y": 14},{"x": 18, "y": 14},{"x": 17, "y": 14},{"x": 16, "y": 14},{"x": 15, "y": 14}]}
                    ]


        '''
        max_coord = 8
        object_models = ['iron_box','4ft_shelf_metal','trunck','lg_table_marble_green','b04_backpack','36_in_wall_cabinet_wood_beach_honey']
        coords = {}
        
        coords[object_models[0]] = [[max_coord,max_coord],[max_coord-1,max_coord-0.1],[max_coord-0.5,max_coord-0.2],[max_coord-0.4,max_coord],[max_coord,max_coord-0.5]]
        coords[object_models[1]] = [[max_coord-3,max_coord]]

        coords[object_models[2]] = [[max_coord,max_coord-3]]
        coords[object_models[3]] = [[max_coord-2,max_coord-2]]
        coords[object_models[4]] = [[max_coord-1,max_coord-2]]
        coords[object_models[5]] = [[10-6,10-6]]

        modifications = [[1.0,1.0],[-1.0,1.0],[1.0,-1.0],[-1.0,-1.0]]

        final_coords = {}

        for objm in object_models:
            final_coords[objm] = []
        

        for fc in final_coords.keys():
            for m in modifications:
                final_coords[fc].extend(np.array(coords[fc])*m)

        for fc in final_coords.keys():
            for c in final_coords[fc]:

                commands.extend(self.get_add_physics_object(model_name=fc,
                                                 object_id=self.get_unique_id(),
                                                 position={"x": c[0], "y": 0, "z": c[1]},
                                                 rotation={"x": 0, "y": 0, "z": 0},
                                                 default_physics_values=False,
                                                 mass=10,
                                                 scale_mass=False))



        '''

        #self.communicate(self.get_add_scene(scene_name="tdw_room"))
        #commands = []
        # Add post-processing.
        commands.extend(get_default_post_processing_commands())
        

        self.communicate(commands)


    


    
    def run(self):

        commands = []
      
        timer = 100
        print(self.magnebot.robot_id)
        
        while True:

           
            resp = self.communicate(commands)
            commands.clear()
            #pdb.set_trace()
            #print(self.av.transform.position,QuaternionUtils.quaternion_to_euler_angles(self.av.transform.rotation))
            all_images = []
            
            if not timer:
                print("Destroying robot")
                #commands.append({"$type": "destroy_robot", "id": self.magnebot.robot_id})
                self.magnebot.reset()

                self.communicate([TDWUtils.create_empty_room(20, 20)])
                print("finish")
                timer -= 1
            elif timer > 0:
                timer -= 1




        self.communicate({"$type": "terminate"})


    


if __name__ == "__main__":
    c = ChaseBall()
    print('hello')
    c.run()
