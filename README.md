# AI-Collaboration Simulator

This simulator builds upon [ThreeDWorld](https://github.com/threedworld-mit/tdw) (TDW), a platform for interactive multi-modal physical simulation. This simulator as of now allows multiple human users to control the agents present in a single scene in a concurrent manner. It also incorporates an HTTP server to which users can connect to remotely control the agents.

## Setup

Use `git clone --recurse-submodules https://github.com/nesl/AI-Collab.git` to clone the repository with all the submodules.

### TDW Simulator
    
1. Create an environment with python == 3.7.0

2. Run `pip install -r requirements.txt`

3. Change to the **magnebot** directory and run `pip install .`

After this, you will be able to run the simulation by going into the **simulator** directory and using `python simulation.py --local --no_virtual_cameras`. This will display a simulator window with the third person view camera, as well as an opencv window with a first person view of one of the robots. You can control this robot by focusing on the simulator window and using the arrows in the keyboard. Check the file **keysets.csv** for all the keys one can use for each robot.

### Virtual Video Devices

In order to allow us to stream the generated videos to the respective users through WebRTC, we need to create virtual video devices to which we send the generated frames and from which the HTTP server gets the frames as streams.

Follow the steps in [https://github.com/umlaeute/v4l2loopback](https://github.com/umlaeute/v4l2loopback) to build the v4l2loopback module needed to simulate these virtual video interfaces and then just use the next command: `modprobe v4l2loopback devices=4`, where the devices parameter can be changed to create as many virtual devices as you want (here it is 4). Be sure to use one of the tagged versions for *v4l2loopback* (0.12.7 in our case).

After this you will now be able to run the simulator using the next command: `python simulation.py --local`, which shouldn't present be any different as when using the **--no_virtual_cameras** option.

### Web Interface

Our web interface uses Node.js, as well as WebRTC and Socket.io

Install:

- nodejs 16.17.0
- npm 8.15.0

Change to the **webrtc** directory and issue the next command: `npm install`.
Before running the server, you will need to create a key and a self-signed certificate to enable HTTPS. To do this, just run the next command: `openssl req -nodes -new -x509 -keyout server.key -out server.cert`. It will ask a series of questions, ignore them, only when asking **Common Name** put **localhost** and use your email address when asked for it.

Be sure to change the address in **server.js** before running the server.

The implementation of the WebRTC server was based on [https://github.com/TannerGabriel/WebRTC-Video-Broadcast](https://github.com/TannerGabriel/WebRTC-Video-Broadcast)


### AI Controller

Change to the **ai_controller** directory and install the gym environment by using the next command `pip install -e gym_collab`

## Operation

1. Run the server using `node server --address "address" --port "port"`. The simulator assumes the virtual devices to be used are the ones starting at /dev/video0, but if you already have some real webcams, you need to specify the parameter `--video-index <number>` and include the index number of your first simulated webcam corresponding to the ones created for the simulator.
2. Run the simulator using `python simulation.py --address "https://address:port"`
3. Using your web browser, go to **https://address:port/broadcast.html**. This will present a view with all the camera views being streamed.
4. When you run the first command, there will be an output indicating a code that you need to use as password when connecting through the browser.

### User Control of a Robot 

1. Using your web browser in the same or a different computer, go to **https://address:port/?client=1**, where the client parameter controls which robot you get assigned. This parameter goes from 1 to the number of user controllable robots you have in the simulation.

### AI Control of a Robot

1. Change to the **ai_controller** directory and run the **server_command** script. You have to also create a new certificate + key as this script executes an HTTPS server to setup the WebRTC parameters. Inside the **server_command**, specify the certificate, key and host address associated with this server, as well as the address to connect to.

#### Note

To make the HTTPS self-signed certificate work:
1. Run **server_command**
2. Access through the web browser to the address provided by the HTTPS server and accept the certificate
3. Try again running **server_command** and it should work!

## AI Controller

The **ai_controller.py** program uses an HTTPS server to negotiate the WebRTC parameters. Socket.IO is used for normal commmunication with the simulator server. The controller uses the same API functions defined in the [Magnebot repository](https://github.com/alters-mit/magnebot/blob/main/doc/manual/magnebot/actions.md). To receive occupancy maps of a certain view radius instead of camera images, you can run the **ai_controller.py** program as `python ai_controller.py --use-occupancy --view-radius <number>`, this way you don't need to make use of the HTTPS server.

### Action Space

The action space consists of the next fields:
1. *"action"* - argument: number of action to be executed. There are two types of actions that can be executed concurrently (issue one action while the other is completing), but actions of the same type cannot be executed this way, only sequentially. If you try executing an action that is of the same type as another before this last one is completed, your new action will be ignored. Actions may take different amount of steps. You can always execute the wait action (wait = 26). The two types of actions are the next ones:
	1. Locomotion/Actuation
		* move_up = 0
		* move_down = 1
		* move_left = 2
		* move_right = 3
		* move_up_right = 4
		* move_up_left = 5
		* move_down_right = 6
		* move_down_left = 7
		* grab_up = 8
		* grab_right = 9
		* grab_down = 10
		* grab_left = 11
		* grab_up_right = 12
		* grab_up_left = 13
		* grab_down_right = 14
		* grab_down_left = 15
		* drop_object = 16
	2. Sensing/Communication
		
		* danger_sensing = 17
		* get_occupancy_map = 18
		* get_objects_held = 19
		* check_item = 20
		* check_robot = 21
		* get_messages = 22
		* send_message = 23
		* request_item_info = 24
		* request_agent_info = 25


2. *"item"* - argument: index of the object to be checked (useful for action = 20, check_item). The robot environment saves the object information collected so far, but to actually get the entries of any of these objects, you should specify the item number and execute the corresponding action. You can get the number of objects known so far by checking the corresponding observation output.
3. *"robot"* - argument: index of robot to be checked (useful for action = 21, check_robot). The robot environment saves information about other robots and you can get their information by specifying the index of the robot you want.
4. *"message"* - argument: text message (usefule for action = 23, send_message). If the action is to send a message, this is where to put the message. Use the *"robot"* field to specify the index of the robot you want to receive the message, use 0 if you want everyone to get the message.

Notes: action = 17 (danger_sensing), gets an estimation of the danger level of neighboring objects and updates the necessary information. To actually display this information you need to issue the action = 20 (check_item). Action = 24 (request_item_info) and action = 25 (request_agent_info) are special types of messages that get sent to other robots to get the respective information, with the difference that any information they receive updates the internal information they have over specific objects and robots. They are used by specifying the robot index to which they want to request information, 0 if everyone.

### Observation Space

The observation space consists of the next fields:
1. *"frame"* - an nxm map, its actual dimensions determined by simulator parameters. This output shows at every step the location of the robot with respect to the grid world coordinates (index 5 in our occupancy map as shown in the Occupancy Maps section), nothing else. When taking action = 18 (get_occupancy_map), you will receive the entire occupacy map as part of this field's output. Remember that the occupancy map will be limited by the field of view, which is a configurable parameter.
2. *"objects_held"* - boolean value. Whether the robot is carrying an object in any of its arms.
3. *"action_status"* - list of binary values of size 4. A positive value will mean the following accoring to its position in the list: 
	* 1: a locomotion/actuation action has completed
	* 2: a locomotion/actuation action failed to complete correctly
	* 3: a sensing/communication action has completed
	* 4: a sensing/communication action failed to complete correctly

4. *"item_output"* - dictionary that contains the information requested when using action = 20 (check_item). In it the next fields are present: *"item_weight"*, *"item_danger_level"* (0 if unknown) and *"item_location"* (grid location as an x,y point represented with a list).
5. *"num_items"* - number of items discovered so far.
6. *"neighbors_output"* - dictionary that contains the information requested when using action = 21 (check_robot). In it the next fields are present: *"neighbor_type"* (0 for human, 1 for AI), *"neighbor_location"* (same format as "item_location").
7. *"strength"* - current strength.
8. *"num_messages"* - number of messages in the receiving queue. To get all messages, use action = 22 (get_messages), which will return the messages as part of the **info** output of the step function.



### Occupancy Maps

For occupancy maps, the map is divided into cells of the size defined in **simulator/config.yaml**. The parameter *view_radius* specifies how many of these cells will conform the current view around the magnebot being controlled. The next values conform the occupancy map:

* -2: Unknown
* -1: Map boundaries
* 0: No obstacle present
* 1: Ambient obstacle (wall)
* 2: Manipulable object
* 3: Magnebot
* 4: Object being held by a magnebot
* 5: Magnebot being controlled


```
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 2 0 3 0 0 0 0 0 0 0 0]
 [0 5 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
 ```


## User Interface

The web interface consists of the camera views assigned to you robot, and a sidebar chat. This chat allows you to communicate with nearby robots, and to get information about your neighbors and scanned objects.

To control the robot through the web interface, you need to first click in the video area and then you can use one of the next keyboard commands:

* Arrows: To move the robot
* Z: To grab a focused object or drop it with the left arm
* X: To grab a focused object or drop it with the right arm
* C: To move the camera downwards
* V: To move the camera upwards
* B: To danger sense around you
* N: To focus on an object (this also gives you information about it)

![Interface](interface.png)



