from enum import Enum
from magnebot import ActionStatus

class State(Enum):
    waiting = 1
    moving_to_objective = 2


class AI_Magnebot_Controller():
    def __init__(self,magnebot):
        self.magnebot = magnebot
        self.state = State.waiting

    def controller(self,object_manager,sio):

        if self.state == State.waiting:
            if self.magnebot.messages:
                message = self.magnebot.messages.pop(0)
                if "I need help with " in message[1]:
                    if "sensing" in message[1]:
                        pass
                    elif "lifting" in message[1]:
                        self.object_id = int(message[1][25:])
                        if sio:
                            sio.emit("message", ("I will help " + message[0], self.magnebot.company,str(self.magnebot.robot_id)))
                        self.state = State.moving_to_objective
                        
        elif self.state == State.moving_to_objective:
            
            if self.magnebot.action.status == ActionStatus.tipping:
                self.magnebot.reset_position()
            elif self.magnebot.action.status != ActionStatus.ongoing:
                #print("moving to objective")
                self.magnebot.move_to(target=self.object_id, arrived_offset=1)


