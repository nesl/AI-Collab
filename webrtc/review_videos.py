import cv2
import csv
import tkinter as tk
from PIL import Image, ImageTk
import re

dtime = "2025_07_10_16_21_36"#"2025_07_10_16_48_09"

# —————— Configuration ——————
VIDEO_PATH         = "../simulator/videos/" + dtime + "_0.mp4"
FRAME_TS_PATH      = "../simulator/videos/" + dtime + "_meta.txt"   # one real‐time timestamp (float seconds) per line
LOG_CSV_PATH       = "../ai_controller/log/" + dtime + "_C_monitor.txt" #"log/" + dtime + ".txt"               # CSV with "timestamp","message" columns
LOG_CSV_PATH2      = "log/" + dtime + ".txt"
POPUP_DURATION     = 5.0                      # seconds each log stays visible

# —————— Load your mapping from frame index → real timestamp ——————
with open(FRAME_TS_PATH, "r") as f:
    frame_timestamps = [float(line.strip()) for line in f]

# —————— Load your logs ——————
logs = []

        
log_file = open(LOG_CSV_PATH)
new_line = log_file.readline()
while new_line:
    split_line = new_line.strip().split(',')
    agent_id = split_line[2]
    '''
    if int(split_line[1]) == 2:
                
        first_bracket = new_line.find('{')
        second_bracket = new_line.rfind('}')
        
        
        messages = eval(new_line[first_bracket:second_bracket+1])
        
        regex_str = "My goal is (\w+) (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object (\w+).)?( Helping (\w+).)?"
                    
        rematch = re.search(regex_str,messages["whole"])
        if rematch:
            messages["whole"] = messages["whole"].replace(rematch.group(0), "").strip()
        
        if messages["whole"]:
            mess = split_line[0] + " " + split_line[2] + ": " + messages["whole"]
            logs.append({
                "timestamp": float(split_line[0]),
                "message":   mess
            })
    '''
    
    if int(split_line[1]) == 0:
         
        agent_reliability = {}      
        first_bracket = new_line.find('{')
        second_bracket = new_line.rfind('}')

        info = eval(new_line[first_bracket:second_bracket+1])
        info["agent_id"] = agent_id
        if agent_id not in agent_reliability.keys():
            agent_reliability[agent_id] = info

        else:
            changed_reliability = []
            for i in info.keys():
            
                if agent_reliability[agent_id][i] != info[i]:
                    changed_reliability.append((i,info[i]))
                    
            #print("Changed reliability for agent " + agent_id + " at time " + str(split_line[0]), changed_reliability)
            agent_reliability[agent_id] = info
            
        logs.append({
            "timestamp": float(split_line[0]),
            "message":   str(info)
        })
    elif int(split_line[1]) == 1:
                    
        first_bracket = new_line.find('{')
        second_bracket = new_line.rfind('}')
        
        agent_model = {}

        info = eval(new_line[first_bracket:second_bracket+1])
        info["agent_id"] = agent_id
        if agent_id not in agent_model.keys():
            agent_model[agent_id] = info
            pass
            #print("Model for agent " + agent_id + " at time " + str(split_line[0]), info)
        else:
            changed_model = []
            for i in info.keys():
                if agent_model[agent_id][i] != info[i]:
                    changed_model.append((i,info[i]))
                    
            #print("Changed model for agent " + agent_id + " at time " + str(split_line[0]), changed_model)
            agent_model[agent_id] = info
            
        logs.append({
            "timestamp": float(split_line[0]),
            "message":   str(info)
        })
   
    new_line = log_file.readline()
    

log_file = open(LOG_CSV_PATH2)
new_line = log_file.readline()
while new_line:
    split_line = new_line.strip().split(',')
    
    
    if int(split_line[1]) == 2:
                
        first_bracket = new_line.find('{')
        second_bracket = new_line.rfind('}')
        
        
        messages = eval(new_line[first_bracket:second_bracket+1])
        
        regex_str = "My goal is (\w+) (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object (\w+).)?( Helping (\w+).)?"
                    
        rematch = re.search(regex_str,messages["whole"])
        if rematch:
            messages["whole"] = messages["whole"].replace(rematch.group(0), "").strip()
            pass
        
        if messages["whole"]:
            mess = split_line[0] + " " + split_line[2] + ": " + messages["whole"]
            logs.append({
                "timestamp": float(split_line[0]),
                "message":   mess
            })       
    new_line = log_file.readline()
    
    
# sort by timestamp to make lookups easier
logs.sort(key=lambda x: x["timestamp"])

pattern_search = "You are steering away"

class VideoLogApp:
    def __init__(self, root):
        self.root      = root
        self.cap       = cv2.VideoCapture(VIDEO_PATH)
        self.total_fr  = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.photo     = None
        self.active    = []
        self.playing   = True

        # — Seek slider —
        self.seek = tk.Scale(
            root,
            from_=0,
            to=self.total_fr-1,
            orient="horizontal",
            length=500,
            command=self.on_seek
        )
        self.seek.pack(side="bottom", fill="x", padx=10, pady=(10, 0))

        # — Pause/Play button —
        self.play_btn = tk.Button(root, text="Pause", command=self.toggle_play)
        self.play_btn.pack(side="bottom", pady=(5, 15))

        # — Video display —
        self.video_lbl = tk.Label(root)
        self.video_lbl.pack(side="left", padx=5, pady=5)

        # — Log panel —
        self.log_txt = tk.Text(root, width=40, height=25, state="disabled", wrap="word")
        self.log_txt.pack(side="right", padx=10, pady=10)
        
        self.log_txt.tag_configure('highlight', foreground='red')
        self.log_txt.tag_configure('highlight2', foreground='blue')
        self.log_txt.tag_configure('highlight3', foreground='green')
        self.log_txt.tag_configure('highlight4', foreground='yellow')

        # — Playback timing —
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.delay = int(1000 / fps)

        # Kick off playback
        self._play()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play()

    def on_seek(self, val):
        idx = int(val)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self.frame_idx = idx
        self.active.clear()
        self._render_frame()

    def _play(self):
        if not self.playing:
            return
        self._render_frame()
        self.frame_idx = (self.frame_idx + 1) % self.total_fr
        self.seek.set(self.frame_idx)
        self.root.after(self.delay, self._play)

    def _render_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img        = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.photo = img
        self.video_lbl.config(image=img)

        ts = frame_timestamps[self.frame_idx]
        for log in logs:
            possible_log = {
                    "msg":        log["message"],
                    "expires_at": log["timestamp"] + POPUP_DURATION
                }
            if ts <= log["timestamp"] < ts + POPUP_DURATION and possible_log not in self.active:
                self.active.append(possible_log)
        self.active = [L for L in self.active if L["expires_at"] >= ts]

        self.log_txt.config(state="normal")
        self.log_txt.delete("1.0", tk.END)
        for L in self.active:
        
            msg = L["msg"] + "\n"    
            if pattern_search in L["msg"]:
                self.log_txt.insert(tk.END, msg, "highlight")
            elif "carry_heavy" in L["msg"]:
                self.log_txt.insert(tk.END, msg, "highlight2") 
            elif "move_straight" in L["msg"]:
                self.log_txt.insert(tk.END, msg, "highlight3")
            elif "My goal is location (2.5,3.5), I'm moving towards (-8.5,-2.5). My current location is (-8.5,-1.5). Helping B." in L["msg"]:
                self.log_txt.insert(tk.END, msg, "highlight4")
            else:
                self.log_txt.insert(tk.END, msg)
        self.log_txt.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    root.title(dtime)
    VideoLogApp(root)
    root.mainloop()
