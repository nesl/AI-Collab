import cv2
import csv
import tkinter as tk
from PIL import Image, ImageTk
import re
import argparse
import tkinter.ttk as ttk
import pdb

parser = argparse.ArgumentParser(
    description="Review_Video"
)
parser.add_argument('--file', type=str, default='2025_07_10_16_21_36', help='Log file time')


args = parser.parse_args()

dtime = args.file

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

txt_log = open("text_log.txt", 'w')

log_state_file = open("../simulator/" + "log" + "/" + dtime + "_state.txt")





while False: #new_line:
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
            txt_log.write(mess+'\n')  
    
    elif int(split_line[1]) == 4:
        first_bracket = new_line.find('{')
        second_bracket = new_line.rfind('}')
        
        newer_line = new_line[first_bracket:second_bracket+1].replace("false", "False").replace("true","True")
        if newer_line:
            team_strategy = eval(newer_line)
            
            for h_key in team_strategy["hierarchy"].keys():
            
                if team_strategy["hierarchy"][h_key] == "order":
                    leader = h_key
                    break
                
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
        self.shown   = set()

        # --- helper to format seconds -> H:MM:SS.mmm ---
        def fmt_time(seconds: float) -> str:
            if seconds is None:
                return "00:00.000"
            ms_total = int(round(seconds * 1000))
            sec, ms  = divmod(ms_total, 1000)
            h, sec   = divmod(sec, 3600)
            m, s     = divmod(sec, 60)
            if h:
                return f"{sec:02d}.{ms:03d}"
            return f"{sec:02d}.{ms:03d}"

        self._fmt_time = fmt_time  # keep for reuse

        # Keep total duration for the right side of the label
        self.total_dur = frame_timestamps[-1] if frame_timestamps else 0.0

        # — Readout label (timestamp) —
        self.pos_lbl_var = tk.StringVar(value=f"{self._fmt_time(0.0)} / {self._fmt_time(self.total_dur)}")
        self.pos_lbl = tk.Label(root, textvariable=self.pos_lbl_var, font=("TkDefaultFont", 10))
        self.pos_lbl.pack(side="bottom", pady=(6, 0))

        # — Seek slider —
        self.seek = tk.Scale(
            root,
            from_=0,
            to=self.total_fr - 1,
            orient="horizontal",
            length=500,
            command=self.on_seek,
            showvalue=0,             # hide the default numeric value
            label="Position",        # optional: static caption above the slider
        )
        self.seek.pack(side="bottom", fill="x", padx=10, pady=(4, 0))

        # — Pause/Play button —
        self.play_btn = tk.Button(root, text="Pause", command=self.toggle_play)
        self.play_btn.pack(side="bottom", pady=(5, 15))

        # — Video display —
        self.video_lbl = tk.Label(root)
        self.video_lbl.pack(side="left", padx=5, pady=5)

        # — Log panel —
        log_frame = tk.Frame(root)
        log_frame.pack(side="right", padx=10, pady=10, fill="y")
        self.log_txt = tk.Text(log_frame, width=40, height=25, state="disabled", wrap="word", font=("Courier", 18))
        self.log_txt.pack(side="left", fill="both", expand=False)
        
        log_scroll = tk.Scrollbar(log_frame, orient="vertical", command=self.log_txt.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_txt.config(yscrollcommand=log_scroll.set)

        self.log_txt.tag_configure('highlight',  foreground='red')
        self.log_txt.tag_configure('highlight2', foreground='blue')
        self.log_txt.tag_configure('highlight3', foreground='green')
        self.log_txt.tag_configure('highlight4', foreground='yellow')
        
        self._install_copy_support()
        
        self.play_btn = tk.Button(root, text="Pause", command=self.toggle_play)
        self.play_btn.pack(side="bottom", pady=(5, 5))

        # --- Playback speed ---
        self.play_speed = tk.DoubleVar(value=1.0)  # 1.0x, 1.5x, 2.0x, etc.
        self._frame_accum = 0.0                    # fractional frame accumulator

        speed_frame = tk.Frame(root)
        speed_frame.pack(side="bottom", pady=(0, 10))

        tk.Label(speed_frame, text="Speed:").pack(side="left", padx=(0, 6))

        self.speed_box = ttk.Combobox(
            speed_frame,
            textvariable=self.play_speed,
            values=[0.5, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0],
            width=5,
            state="readonly"
        )
        self.speed_box.pack(side="left")
        self.speed_box.set("8.0")  # show 1.0 in the UI

        # — Playback timing —
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.delay = int(1000 / fps)

        # Kick off playback
        self._play()

    def _install_copy_support(self):
        # Ctrl+C / Cmd+C support
        def do_copy(event=None):
            try:
                sel = self.log_txt.get("sel.first", "sel.last")
            except tk.TclError:
                return "break"  # nothing selected
            self.root.clipboard_clear()
            self.root.clipboard_append(sel)
            return "break"
        self.log_txt.bind("<Control-c>", do_copy)
        self.log_txt.bind("<Control-C>", do_copy)
        self.log_txt.bind("<<Copy>>",    do_copy)

        # Right-click menu
        menu = tk.Menu(self.log_txt, tearoff=0)
        menu.add_command(label="Copy", command=do_copy)
        def show_menu(event):
            try:
                # Focus so shift/ctrl selection works reliably
                self.log_txt.focus_set()
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
        self.log_txt.bind("<Button-3>", show_menu)   # Windows/Linux
        self.log_txt.bind("<Button-2>", show_menu)   # macOS secondary click
        
    def _update_time_readout(self):
        # Guard for bounds
        idx = max(0, min(self.frame_idx, len(frame_timestamps) - 1))
        ts  = frame_timestamps[idx] if frame_timestamps else 0.0
        self.pos_lbl_var.set(f"{self._fmt_time(ts)} / {self._fmt_time(self.total_dur)}")

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
        self._update_time_readout()  # reflect drag position immediately

    def _play(self):
        if not self.playing:
            return

        # Draw current frame
        self._render_frame()
        self._update_time_readout()

        # How many frames to advance this tick?
        spd = float(self.play_speed.get())
        # Accumulate fractional frames
        self._frame_accum += spd
        advance = int(self._frame_accum)

        # Ensure we move at least 1 frame per tick (in case UI gives <1.0)
        if advance < 1:
            advance = 1

        # Consume the integer part, keep the remainder
        self._frame_accum -= advance

        # Step frames and update slider
        self.frame_idx = (self.frame_idx + advance) % self.total_fr
        self.seek.set(self.frame_idx)

        # Schedule next tick
        self.root.after(self.delay, self._play)

    def _rebuild_log_up_to(self, ts_now: float):
        entries = [log for log in logs if log["timestamp"] <= ts_now]
        # oldest → newest
        entries.sort(key=lambda L: L["timestamp"])

        self.log_txt.config(state="normal")
        first, _ = self.log_txt.yview()
        self.log_txt.delete("1.0", tk.END)

        for log in entries:
            msg = log["message"] + "\n"
            
            #msg_splitted = msg.split(" ")
            tag = None
            if pattern_search in log["message"]:
                tag = "highlight"
            elif " " + leader + ": " in log["message"]:
                tag = "highlight2"
            elif "move_straight" in log["message"]:
                tag = "highlight3"
            elif "My goal is location (2.5,3.5), I'm moving towards (-8.5,-2.5). My current location is (-8.5,-1.5). Helping B." in log["message"]:
                tag = "highlight4"

            # Prepend so newest ends up at the very top
            if tag:
                self.log_txt.insert("1.0", msg, tag)
            else:
                bold_part, rest = msg.split(" ", 1)
                self.log_txt.insert("1.0", rest + "\n")               # normal
                self.log_txt.insert("1.0", bold_part + " ", "bold") 
                #self.log_txt.insert("1.0", msg)

        if abs(first - 0.0) < 1e-6:
            self.log_txt.see("1.0")
        else:
            self.log_txt.yview_moveto(first)
        self.log_txt.config(state="disabled")
        
    def _render_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale = 2.5   # adjust as needed (e.g., 2.0 for double size)
        h, w = frame_rgb.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        frame_rgb = cv2.resize(frame_rgb, new_size, interpolation=cv2.INTER_LINEAR)

        
        img        = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.photo = img
        self.video_lbl.config(image=img)

        ts = frame_timestamps[self.frame_idx] if frame_timestamps else 0.0

        self._rebuild_log_up_to(ts)
        
        # Collect messages that "fire" in this time window
        new_msgs = []
        for log in logs:
            if ts <= log["timestamp"] < ts + POPUP_DURATION:
                key = (log["timestamp"], log["message"])
                if key not in self.shown:
                    new_msgs.append(log)

        if not new_msgs:
            return

        # Insert newest first at top; if your logs aren't time-sorted, sort them
        new_msgs.sort(key=lambda L: L["timestamp"], reverse=True)

        # Preserve user's scroll position: if they are at the very top (first==0.0),
        # keep them at the top after we prepend; otherwise, don't jump the view.
        self.log_txt.config(state="normal")
        first, last = self.log_txt.yview()

        for log in new_msgs:
            msg = log["message"] + "\n"
            #tlabel = f"[{self._fmt_time(log['timestamp'])}] "
            #msg = tlabel + log["message"] + "\n"
            # Choose a tag
            tag = None
            if pattern_search in log["message"]:
                tag = "highlight"
            elif " " + leader + ": " in log["message"]:
                tag = "highlight2"
            elif "move_straight" in log["message"]:
                tag = "highlight3"
            elif "My goal is location (2.5,3.5), I'm moving towards (-8.5,-2.5). My current location is (-8.5,-1.5). Helping B." in log["message"]:
                tag = "highlight4"

            # Prepend at the very top
            if tag:
                self.log_txt.insert("1.0", msg, tag)
            else:
                self.log_txt.insert("1.0", msg)

            # Mark as shown
            self.shown.add((log["timestamp"], log["message"]))

        # If user was at top, keep them at top after prepending; otherwise leave them where they scrolled.
        if abs(first - 0.0) < 1e-6:
            self.log_txt.see("1.0")

        self.log_txt.config(state="disabled")



if __name__ == "__main__":
    root = tk.Tk()
    root.title(dtime)
    VideoLogApp(root)
    root.mainloop()
