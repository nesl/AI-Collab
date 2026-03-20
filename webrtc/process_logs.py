import glob
import pdb
import json
import numpy as np
from sklearn.cluster import DBSCAN
import re
import pandas as pd
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import shutil
from matplotlib.patches import Patch
from collections import Counter
import heapq
import math
import itertools
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import statsmodels.api as sm
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import defaultdict, OrderedDict
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import textwrap
from itertools import chain, combinations
from matplotlib.patches import Rectangle
from typing import Dict, Any, Iterable, Tuple, List, Optional, Hashable, Set
from scipy.stats import kruskal

#import shap

pd.set_option("display.max_columns", None)   # show all columns
pd.set_option("display.max_rows", None)      # show all rows

total_sessions = 0
communication_distance = 5
good_sessions = {}
good_names = {}


preamble = """Analyze the content of each message and try to tell me who are they answering to? Some messages may be directed towards a subset of the group. Output a JSON format with the following fields for each analyzed message: "number" and "reply_to", where "number" is the number of the analyzed message, and "reply_to" is any of the set {"Agent A", "Agent B", "Agent C", "Agent D", "Everyone"}.

For example, if I provide you with the next log of messages:

{Number: 1 | Sender: Agent D | Message: Hi everyone}
{Number: 2 | Sender: Agent A | Message: hi}
{Number: 3 | Sender: Agent C | Message: Hi}
{Number: 4 | Sender: Agent C | Message: I ll follow A}
{Number: 5 | Sender: Agent A | Message: ok}
{Number: 6 | Sender: Agent D | Message: Sounds good, lets each scan and establish which objects are dangerous}

[{"number": 1, "reply_to": "Everyone"},{"number": 2, "reply_to": "Agent D"},{"number": 3, "reply_to": "Agent D"},{"number": 4, "reply_to": "Agent A"},{"number": 5, "reply_to": "Agent C"},{"number": 6, "reply_to": "Agent A, Agent C"}]"""

#print(glob.glob("*.txt"))

session_log = []
json_session = {}
different_agents = []
conversations = {0: [[]]}
history_conversation_groups = {0: [[]]}
messages_present = False
distances = {}
conversation_groups = {}
last_time = {0: [[]]}
all_messages = {"messages":[],"agents":[], "preambles":[], "conversation_groups": []}
scenario_options = None
settings = {}

csvfile = open("messages.csv", "w")
csvwriter = csv.writer(csvfile)
csvwriter.writerow(["Name","Time","Sender","Message","Leader"])


def _as_list2(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [x for x in v if isinstance(x, str)]
    return [v] if isinstance(v, str) else []

def find_and_remove_carry_overlaps(
    orders_to_agent: Dict[str, Any],
    current_orders: Dict[str, Any]
) -> List[Tuple[str, str]]:
    """
    Find overlaps where:
      - agent is in current_orders but not in orders_to_agent
      - the agent has a 'carry_*' that also appears in orders_to_agent values
    
    Returns list of (carry_str, agent_id).
    Also mutates current_orders: removes those carry_str occurrences from the agent.
    """
    ota_keys = set(orders_to_agent.keys())
    cur_keys = set(current_orders.keys())
    missing_in_ota = cur_keys - ota_keys

    overlaps: List[Tuple[str, str]] = []

    for agent in missing_in_ota:
        vals_list = _as_list(current_orders.get(agent))
        new_vals = []
        for o in vals_list:
            if o.startswith("carry_"):
                # check if this carry order exists in any orders_to_agent values
                if any(o in _as_list(v) for v in orders_to_agent.values()):
                    overlaps.append((o, agent))
                    # skip adding it back to new_vals (i.e., remove from current_orders)
                    continue
            new_vals.append(o)

        if new_vals:
            current_orders[agent] = new_vals
        else:
            # if no orders left, drop the agent entirely
            current_orders.pop(agent, None)

    return overlaps

def dispersion_scalar(agents):
    positions = np.array([convert_to_grid_coordinates(list(a["last_position"])) for a in agents.values()])
    if len(positions) < 2:
        return 0.0  # no dispersion with 0 or 1 agent

    dists = [len(findPath(positions[i], positions[j], occ_map))
             for i, j in combinations(range(len(positions)), 2)]
    return float(np.mean(dists))

def dispersion_to_leader(agents, leader_id):
    """
    agents: dict like {'A': {'last_position': np.array([x,y])}, ...}
    leader_id: str (e.g., 'A', 'B', 'C', 'D')
    partner_map: dict mapping leader -> list of partners to compare against
                 e.g., {'C': ['A','B'], 'A': ['C','D'], ...}

    Returns a single scalar: mean distance from leader to listed partners.
    """
    
    partner_map = {
        "C": ["A", "B"],
        "A": ["C", "D"],
        # add more if you want:
        # "B": ["A", "D"],
        # "D": ["A", "B"],
    }
    

    if leader_id not in agents or "last_position" not in agents[leader_id]:
        return np.nan

    leader_pos = convert_to_grid_coordinates(list(agents[leader_id]["last_position"]))
    partners = partner_map.get(leader_id, [])
    dists = []
    for pid in partners:
        pinfo = agents.get(pid)
        if pinfo is not None and "last_position" in pinfo:
            dists.append(float(len(findPath(leader_pos, convert_to_grid_coordinates(list(pinfo["last_position"])), occ_map))))

    if len(dists) == 0:
        return np.nan
    return float(np.mean(dists))
    
def overall_mean_distance(team_tracker):
    all_dists = [
        e["mean_dist"]
        for entries in team_tracker.distance_log.values()
        for e in entries
    ]
    if not all_dists:
        return float("nan")  # or 0.0 if you prefer
    return float(np.mean(all_dists))
    
def _as_list(v):
    if v is None: return []
    return v if isinstance(v, list) else [v]

def _carry_obj(order: str) -> Optional[str]:
    return order.split("_", 1)[1] if isinstance(order, str) and order.startswith("carry_") else None

def _follow_target(order: str) -> Optional[str]:
    return order.split("_", 1)[1] if isinstance(order, str) and order.startswith("follow_") else None

def count_team_changes(team_tracker, include_conflicts: bool = True):
    """
    Returns {obj_id: num_changes}, where a 'change' is a transition between
    consecutive snapshots in the timeline. The first snapshot is not counted.
    If include_conflicts=False, ignores snapshots with leader=None.
    """
    counts = {}
    for obj_id, st in team_tracker.current_team.items():
        tl = st.get("timeline", [])
        if not include_conflicts:
            tl = [snap for snap in tl if snap["leader"] is not None]
        counts[obj_id] = max(0, len(tl) - 1)
    return counts

def average_team_composition_fraction_global(
    team_tracker,
    is_human: dict,
    object_weights: dict,
    time_weighted: bool = False,
):
    """
    Global average team composition (fractions) across all objects/snapshots,
    excluding objects with weight == 1. Weighting is by team size; optionally also by duration.

    Returns: {'avg_human_fraction': float, 'avg_ai_fraction': float}
    """
    total_humans = 0.0
    total_members = 0.0

    for obj_id, st in team_tracker.current_team.items():
        # Skip weight-1 objects
        #if object_weights.get(obj_id) == 1:
        #    continue

        tl = st.get("timeline", [])
        if not tl:
            continue

        for i, snap in enumerate(tl):
            # Team members in this snapshot
            members = set(snap["followers"])
            if snap["leader"] is not None:
                members.add(snap["leader"])
            if not members:
                continue

            # Weight = duration (if enabled); otherwise 1. Always multiplied by team size.
            if time_weighted and i < len(tl) - 1:
                duration = max(0.0, tl[i+1]["time"] - snap["time"])
            else:
                duration = 1.0

            size = len(members)
            humans = sum(1 for m in members if is_human.get(m, False))

            w = duration  # duration weight (size is accounted by counts below)
            total_humans  += humans * w
            total_members += size   * w

    if total_members == 0.0:
        #pdb.set_trace()
        return {"avg_human_fraction": 0.0, "avg_ai_fraction": 0.0}

    human_frac = total_humans / total_members
    return {"avg_human_fraction": human_frac, "avg_ai_fraction": 1.0 - human_frac}

class TeamTracker:
    """
    Refresh-based tracker of per-object team formations.

    Call `refresh(time_f, current_orders, orders_to_agent=None, agents=None, objects=None)` ONCE per timestamp,
    after you've applied all `current_orders[...]` updates for that tick (and while you still have `orders_to_agent`).

    Tracks:
      - Timeline of formation CHANGES from CURRENT STATE (normal or conflict).
      - Counts of ORDERED formations seen in `orders_to_agent` at each tick (including repeats).
      - For each `carry_X` seen in `orders_to_agent`, logs mean distance from ALL agents to object X.
    """

    def __init__(self):
        # State-based per-object formation timeline
        self.current_team = defaultdict(lambda: {
            "leader": None,             # Optional[str]
            "followers": set(),         # Set[str]
            "timeline": []              # List[{"time": float, "leader": Optional[str], "followers": Tuple[str,...]}]
        })

        # Followers queued for a target who isn't a unique leader yet (state-driven)
        self.pending_followers_by_leader: Dict[str, Set[str]] = defaultdict(set)

        # ORDERED formation counts (from orders_to_agent)
        # obj_id -> Counter[ (leader_or_None, followers_tuple) ] = int
        self._ordered_counts: Dict[str, Counter] = defaultdict(Counter)

        # Distance samples per object for each carry order seen in orders_to_agent
        # obj_id -> [ {"time": float, "carry": str, "mean_dist": float, "n_agents": int} ]
        self.distance_log: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ---------- internals ----------
    def _snapshot_if_changed(self, obj_id: str, t: float):
        st = self.current_team[obj_id]
        leader = st["leader"]  # may be None (conflict)
        followers = tuple(sorted(st["followers"]))
        tl = st["timeline"]
        if tl and tl[-1]["leader"] == leader and tl[-1]["followers"] == followers:
            return
        tl.append({"time": t, "leader": leader, "followers": followers})

    def _formation_from_orders(self, obj_id: str, orders_to_agent: Dict[str, Any]) -> Optional[Tuple[Optional[str], Tuple[str, ...]]]:
        """Build an ORDERED formation for obj_id from this tick's orders_to_agent."""
        carriers = {ag for ag, vals in orders_to_agent.items()
                    for o in _as_list(vals) if _carry_obj(o) == obj_id}
        if not carriers:
            return None
        if len(carriers) >= 2:
            return (None, tuple(sorted(carriers)))  # conflict ordered
        (leader,) = carriers
        followers = {ag for ag, vals in orders_to_agent.items()
                     for o in _as_list(vals) if _follow_target(o) == leader and ag != leader}
        return (leader, tuple(sorted(followers)))

    # ---------- public API ----------
    def refresh(self,
                time_f: float,
                current_orders: Dict[str, List[str]],
                orders_to_agent: Optional[Dict[str, Any]] = None,
                agents: Optional[Dict[str, Dict[str, Any]]] = None,
                objects: Optional[Dict[str, Dict[str, Any]]] = None):
        """Recompute formations from CURRENT STATE; optionally count ordered formations and log distances."""
        # 1) Build carriers_by_obj and followers_of from STATE (current_orders)
        carriers_by_obj: Dict[str, Set[str]] = defaultdict(set)
        for ag, orders in current_orders.items():
            for o in _as_list(orders):
                obj = _carry_obj(o)
                if obj is not None:
                    carriers_by_obj[obj].add(ag)

        followers_of: Dict[str, Set[str]] = defaultdict(set)
        for ag, orders in current_orders.items():
            for o in _as_list(orders):
                tgt = _follow_target(o)
                if tgt is not None and tgt != ag:
                    followers_of[tgt].add(ag)

        used_pending: Set[str] = set()

        # 2) Snapshot STATE formations (normal or conflict) when they change
        for obj_id, carriers in carriers_by_obj.items():
            st = self.current_team[obj_id]

            if len(carriers) >= 2:
                # Conflict state: leader=None, followers=carriers
                st["leader"] = None
                st["followers"] = set(carriers)
                self._snapshot_if_changed(obj_id, time_f)
                continue

            # Exactly one carrier -> normal state
            (leader,) = tuple(carriers)
            st["leader"] = leader
            pend = self.pending_followers_by_leader.get(leader, set())
            st["followers"] = (followers_of.get(leader, set()) | pend) - {leader}
            used_pending |= (st["followers"] & pend)
            self._snapshot_if_changed(obj_id, time_f)

        # 3) Maintain pending followers (state-driven)
        unique_leaders = {next(iter(cset)) for cset in carriers_by_obj.values() if len(cset) == 1}
        for tgt, pset in list(self.pending_followers_by_leader.items()):
            if tgt in unique_leaders:
                remaining = pset - used_pending
                if remaining:
                    self.pending_followers_by_leader[tgt] = remaining
                else:
                    self.pending_followers_by_leader.pop(tgt, None)
        for tgt, flws in followers_of.items():
            if tgt not in unique_leaders:
                self.pending_followers_by_leader[tgt].update(flws)

        # 4) If orders_to_agent present: count ORDERED formations and log distances per carry order
        if orders_to_agent is not None:
            # 4a) Count ordered formations per object (including repeats)
            objs_in_orders: Set[str] = { _carry_obj(o)
                                         for vals in orders_to_agent.values()
                                         for o in _as_list(vals)
                                         if _carry_obj(o) is not None }
            for obj_id in objs_in_orders:
                form = self._formation_from_orders(obj_id, orders_to_agent)
                if form is not None:
                    leader, followers = form
                    self._ordered_counts[obj_id][(leader, followers)] += 1

            # 4b) TEAM-ONLY distance logging (one entry per object per tick)
            if agents is not None and objects is not None:
                # carriers per object from ORDERS at this tick
                carriers_by_obj: Dict[str, Set[str]] = defaultdict(set)
                for ag, vals in orders_to_agent.items():
                    for o in _as_list(vals):
                        obj_id = _carry_obj(o)
                        if obj_id is not None:
                            carriers_by_obj[obj_id].add(ag)

                # followers per leader from ORDERS at this tick
                followers_of: Dict[str, Set[str]] = defaultdict(set)
                for ag, vals in orders_to_agent.items():
                    for o in _as_list(vals):
                        tgt = _follow_target(o)
                        if tgt is not None and tgt != ag:
                            followers_of[tgt].add(ag)

                # compute exactly one mean distance per object with carry orders
                for obj_id in objs_in_orders:
                    obj_meta = objects.get(obj_id)
                    if not obj_meta or obj_meta.get("last_position") is None:
                        continue
                    obj_pos = np.asarray(obj_meta["last_position"])
                    object_pos = convert_to_grid_coordinates(list(obj_pos))

                    carriers = carriers_by_obj.get(obj_id, set())

                    if len(carriers) == 0:
                        continue  # no team this tick
                    elif len(carriers) >= 2:
                        participants = set(carriers)  # conflict: team = all carriers
                    else:
                        (leader,) = tuple(carriers)
                        participants = {leader} | followers_of.get(leader, set())

                    # distances from TEAM MEMBERS ONLY
                    dists = []
                    for aid in participants:
                        pos = agents.get(aid, {}).get("last_position")
                        if pos is None:
                            continue
                        a = np.asarray(pos)
                        if a.shape != obj_pos.shape:
                            continue
                        #dists.append(float(np.linalg.norm(a - obj_pos)))
                        
                        a_pos = convert_to_grid_coordinates(list(pos))
                        dists.append(len(findPath(a_pos, object_pos, occ_map)))

                    if dists:
                        self.distance_log[obj_id].append({
                            "time": time_f,
                            "carry": f"carry_{obj_id}",
                            "mean_dist": float(np.mean(dists)),
                            "n_agents": len(dists),
                            "team": tuple(sorted(participants)),
                        })

    # ---------- accessors ----------
    def unique_team_configurations(self, include_conflicts: bool = True) -> Set[Tuple[str, Optional[str], Tuple[str, ...]]]:
        """
        Distinct (obj_id, leader_or_None, followers_tuple) observed in the STATE timeline.
        If include_conflicts=False, omits leader=None rows.
        """
        uniq = set()
        for obj_id, st in self.current_team.items():
            for snap in st["timeline"]:
                if not include_conflicts and snap["leader"] is None:
                    continue
                uniq.add((obj_id, snap["leader"], snap["followers"]))
        return uniq

    def ordered_formation_counts(self) -> Dict[str, Dict[Tuple[Optional[str], Tuple[str, ...]], int]]:
        """
        obj_id -> { (leader_or_None, followers_tuple) : count }
        Counts how many times each formation was explicitly ORDERED (from orders_to_agent), including repeats.
        """
        return {obj: dict(cnt) for obj, cnt in self._ordered_counts.items()}

    def get_distance_log(self, obj_id: Optional[str] = None):
        """Per-object distance samples logged for each carry order seen in orders_to_agent."""
        if obj_id is None:
            return dict(self.distance_log)
        return self.distance_log.get(obj_id, [])

orders_by_human_agent = {}

def process_human_orders_pd():
    orders_pd = pd.read_csv("orders.csv")
    orders_by_agent = {}
    for f in orders_pd['Name'].unique():
        extract_orders = orders_pd[(orders_pd['Name'] == f) & (orders_pd['Human leader'] == True)]
        total_orders = []
        for p in range(len(extract_orders)):
            
            function = extract_orders.iloc[p]['Function']
            try:
                if function[0] == '[':
                    functions = function[1:].split('),')
                else:
                    functions = [function]
            except:
                pdb.set_trace()
            new_order = {'A':[],'B':[],'C':[],'D':[]}
            for ff in functions:
                ff_split = ff.split(',',maxsplit=1)
                
                if ff == 'CARRY' or ff == 'SENSE':
                    continue
                
                if 'SENSE' in ff or 'SENSE_OBJECT' in ff or 'CARRY' in ff:
                    try:
                        arg = ff_split[0].split('(')[1]
                    except:
                        pdb.set_trace()
                    if arg.isdigit():
                        room_num = arg
                    elif arg == 'ALL':
                        room_num = 'ALL'
                    else:
                        continue
                        
                    arg2 = ff_split[1]
                    
                    if '[' not in arg2:
                        if arg2[0] == '?':
                            continue
                        agents = [arg2[0]]
                    else:
                        try:
                            end_list = arg2.index(']')
                            agents = arg2[1:end_list].split(',')
                        except:
                            pdb.set_trace()
                      
                    leader = ''  
                    if 'CARRY' in ff and ']' in arg2:
                        end_list = arg2.index(']')
                        if ',' in arg2[end_list:]:
                            leader = arg2[end_list:].split(',')[1]
                            leader = leader[0]
                        
                    
                    for ag in agents:
                        if 'SENSE_OBJECT' in ff and ag != "A":
                            new_order[ag].append('sense_' + room_num)
                        elif 'SENSE' in ff and ag != "A":
                            new_order[ag].append('sense_room' + room_num)
                        elif 'CARRY' in ff:
                            if not leader or leader == ag:
                                try:
                                    new_order[ag].append('carry_' + room_num)
                                except:
                                    pdb.set_trace()
                            else:
                                new_order[ag].append('follow_' + leader)
                                
                                
                elif 'FOLLOW' in ff:
                    arg = ff.split('(')[1]
                    if '[' not in arg:
                        if arg[0] == '?':
                            continue
                        agents = [arg[0]]
                    else:
                        end_list = arg.index(']')
                        agents = arg[1:end_list].split(',')   
                      
                    arg2 = ff.split(',')[-1]
                    
                    if arg2[0] == '?':
                        continue
                    leader = arg2[0]
                    
                    for ag in agents:
                        new_order[ag].append('follow_' + leader)
                    
            total_orders.append([extract_orders.iloc[p]['Time'], new_order])           
        
            print(extract_orders.iloc[p]['Time'],extract_orders.iloc[p]['Function'], new_order)
            
        if total_orders:
            orders_by_agent[f] = total_orders
                
    return orders_by_agent
       
def process_AI_orders_pd(message):
    
    regex_sense_room = "(\w+), sense room (\d+)"
    regex_carry_object = "(\w+), collect object (\d+)"
    regex_team_carry_object = "Team leader: (\w+). Helpers: (\[(,?\w+)+\]). Collect object (\d+)"
    regex_sense_object = "(\w+), sense object (\d+) at location (\(-?\d+\.\d+,-?\d+\.\d+\))"
    
    regex_leader_sense_room = "I'm sensing room (\d+)"
    regex_leader_sense_object = "I'm sensing object (\d+)"
    regex_leader_collect_object = "I'm collecting object (\d+)"
    
    orders_by_agent = {}
    for rematch in re.finditer(regex_sense_room,message):
        orders_by_agent[rematch.group(1)] = ['sense_room' + rematch.group(2)]
    for rematch in re.finditer(regex_sense_object,message):
        orders_by_agent[rematch.group(1)] = ['sense_' + rematch.group(2)]
    for rematch in re.finditer(regex_carry_object,message):
        orders_by_agent[rematch.group(1)] = ['carry_' + rematch.group(2)]
    for rematch in re.finditer(regex_team_carry_object,message):
        orders_by_agent[rematch.group(1)] = ['carry_' + rematch.group(4)]
        teammates = rematch.group(2)[1:].replace("]", "").split(",")
        
        for t in teammates:
            orders_by_agent[t] = ['follow_' + rematch.group(1)]
    
    orders_leader = ""
    for rematch in re.finditer(regex_leader_sense_room,message):
        orders_leader = 'sense_room' + rematch.group(1)
    for rematch in re.finditer(regex_leader_sense_object,message):
        orders_leader = 'sense_' + rematch.group(1)
    for rematch in re.finditer(regex_leader_collect_object,message):
        orders_leader = 'carry_' + rematch.group(1)
            
    return orders_by_agent, orders_leader
       
orders_by_human_agent = process_human_orders_pd()


def write_grouped_sums(data: dict, filename: str, group_size: int = 4):
    """
    Write a CSV file with sums of grouped values.
    
    Parameters:
        data (dict): Dictionary where keys map to lists of numbers.
        filename (str): Output CSV filename.
        group_size (int): How many values to group together (default = 4).
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Key", "Group", "Sum"])  # header
        
        for key, values in data.items():
            for i in range(0, len(values), group_size):
                group = i // group_size + 1  # group number
                chunk = values[i:i+group_size]
                total = round(sum(chunk),2)
                writer.writerow([key, group, total])


#
# Correlations with transitions between AI to human, human to AI, AI to AI, etc.
#

def _normalize_leader(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, np.integer)):
        return "AI" if int(val) == 1 else "Human"
    s = str(val).strip().lower()
    if s in ("ai","a","robot","bot","1","ai_leader","ai-led","ai led"): return "AI"
    if s in ("human","h","0","human_leader","human-led","human led"):    return "Human"
    if s in ("true","false"): return "AI" if s == "true" else "Human"
    if "ai" in s: return "AI"
    if "human" in s: return "Human"
    raise ValueError(f"Unrecognized leader value: {val!r}")

def add_transition_columns(df, team_col="team_id", session_col="session",
                           leader_col="leader_type", perf_col="performance",
                           feature_cols=None, compute_feature_deltas=False):
    """
    Adds: leader_norm, prev_leader, prev_performance, is_contiguous, transition, delta_performance.
    If compute_feature_deltas=True, also adds delta_<feature> for each feature in feature_cols.
    """
    df = df.copy()
    df = df.sort_values([team_col, session_col])
    df["leader_norm"] = df[leader_col].map(_normalize_leader)
    df["prev_leader"] = df.groupby(team_col)["leader_norm"].shift(1)
    df["prev_performance"] = df.groupby(team_col)[perf_col].shift(1)

    # contiguity: next session exactly +1; adjust if you use dates or different rule
    df["is_contiguous"] = df.groupby(team_col)[session_col].diff().eq(1)

    df["transition"] = np.where(
        df["is_contiguous"],
        df["prev_leader"].fillna("NA") + "->" + df["leader_norm"].fillna("NA"),
        np.nan
    )

    df["delta_performance"] = df[perf_col] - df["prev_performance"]

    if compute_feature_deltas and feature_cols:
        for f in feature_cols:
            # compute shift per team, then wipe non-contiguous deltas to NaN
            d = df[f] - df.groupby(team_col)[f].shift(1)
            df[f"delta_{f}"] = d.where(df["is_contiguous"], np.nan)

    return df

def compute_correlations(df, y_col, feature_cols):
    rows = []
    for feat in feature_cols:
        # only numeric features
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        sub = df[[feat, y_col]].dropna()
        n = len(sub)
        if n < 3:
            rows.append(dict(feature=feat, N=n, pearson_r=np.nan, pearson_p=np.nan,
                             spearman_rho=np.nan, spearman_p=np.nan))
            continue
        r, rp = stats.pearsonr(sub[feat], sub[y_col])
        rho, rhop = stats.spearmanr(sub[feat], sub[y_col])
        rows.append(dict(feature=feat, N=n, pearson_r=r, pearson_p=rp,
                         spearman_rho=rho, spearman_p=rhop))
    out = pd.DataFrame(rows).sort_values(["pearson_p","spearman_p","feature"])
    return out

def correlations_by_transition(df,
                               team_col="team_id", session_col="session",
                               leader_col="leader_type", perf_col="performance",
                               use_delta=False,  # when True: Δfeature vs Δperformance
                               drop_non_numeric=True):
    """
    If use_delta=False: correlates raw features with performance.
    If use_delta=True:  correlates Δfeatures with Δperformance (contiguous only).
    Uses your build_features(df) output directly (no merge).
    """
    # 1) Build features with your pipeline (keeps original columns)
    feat_df = build_features(df)

    # 2) Determine the feature set produced by build_features
    exclude = {team_col, session_col, leader_col, perf_col,
               "leader_norm", "prev_leader", "prev_performance",
               "is_contiguous", "transition", "delta_performance"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    # (optional) keep only numeric features up-front
    if drop_non_numeric:
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(feat_df[c])]

    # 3) Add transitions (+ feature deltas only if we will use them)
    df_t = add_transition_columns(
        feat_df,
        team_col=team_col, session_col=session_col,
        leader_col=leader_col, perf_col=perf_col,
        feature_cols=feature_cols,
        compute_feature_deltas=use_delta
    )

    # 4) Choose outcome & feature list for correlation
    if use_delta:
        y_col = "delta_performance"
        corr_features = [f"delta_{f}" for f in feature_cols if f"delta_{f}" in df_t.columns]
    else:
        y_col = perf_col
        corr_features = feature_cols

    # 5) Compute per-transition correlations
    results = {}
    for trans in ["AI->Human", "Human->AI", "AI->AI", "Human->Human"]:
        mask = df_t["transition"].eq(trans)
        results[trans] = compute_correlations(df_t.loc[mask], y_col, corr_features)

    return results



def save_latex_tables(results, filename_prefix="corr_tables", top_n=10, caption_suffix="performance"):
    """
    Save LaTeX tables for each transition type with the top N correlated features.
    results: dict from correlations_by_transition
    """
    for trans, df in results.items():
        if df.empty:
            continue
        # Take top N by absolute Pearson r
        df_top = df.reindex(df['pearson_r'].abs().sort_values(ascending=False).index).head(top_n)

        # LaTeX table
        latex_str = r"""\begin{table}[htbp]
\centering
\caption{Top correlations with %s for transition %s}
\label{tab:corr_%s}
\begin{tabular}{lrrrrr}
\toprule
Feature & $N$ & $r_{pearson}$ & $p_{pearson}$ & $\rho_{spearman}$ & $p_{spearman}$ \\
\midrule
""" % (caption_suffix, trans, trans.replace("->","to"))

        for _, row in df_top.iterrows():
            latex_str += f"{row['feature']} & {int(row['N'])} & {row['pearson_r']:.3f} & {row['pearson_p']:.3g} & {row['spearman_rho']:.3f} & {row['spearman_p']:.3g} \\\\\n"

        latex_str += r"""\bottomrule
\end{tabular}
\end{table}
"""

        with open(f"{filename_prefix}_{trans.replace('->','to')}.tex", "w") as f:
            f.write(latex_str)


def _detect_corr_col_trans(df: pd.DataFrame) -> str:
    """Pick the correlation column; prefer common names."""
    preferred = ["corr_with_Δperf", "corr_with_delta_perf", "corr_with_performance", "corr", "r"]
    for c in preferred:
        if c in df.columns:
            return c
    # fallback: first numeric column that isn't 'abs_corr'
    for c in df.columns:
        if c != "abs_corr" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric correlation column found in provided DataFrame.")

def _detect_corr_col_trans2(df: pd.DataFrame) -> str:
    """Always use Spearman's rho for transition correlation tables."""
    if "spearman_rho" not in df.columns:
        raise ValueError(
            f"'spearman_rho' not found in DataFrame. Available columns: {list(df.columns)}"
        )
    return "spearman_rho"


def _labelify(s: str) -> str:
    """Make a safe LaTeX label suffix from a transition string."""
    return s.replace(" ", "_").replace(">", "to").replace("/", "-")

def write_by_transition_latex_with_meta(
    corr_by_transition: Dict[str, pd.DataFrame],
    out_tex: str = "transition_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance",
    top_k: int = 50,
    abs_thresh: Optional[float] = None,     # filter inside single-transition & ALL-groups meta
    # Pairwise knobs:
    do_pairwise: bool = True,
    abs_thresh_pair: float = 0.0,          # strong in BOTH transitions means |r|>=this in each
    diff_thresh_pair: float = 0.40,         # if same sign but |Δr|>=this → dissimilar
    top_k_pair: int = 50,
    pretty_names: Optional[Dict[str, str]] = None,
    strength_thresh: int = 0.4  # map transition key to pretty caption text
):
    """
    corr_by_transition: dict like {
        "AI->Human": DataFrame(feature, corr_with_*),
        "Human->AI": ...,
        "AI->AI": ...,
        "Human->Human": ...
    }

    Produces:
      • One LaTeX table per transition
      • ALL-transitions meta tables: Top shared (min |r|), Top dissimilar (range of r)
      • Pairwise tables for every transition pair: Shared & Dissimilar (strong in both)
    """
    # ---------------- 1) Per-transition tables ----------------
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: correlations with performance by transition\n\n")
        for trans, tbl in corr_by_transition.items():
            name = pretty_names.get(trans, trans) if pretty_names else trans
            df = tbl.copy()
            corr_col = _detect_corr_col_trans2(df)
            df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
            df = df[df[corr_col].notna()].copy()
            if abs_thresh is not None:
                df = df.assign(abs_corr=df[corr_col].abs())
                df = df[df["abs_corr"] >= abs_thresh].drop(columns="abs_corr")
            f.write(f"%% -------- Transition {name} --------\n")
            f.write(df.to_latex(
                index=False, float_format="%.3f",
                caption=f"Correlations with performance — Transition {name} ({note_excluded})",
                label=f"tab:trans_corr_{_labelify(name)}"
            ))
            f.write("\n\n")

    # ---------------- 2) Long-form for ALL-transitions meta & pairwise ----------------
    long_rows = []
    for trans, tbl in corr_by_transition.items():
        name = pretty_names.get(trans, trans) if pretty_names else trans
        df = tbl.copy()
        corr_col = _detect_corr_col_trans2(df)
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
        df = df[df[corr_col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({
                "transition": name,
                "feature": getattr(r, "feature"),
                "corr": getattr(r, corr_col),
            })
    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return

    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()
    transitions = sorted(longdf["transition"].unique().tolist())

    # ---------------- 3) ALL-transitions meta ----------------
    meta_df = longdf.copy()
    if abs_thresh is not None:
        meta_df = meta_df[meta_df["abs_corr"] >= abs_thresh]

    agg = (
        meta_df.groupby("feature")
               .agg(n_trans=("transition","nunique"),
                    mean_corr=("corr","mean"),
                    mean_abs=("abs_corr","mean"),
                    min_abs=("abs_corr","min"),
                    max_abs=("abs_corr","max"),
                    min_corr=("corr","min"),
                    max_corr=("corr","max"))
               .reset_index()
    )
    shared_all = agg.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k).copy()
    agg["range_corr"] = agg["max_corr"] - agg["min_corr"]
    dissim_all = agg.sort_values("range_corr", ascending=False).head(top_k).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    shared_cols_all = ["feature","n_trans","min_abs","mean_abs","max_abs","min_corr","max_corr"]
    dissim_cols_all = ["feature","n_trans","range_corr","min_corr","max_corr"]

    with open(out_tex, "a") as f:
        f.write("%% -------- ALL TRANSITIONS: Top shared features --------\n")
        f.write(shared_all[shared_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} features shared across transitions "
                     f"(ranked by minimum |r| across transitions; {note_excluded})."),
            label="tab:trans_shared_top"
        ))
        f.write("\n\n")

        f.write("%% -------- ALL TRANSITIONS: Top dissimilar features --------\n")
        f.write(dissim_all[dissim_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} most dissimilar features across transitions "
                     f"(ranked by correlation range; {note_excluded})."),
            label="tab:trans_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------------- 4) Pairwise transition comparisons ----------------
    if not do_pairwise or len(transitions) < 2:
        print(f"Wrote LaTeX tables to: {out_tex}")
        return

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN TRANSITIONS --------\n\n")

    # Pre-filtered per-transition for pairwise logic (strong features only)
    per_trans_df = {}
    for T in transitions:
        sub = longdf[longdf["transition"] == T].copy()
        sub = sub[sub["abs_corr"] >= abs_thresh_pair]
        per_trans_df[T] = sub.set_index("feature")[["corr","abs_corr"]]

    for A, B in itertools.combinations(transitions, 2):
        A_df = per_trans_df[A]
        B_df = per_trans_df[B]
        if A_df.empty or B_df.empty:
            continue

        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        
        inter = inter[(inter["abs_corr_A"] >= strength_thresh) | 
              (inter["abs_corr_B"] >= strength_thresh)]
              
        if inter.empty:
            continue

        inter["sign_A"] = inter["corr_A"].apply(lambda x: 1 if x >= 0 else -1)
        inter["sign_B"] = inter["corr_B"].apply(lambda x: 1 if x >= 0 else -1)
        inter["same_sign"] = (inter["sign_A"] == inter["sign_B"])
        inter["delta_r"] = (inter["corr_B"] - inter["corr_A"]).abs()
        inter["min_abs"] = inter[["abs_corr_A","abs_corr_B"]].min(axis=1)

        # Shared (same sign, strong in both)
        shared = inter[inter["same_sign"]].sort_values(
            ["delta_r", "min_abs"], ascending=[True, False]
        ).head(top_k_pair).copy()

        # Dissimilar (opposite sign OR large Δ in magnitude)
        dissim_mask = (~inter["same_sign"]) | (inter["delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(
            ["delta_r","min_abs"], ascending=[False, False]
        ).head(top_k_pair).copy()

        shared_tbl = shared.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r"]].copy()
        shared_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|"}, inplace=True)
        dissim_tbl = dissim.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r","same_sign"]].copy()
        dissim_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|", "same_sign":"same_sign"}, inplace=True)

        for df_ in (shared_tbl, dissim_tbl):
            for c in df_.select_dtypes(include="number").columns:
                df_[c] = df_[c].round(3)

        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign, strong in both) --------\n")
            f.write(shared_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Transitions {A} vs {B}: shared strong correlations "
                         f"(|r|≥{abs_thresh_pair}, same sign; ranked by min|r|; {note_excluded})."),
                label=f"tab:trans_pair_shared_{_labelify(A)}_vs_{_labelify(B)}"
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Transitions {A} vs {B}: dissimilar strong correlations "
                         f"(|r|≥{abs_thresh_pair} in both; opposite sign or |Δr|≥{diff_thresh_pair}; {note_excluded})."),
                label=f"tab:trans_pair_dissimilar_{_labelify(A)}_vs_{_labelify(B)}"
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (including pairwise transition comparisons) to: {out_tex}")

#
# Calculate H
#

# ---------- numerics ----------
def _clip(p: float, eps: float = 1e-12) -> float:
    return min(max(p, eps), 1 - eps)

def entropy_binary(p: float, base: float = 2.0) -> float:
    p = _clip(p); q = 1 - p
    return -(p * math.log(p, base) + q * math.log(q, base))

def odds_from_prob(p: float) -> float:
    p = _clip(p); return p / (1 - p)

def prob_from_odds(o: float) -> float:
    return o / (1 + o)

# ---------- single-reading LR ----------
def lr_from_confidence(report: int, conf: float) -> float:
    conf = _clip(conf)
    if report == 2:   # "danger"
        return conf / (1 - conf)
    elif report == 1: # "benign"
        return (1 - conf) / conf
    else:
        raise ValueError("report must be 1 or 2")

# ---------- structure helpers ----------
def normalize_object_readings(obj_entry):
    """
    Accepts any of:
      1) A single-reading dict: {'from','danger_level','prob'}
      2) A list of per-reading dicts (each with keys above)
      3) A wrapper: {'readings': [...] } where [...] is (1) or (2)
      4) A 'vectorized' dict with parallel lists:
         {'from': [...], 'danger_level': [...], 'prob': [...]}
      5) None / missing -> returns []
    Returns a list of per-reading dicts: [{'from':str,'danger_level':1|2,'prob':float}, ...]
    """
    if obj_entry is None:
        return []

    # Case 3: wrapper
    if isinstance(obj_entry, dict) and "readings" in obj_entry:
        return normalize_object_readings(obj_entry["readings"])

    # Case 2: already a list of dicts
    if isinstance(obj_entry, list):
        # If the list contains a single vectorized dict, expand it
        if len(obj_entry) == 1 and isinstance(obj_entry[0], dict) and any(
            isinstance(obj_entry[0].get(k), list) for k in ("from", "danger_level", "prob")
        ):
            return normalize_object_readings(obj_entry[0])
        return obj_entry

    # Case 1: scalar single-reading dict
    if isinstance(obj_entry, dict) and not any(isinstance(obj_entry.get(k), list) for k in ("from","danger_level","prob")):
        if "danger_level" in obj_entry:
            return [obj_entry]
        return []

    # Case 4: vectorized dict with parallel lists
    if isinstance(obj_entry, dict):
        from_list  = obj_entry.get("from", [])
        level_list = obj_entry.get("danger_level", [])
        prob_list  = obj_entry.get("prob", [])

        # Normalize scalars to lists if mixed
        if not isinstance(from_list, list):  from_list  = [from_list]
        if not isinstance(level_list, list): level_list = [level_list]
        if not isinstance(prob_list, list):  prob_list  = [prob_list]

        n = max(len(from_list), len(level_list), len(prob_list))
        # Pad with None to align lengths, then filter invalids
        def _pad(v, n):
            return (v + [None] * (n - len(v)))[:n]

        from_list, level_list, prob_list = _pad(from_list, n), _pad(level_list, n), _pad(prob_list, n)

        out = []
        for f, dl, p in zip(from_list, level_list, prob_list):
            if f is None or dl is None or p is None:
                continue
            out.append({"from": f, "danger_level": int(dl), "prob": float(p)})
        return out

    # Fallback: treat as no readings
    return []

# ---------- fusion (supports zero readings) ----------
def fuse_readings_for_object(
    readings: Iterable[Dict[str, Any]],
    prior_danger: float = 0.5,
) -> Tuple[float, float, int]:
    """
    Fuses present readings; if none, returns (prior, H(prior), 0).
    Assumes each reading has keys: 'danger_level' (1|2) and 'prob' (confidence).
    """
    readings = list(readings)
    if len(readings) == 0:
        q = float(prior_danger)
        H = entropy_binary(q, base=2.0)
        return q, H, 0

    odds = odds_from_prob(prior_danger)
    for r in readings:
        report = int(r["danger_level"])     # 1 or 2
        conf   = float(r["prob"])           # your logger's single-reading posterior (prior=0.5)
        lr = lr_from_confidence(report, conf)
        odds *= lr

    q = prob_from_odds(odds)
    H = entropy_binary(q, base=2.0)
    return q, H, len(readings)

# ---------- main evaluation ----------
def evaluate_objects(
    known_objects_to_leader: Dict[Any, Any],
    groundtruth: Dict[Any, int],
    prior_danger: float = 0.5,
    max_readings: int = 4,
    entropy_epistemic_thresh_bits: float = 0.9,
) -> Dict[str, Any]:
    """
    Iterates over the union of object ids from groundtruth and leader-known.
    For unseen objects (no readings), posterior = prior and n_readings=0.
    """
    per = {}
    entropies = []
    epi_cnt = proc_cnt = 0

    all_obj_ids = set(groundtruth.keys()) | set(known_objects_to_leader.keys())

    for obj_id in all_obj_ids:
        gt = int(groundtruth[obj_id])  # must exist
        obj_entry = known_objects_to_leader.get(obj_id, None)
        readings = normalize_object_readings(obj_entry)

        q, H, n = fuse_readings_for_object(readings, prior_danger=prior_danger)

        fully_sensed = (n >= max_readings)
        epistemic_ambiguous   = (fully_sensed and H >= entropy_epistemic_thresh_bits)
        procedural_insuff     = ((not fully_sensed) and H >= entropy_epistemic_thresh_bits)

        per[obj_id] = {
            "q_danger": q,
            "entropy_bits": H,
            "n_readings": n,
            "seen": (n > 0),
            "fully_sensed": fully_sensed,
            "gt": gt,
            "epistemic_ambiguous": epistemic_ambiguous,
            "procedural_insufficient": procedural_insuff,
        }

        entropies.append(H)
        epi_cnt  += int(epistemic_ambiguous)
        proc_cnt += int(procedural_insuff)

    N = len(all_obj_ids)
    summary = {
        "N": N,
        "avg_entropy_bits": (sum(entropies) / N) if N else float("nan"),
        "epistemic_rate": epi_cnt / N if N else 0.0,
        "procedural_insuff_rate": proc_cnt / N if N else 0.0,
        "prior_used_count": sum(1 for v in per.values() if v["n_readings"] == 0),
    }
    return {"per_object": per, "summary": summary}

# ------------------- optional: calibration -------------------
def _make_calibration_bins(per_object: Dict[Any, Dict[str, Any]], n_bins: int = 10) -> List[Dict[str, float]]:
    """
    Reliability diagram bins for quick calibration checks.
    Returns a list of dicts with:
      'q_mid', 'empirical_danger_rate', 'count'
    """
    items = [(v["q_danger"], 1 if v["gt"] == 2 else 0) for v in per_object.values()]
    items.sort(key=lambda t: t[0])
    bins = []
    if not items:
        return bins

    size = max(1, len(items) // n_bins)
    for i in range(0, len(items), size):
        chunk = items[i:i+size]
        q_avg = sum(q for q,_ in chunk) / len(chunk)
        y_avg = sum(y for _,y in chunk) / len(chunk)
        bins.append({"q_mid": q_avg, "empirical_danger_rate": y_avg, "count": len(chunk)})
    return bins


def analyze_ambiguous_cases(per_object, entropy_thresh=0.9):
    ambiguous = []
    ambiguous_dangerous = []
    total_dangerous = sum(1 for m in per_object.values() if m["gt"] == 2)

    for oid, m in per_object.items():
        if m["epistemic_ambiguous"]:  # flagged by earlier evaluation
            ambiguous.append({
                "obj": oid,
                "q_danger": round(m["q_danger"], 3),
                "entropy_bits": round(m["entropy_bits"], 3),
                "gt": m["gt"]
            })
            if m["gt"] == 2:
                ambiguous_dangerous.append(oid)

    total = len(ambiguous)
    if total == 0:
        print("No epistemic ambiguous cases.")
        return 0,[]

    dangerous_count = sum(1 for a in ambiguous if a["gt"] == 2)
    benign_count = total - dangerous_count

    print(f"Ambiguous objects: {total}")
    print(f"  Dangerous: {dangerous_count} ({dangerous_count/total:.1%})")
    print(f"  Benign:    {benign_count} ({benign_count/total:.1%})")

    frac_danger_ambiguous = 0
    if total_dangerous > 0:
        frac_danger_ambiguous = dangerous_count / total_dangerous
        print(f"\nFrom all {total_dangerous} dangerous objects, "
              f"{dangerous_count} were ambiguous ({frac_danger_ambiguous:.1%}).")
    else:
        print("\nNo dangerous objects in dataset.")

    print("\nDetails of ambiguous cases:")
    for a in ambiguous:
        label = "DANGEROUS" if a["gt"] == 2 else "benign"
        print(f"Obj {a['obj']}: q={a['q_danger']}, "
              f"H={a['entropy_bits']} bits, GT={label}")

    return frac_danger_ambiguous, ambiguous_dangerous
    
    

def find_high_conf_misclassifications_fully_sensed(
    per_object: Dict[Any, Dict[str, Any]],
    metrics: Optional[Dict[Any, Dict[str, float]]] = None,
    *,
    # Confidence definition (pick one style)
    q_low: float = 0.20,        # q <= 0.20 -> confidently benign
    q_high: float = 0.80,       # q >= 0.80 -> confidently dangerous
    use_margin: bool = False,   # if True, ignore q_low/q_high and use |q-0.5| >= margin_delta
    margin_delta: float = 0.30, # high-confidence if |q-0.5| >= 0.30 (≈ q<=0.20 or q>=0.80)
    # Full-sensing requirement
    only_fully_sensed: bool = True,
    max_readings: int = 4       # defines "fully sensed" when only_fully_sensed=True
):
    """
    per_object[obj_id] must include:
        - 'q_danger' (float)
        - 'gt' in {1,2}
        - either 'n_readings' (int) or 'fully_sensed' (bool)
        - (optional) 'entropy_bits'
    metrics[obj_id] (optional): may include 'weight' (float)

    Returns:
      {
        "false_negatives": [ {...}, ... ],   # GT=2 but q indicates confident benign
        "false_positives": [ {...}, ... ],   # GT=1 but q indicates confident dangerous
        "summary": { ... counts and weighted shares ... }
      }
    """
    # Resolve thresholds
    if use_margin:
        q_low = 0.5 - margin_delta
        q_high = 0.5 + margin_delta

    fn: List[Dict[str, Any]] = []
    fp: List[Dict[str, Any]] = []

    total_dangerous = 0
    total_benign = 0
    total_dangerous_w = 0.0
    total_benign_w = 0.0
    fn_w = 0.0
    fp_w = 0.0

    def is_fully_sensed(m: Dict[str, Any]) -> bool:
        if "fully_sensed" in m:
            return bool(m["fully_sensed"])
        if "n_readings" in m and m["n_readings"] is not None:
            return int(m["n_readings"]) >= max_readings
        # If unknown, conservatively treat as not fully sensed
        return False

    for oid, m in per_object.items():
        q = float(m["q_danger"])
        gt = int(m["gt"])
        H = float(m.get("entropy_bits", float("nan")))
        n = m.get("n_readings", None)

        # Enforce fully-sensed filter if requested
        if only_fully_sensed and not is_fully_sensed(m):
            continue

        w = 1.0
        if metrics is not None and oid in metrics:
            w = float(metrics[oid].get("weight", 1.0))

        if gt == 2:
            total_dangerous += 1
            total_dangerous_w += w
            if q <= q_low:
                fn.append({
                    "obj": oid, "q": round(q, 3), "H_bits": round(H, 3),
                    "n_readings": (None if n is None else int(n)),
                    "gt": gt, "weight": w, "fully_sensed": is_fully_sensed(m)
                })
                fn_w += w

        elif gt == 1:
            total_benign += 1
            total_benign_w += w
            if q >= q_high:
                fp.append({
                    "obj": oid, "q": round(q, 3), "H_bits": round(H, 3),
                    "n_readings": (None if n is None else int(n)),
                    "gt": gt, "weight": w, "fully_sensed": is_fully_sensed(m)
                })
                fp_w += w

    summary = {
        "only_fully_sensed": only_fully_sensed,
        "max_readings": max_readings,
        "q_low": q_low, "q_high": q_high,
        "use_margin": use_margin, "margin_delta": margin_delta,
        "counts": {
            "total_dangerous_fs": total_dangerous,
            "total_benign_fs": total_benign,
            "false_negatives_fs": len(fn),
            "false_positives_fs": len(fp),
            "fn_rate_among_dangerous_fs": (len(fn)/total_dangerous) if total_dangerous else 0.0,
            "fp_rate_among_benign_fs": (len(fp)/total_benign) if total_benign else 0.0,
        },
        "weighted": {
            "total_dangerous_w_fs": total_dangerous_w,
            "total_benign_w_fs": total_benign_w,
            "fn_weight_fs": fn_w,
            "fp_weight_fs": fp_w,
            "fn_weight_share_fs": (fn_w/total_dangerous_w) if total_dangerous_w>0 else 0.0,
            "fp_weight_share_fs": (fp_w/total_benign_w) if total_benign_w>0 else 0.0,
        }
    }

    return {"false_negatives": fn, "false_positives": fp, "summary": summary}

#
#   Plot functions
#



def plot_variable_slot_width_diverging_from_df(
    df_sorted,
    order=("0→0","0→1","1→0","1→1"),
    max_slot_width=0.9,        # width for the transition with the largest n_total
    gap=0.1,                   # smaller gap = bars closer
    colors=("tab:orange", "tab:blue"),
    annotate=True,
    show_total=False           # <--- option to toggle the top "n=" label
):
    """
    Variable-width diverging bars:
      - Slot width for transition t:  w_slot[t] ∝ n_total[t] (max_slot_width at max n_total)
      - Inside the slot:
          width(↓) = w_slot[t] * (n_dec / n_total)
          width(↑) = w_slot[t] * (n_inc / n_total)
      - Heights are conditional means (↓ negative, ↑ positive).
    """
    # --- compute per-transition stats directly from df_sorted ---
    rows = []
    for t in order:
        g = df_sorted.loc[df_sorted["transition"] == t, "perf_change"].dropna()
        n_tot = int(g.size)
        n_inc = int((g > 0).sum())
        n_dec = int((g < 0).sum())
        rows.append(dict(
            transition=t, n_total=n_tot, n_inc=n_inc, n_dec=n_dec,
            mean_inc=(g[g > 0].mean() if n_inc > 0 else np.nan),
            mean_dec=(g[g < 0].mean() if n_dec > 0 else np.nan),
        ))
    T = pd.DataFrame(rows).set_index("transition")

    # scale slot widths by total samples
    n_max = max(1, T["n_total"].max())  # avoid divide-by-zero
    T["slot_w"] = max_slot_width * (T["n_total"] / n_max)

    # x centers with variable widths + constant gaps
    centers = []
    x = 0.0
    for t in order:
        w = T.loc[t, "slot_w"]
        centers.append(x + w/2.0)
        x += w + gap
    T["x"] = centers

    # y limits (pad a bit)
    y_vals = []
    if T["mean_inc"].notna().any(): y_vals += T["mean_inc"].dropna().tolist()
    if T["mean_dec"].notna().any(): y_vals += T["mean_dec"].dropna().tolist()
    y_vals = y_vals or [0.0]
    y_min, y_max = min(y_vals + [0]), max(y_vals + [0])
    pad = 0.08 * (y_max - y_min if y_max != y_min else 1.0)
    y_min, y_max = y_min - pad, y_max + pad
    y_span = max(1e-9, y_max - y_min)

    dec_color, inc_color = colors
    fig, ax = plt.subplots(figsize=(10, 5))

    for t in order:
        if t not in T.index or T.loc[t, "n_total"] == 0:
            continue
        x = T.loc[t, "x"]
        w_slot = T.loc[t, "slot_w"]
        n_tot  = int(T.loc[t, "n_total"])
        n_inc  = int(T.loc[t, "n_inc"])
        n_dec  = int(T.loc[t, "n_dec"])
        m_inc  = T.loc[t, "mean_inc"]
        m_dec  = T.loc[t, "mean_dec"]

        # widths inside the slot proportional to their counts
        w_inc = w_slot * (n_inc / n_tot) if n_tot else 0.0
        w_dec = w_slot * (n_dec / n_tot) if n_tot else 0.0

        # ↓ decrease (below 0)
        if n_dec > 0 and pd.notna(m_dec) and (m_dec < 0):
            ax.add_patch(Rectangle((x - w_dec/2, m_dec), w_dec, -m_dec,
                                   facecolor=dec_color, edgecolor="black",
                                   linewidth=0.8, zorder=2))
            if annotate:
                ax.text(x, m_dec - 0.02*y_span, f"n↓={n_dec}",
                        ha="center", va="top", fontsize=8)

        # ↑ increase (above 0)
        if n_inc > 0 and pd.notna(m_inc) and (m_inc > 0):
            ax.add_patch(Rectangle((x - w_inc/2, 0), w_inc, m_inc,
                                   facecolor=inc_color, edgecolor="black",
                                   linewidth=0.8, zorder=3))
            if annotate:
                ax.text(x, m_inc + 0.02*y_span, f"n↑={n_inc}",
                        ha="center", va="bottom", fontsize=8)

        # total n (optional)
        if annotate and show_total:
            ax.text(x, y_max - 0.04*y_span, f"n={n_tot}", ha="center", va="top", fontsize=9)

    # axes cosmetics
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_ylim(y_min, y_max)
    last_center = T.loc[order[-1], "x"]
    last_w = T.loc[order[-1], "slot_w"]
    ax.set_xlim(T.loc[order[0], "x"] - T.loc[order[0], "slot_w"]/2 - gap/2,
                last_center + last_w/2 + gap/2)

    ax.set_xticks([T.loc[t, "x"] for t in order])
    ax.set_xticklabels(order)
    ax.set_ylabel("Δ Performance (means)")
    ax.set_xlabel("Leader transition (0=Human, 1=AI)")
    ax.set_title("Diverging bars with slot width ∝ total samples; bar width ∝ outcome counts")

    handles = [
        Rectangle((0,0),1,1, facecolor=dec_color, edgecolor="black", label="Mean decrease (width ∝ n↓)"),
        Rectangle((0,0),1,1, facecolor=inc_color, edgecolor="black", label="Mean increase (width ∝ n↑)"),
    ]
    ax.legend(handles=handles, frameon=True, loc="best")
    plt.tight_layout()
    plt.show()


#
# Outliers
#


def detect_outliers(df, perf_col="performance", method="iqr", group_col=None):
    """
    Detects outliers in performance values.
    
    Parameters:
        df : pd.DataFrame
            Must contain perf_col (and group_col if used).
        perf_col : str
            Column with performance values.
        method : str
            "iqr" (default) or "zscore".
        group_col : str or None
            If provided, detect outliers within each group (e.g., team_id).
            If None, detect outliers globally across all rows.
            
    Returns:
        pd.DataFrame with added column "is_outlier".
    """
    df = df.copy()
    df["is_outlier"] = False
    
    if group_col:
        groups = df.groupby(group_col)
    else:
        # treat whole df as one group
        groups = [(None, df)]
    
    for _, group in groups:
        values = group[perf_col]
        
        if method == "iqr":
            Q1, Q3 = values.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            mask = (values < lower) | (values > upper)
        
        elif method == "zscore":
            mean, std = values.mean(), values.std()
            z_scores = (values - mean) / (std if std > 0 else 1)
            mask = np.abs(z_scores) > 3
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        df.loc[group.index, "is_outlier"] = mask
    
    return df

# -----------------------------
# PER-SESSION DIFFERENCES
# -----------------------------


# =========================
# Helpers
# =========================
def gini(arr):
    x = np.array(list(arr), dtype=float)
    if len(x)==0: return np.nan
    x = x[x>=0]
    if x.size == 0: return np.nan
    if x.sum() == 0: return 0.0
    x.sort()
    n = len(x)
    return (2*np.sum((np.arange(1, n+1) * x)))/(n*np.sum(x)) - (n+1)/n

def zscore_col(s):
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros_like(s), index=s.index)
    return (s - s.mean())/std

def safe_ratio(a,b):
    if pd.isna(a) or pd.isna(b) or b==0: return np.nan
    return a/b

# =========================
# Row-wise session summary
# =========================
def session_summary(row):
    # Required base fields (raise helpful errors if missing)
    req = [
        'carried_objects','weight_by_object','danger_level_by_object',
        'carried_distance_by_object','time_over_distance',
        'time_first_order_to_carry','sensings_per_object','dist_by_agent',
        'total_orders','misunderstood_orders','orders_changed','completed_orders'
    ]
    for k in req:
        if k not in row or row[k] is None:
            raise ValueError(f"Missing field '{k}' in df_sessions row.")

    carried = set(row['carried_objects'])
    weights = row['weight_by_object']
    danger  = row['danger_level_by_object']  # 1=dangerous, 0=benign (or thresholded)
    carried_dist = row['carried_distance_by_object']
    
    carried_dist_no_filter = row['carried_distance_by_object_no_filter']
    
    tod     = row['time_over_distance']         # per object
    t_first = row['time_first_order_to_carry']  # per object
    sensings = row['sensings_per_object']       # per object in [0..4]
    dist_agents = row['dist_by_agent']          # dict {agent: distance}

    # Dangerous vs benign carried
    carried_danger = [o for o in carried if danger.get(o,0)>=1]
    carried_benign = [o for o in carried if danger.get(o,0)<1]
    all_carried = len(carried)
    n_danger_carried = len(carried_danger)
    n_benign_carried = len(carried_benign)

    precision = safe_ratio(n_danger_carried, all_carried)
    benign_ratio = safe_ratio(n_benign_carried, all_carried)
    total_danger_weight_available = sum(w for o,w in weights.items() if danger.get(o,0)>=1)
    carried_danger_weight = sum(weights.get(o,0) for o in carried_danger)
    danger_weight_coverage = safe_ratio(carried_danger_weight, total_danger_weight_available)

    # Sensing distribution & sufficiency
    sens_counts_all = [sensings.get(o,0) for o in weights.keys()]
    total_objs = len(sens_counts_all) if sens_counts_all else 0
    bins = {k: (sum(1 for s in sens_counts_all if s==k)/total_objs if total_objs else np.nan) for k in range(5)}
    redundancy = np.mean([max(s-1,0) for s in sens_counts_all]) if sens_counts_all else np.nan
    median_sens_danger_carried = np.median([sensings.get(o,0) for o in carried_danger]) if n_danger_carried else np.nan
    pct_danger_carried_ge2 = safe_ratio(sum(1 for o in carried_danger if sensings.get(o,0)>=2), n_danger_carried)

    # Path efficiency (time per distance) for dangerous objects (lower is better)
    eff_vals = [tod.get(o, np.nan) for o in carried_danger]
    path_eff_median = np.nanmedian(eff_vals) if len(eff_vals) else np.nan

    # Latency normalized by (distance * weight)
    lat_vals = []
    for o in carried:
        base = (carried_dist.get(o, np.nan) * weights.get(o, np.nan))
        lat  = t_first.get(o, np.nan)
        if pd.notna(base) and base>0 and pd.notna(lat):
            lat_vals.append(lat / base)
            
    latency_norm_median = np.nanmedian(lat_vals) if len(lat_vals) else np.nan

    # Latency normalized by (distance * weight)
    lat_vals = []
    for o in carried:
        base = (carried_dist_no_filter.get(o, np.nan) * weights.get(o, np.nan))
        lat  = t_first.get(o, np.nan)
        if pd.notna(base) and base>0 and pd.notna(lat):
            lat_vals.append(lat / base)
            
    latency_norm_median_no_filter = np.nanmedian(lat_vals) if len(lat_vals) else np.nan



    # Interaction metrics from orders
    total_orders = row['total_orders']
    misunderstanding_rate = safe_ratio(row['misunderstood_orders'], total_orders)
    change_rate = safe_ratio(row['orders_changed'], total_orders)
    completion_rate = safe_ratio(row['completed_orders'], total_orders)
    completed_orders = row['completed_orders']
    orders_to_AI = row.get('orders_to_AI', total_orders)
    incomplete_order_rate = safe_ratio(row['incomplete_orders'], total_orders)
    underspecified_rate = safe_ratio(row['underspecified'], total_orders)

    # Mobility skew
    mobility_gini = gini(list(dist_agents.values()))

    # Distance allocation
    benign_weighted_distance = sum(weights.get(o,0)*carried_dist.get(o,0) for o in carried_benign)
    dangerous_weighted_distance = sum(weights.get(o,0)*carried_dist.get(o,0) for o in carried_danger)
    dangerous_distance = sum(carried_dist.get(o,0) for o in carried_danger)
    
    benign_weighted_distance_no_filter = sum(weights.get(o,0)*carried_dist_no_filter.get(o,0) for o in carried_benign)
    dangerous_weighted_distance_no_filter = sum(weights.get(o,0)*carried_dist_no_filter.get(o,0) for o in carried_danger)
    dangerous_distance_no_filter = sum(carried_dist_no_filter.get(o,0) for o in carried_danger)
    
    lost_track_indicator = lost_track #/row['total_messages']
    
    fraction_of_orders_to_messages = 0
    if row["total_messages"]:
        fraction_of_orders_to_messages = total_orders/row["total_messages"]

    series_dict = dict(precision=precision,
        benign_ratio=benign_ratio,
        danger_weight_coverage=danger_weight_coverage,
        median_sens_danger_carried=median_sens_danger_carried,
        pct_danger_carried_ge2=pct_danger_carried_ge2,
        path_eff_median=path_eff_median,
        latency_norm_median=latency_norm_median,
        misunderstanding_rate=misunderstanding_rate,
        change_rate=change_rate,
        completion_rate=completion_rate,
        redundancy=redundancy,
        mobility_gini=mobility_gini,
        benign_weighted_distance=benign_weighted_distance,
        dangerous_weighted_distance=dangerous_weighted_distance,
        sensed_0=bins[0], sensed_1=bins[1], sensed_2=bins[2], sensed_3=bins[3], sensed_4=bins[4],
        incomplete_order_rate=incomplete_order_rate,
        underspecified_rate=underspecified_rate,
        lost_track=lost_track_indicator,
        completed_orders=completed_orders,
        orders_to_AI=orders_to_AI,
        total_orders=total_orders,
        fraction_of_orders_to_messages=fraction_of_orders_to_messages,
        latency_norm_median_no_filter=latency_norm_median_no_filter,
        benign_weighted_distance_no_filter=benign_weighted_distance_no_filter,
        dangerous_weighted_distance_no_filter=dangerous_weighted_distance_no_filter,
        dangerous_distance=dangerous_distance,
        dangerous_distance_no_filter=dangerous_distance_no_filter
        
    )
    
    survey_keys = ['Dependable', 'Reliable', 'Unresponsive', 'Predictable', 'Act consistently', 'Function successfully', 'Malfunction', 'Have errors', 'Provide feedback', 'Meet the needs of the mission/task', 'Provide appropriate information', 'Communicate with people', 'Perform exactly as instructed', 'Follow directions', 'Mental Demand: How mentally demanding was the task?', 'Physical Demand: How physically demanding was the task?', 'Temporal Demand: How hurried or rushed was the pace of the task?', 'Performance: How successful were you in accomplishing what you were asked to do?', 'Effort: How hard did you have to work to accomplish your level of performance?', 'Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?']

    for sk_idx,sk in enumerate(survey_keys):
        if sk not in row:
            series_dict[sk] = float("nan")
        else:
            series_dict[sk] = row[sk]
            
        for prefix in ["leader_","nonleader_"]:
            new_sk = prefix + sk
            if new_sk not in row:
                series_dict[new_sk] = float("nan")
            else:
                series_dict[new_sk] = row[new_sk]
        
    for sk_idx,sk in enumerate(demographic_qs):
        if sk_idx: #Skip Name
            if sk not in row:
                series_dict[sk] = float("nan")
            else:
                series_dict[sk] = row[sk]
            
            for prefix in ["leader_","nonleader_"]:
                new_sk = prefix + sk
                if new_sk not in row:
                    series_dict[new_sk] = float("nan")
                else:
                    series_dict[new_sk] = row[new_sk]

    more_keys = ["frac_dangerous_ambiguous", "frac_benign_as_dangerous", "suggestions_exchanged", "PTT", "NASA", 'team_dispersion', 'dispersion_to_leader', 'total_changes_in_team', 'mean_distance_order_carry', 'team_carrying_composition_AI', 'times_reprimanded', 'communication_range_error', 'drop_per_distance', 'completed_other_type', 'uncertainty_at_grabbing', 'probability_at_grabbing', 'reliability_score', "repeated_orders", "dangerous_weighted_work", "relative_performance", "complexity", "normalized_complexity", "suggestions_accepted", "dangerous_weighted_work_no_filter", "relative_performance_no_filter", "complexity_no_filter", "performance_no_filter", "true_positive", "false_negative", "false_positive", "sensitivity_filter", "weight_sum",  "true_positive_goal", "false_negative_goal", "false_positive_goal", "weight_sum_goal", "number_false_negatives", "number_ambigous_dangerous", "total_dangerous", "total_benign", "number_ambiguous", "strategy", "team_composition"]
    for sk in more_keys:
        if sk not in row:
            series_dict[sk] = float("nan")
        else:
            series_dict[sk] = row[sk]

        if sk == "NASA" or sk == "PTT":
            for prefix in ["leader_","nonleader_"]:
                new_sk = prefix + sk
                if new_sk not in row:
                    series_dict[new_sk] = float("nan")
                else:
                    series_dict[new_sk] = row[new_sk]

    return pd.Series(series_dict)
'''
# =========================
# Feature builder
# =========================
def build_features(df_sessions, danger_is_positive=True):
    """
    df_sessions must include at least:
      team_id, session, performance, leader_type
      and the dict fields used by session_summary()
    """
    # Normalize leader_type
    if 'leader_type' not in df_sessions.columns:
        raise ValueError("df_sessions must include 'leader_type' (e.g., 'human' or 'ai').")
    df = df_sessions.copy()
    df['leader_type'] = df['leader_type'].astype(str).str.lower().map(lambda s: 'human' if 'human' in s else ('ai' if 'ai' in s else s))

    feats = df.apply(session_summary, axis=1)
    out = pd.concat([df[['team_id','session','leader_type','performance']].reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)

    # Build Interaction & Competence indices (z-scored across all rows)
    # Interaction: higher = worse interaction; Competence: higher = better competence
    inter_cols_pos = ['misunderstanding_rate','change_rate','redundancy','latency_norm_median','mobility_gini','underspecified_rate','incomplete_order_rate']
    inter_cols_neg = ['completion_rate']  # subtract because higher is good
    comp_cols_pos  = ['precision','danger_weight_coverage','pct_danger_carried_ge2']
    comp_cols_neg  = ['benign_ratio','path_eff_median']       # subtract because lower is good

    # Ensure columns exist
    for c in chain(inter_cols_pos, inter_cols_neg, comp_cols_pos, comp_cols_neg):
        if c not in out.columns: out[c] = np.nan

    Z = out.copy()
    # z-score numeric columns
    for c in out.columns:
        if c not in ['team_id','session','leader_type']:
            Z[c] = zscore_col(out[c].astype(float))

    InteractionIndex = Z[inter_cols_pos].mean(axis=1) - Z[inter_cols_neg].mean(axis=1)
    CompetenceIndex  = Z[comp_cols_pos].mean(axis=1)  - Z[comp_cols_neg].mean(axis=1)

    out['InteractionIndex'] = InteractionIndex
    out['CompetenceIndex']  = CompetenceIndex
    return out

# =========================
# Deltas
# =========================
def build_deltas(feat_df):
    """
    Adds Δ_ columns representing first differences within each team sorted by session.
    """
    feat_df = feat_df.sort_values(['team_id','session'])
    num_cols = [c for c in feat_df.columns if c not in ['team_id','session','leader_type']]
    delta = feat_df.groupby('team_id', group_keys=False).apply(
        lambda g: g.assign(**{f"Δ_{c}": g[c].diff() for c in num_cols})
    )
    return delta

def build_deltas_same_leadership(feat_df):
    """
    Deltas within (team_id, leader_type), i.e., only compare sessions with the same leader type.
    """
    feat_df = feat_df.sort_values(['team_id','leader_type','session'])
    num_cols = [c for c in feat_df.columns if c not in ['team_id','session','leader_type']]
    delta = feat_df.groupby(['team_id','leader_type'], group_keys=False).apply(
        lambda g: g.assign(**{f"Δ_{c}": g[c].diff() for c in num_cols})
    )
    return delta

# =========================
# Regressions
# =========================
def run_fe_regression(feat_df, add_leader_fe=False):
    """
    Fixed-effects via dummies: team FE (and optional leader_type FE).
    """
    data = feat_df.copy()
    d_team = pd.get_dummies(data['team_id'], prefix='team', drop_first=True)
    pieces = [data[['performance','InteractionIndex','CompetenceIndex']], d_team]
    if add_leader_fe:
        d_lead = pd.get_dummies(data['leader_type'], prefix='lead', drop_first=True)
        pieces.append(d_lead)
    X = pd.concat(pieces, axis=1).dropna()
    y = X['performance']
    # Build formula dynamically
    cols = list(X.columns)
    cols.remove('performance')
    formula = 'performance ~ ' + ' + '.join(cols)
    model = smf.ols(formula, data=X).fit(cov_type='HC3')
    return model

def run_fe_regression_by_leadership(feat_df, leader_type):
    """
    FE regression restricted to a specific leader type (ai or human).
    """
    sub = feat_df[feat_df['leader_type'].str.lower()==leader_type.lower()].copy()
    if sub.empty:
        raise ValueError(f"No rows for leader_type={leader_type}")
    d_team = pd.get_dummies(sub['team_id'], prefix='team', drop_first=True)
    X = pd.concat([sub[['performance','InteractionIndex','CompetenceIndex']], d_team], axis=1).dropna()
    y = X['performance']
    cols = list(X.columns); cols.remove('performance')
    formula = 'performance ~ ' + ' + '.join(cols)
    model = smf.ols(formula, data=X).fit(cov_type='HC3')
    return model

# =========================
# Attribution
# =========================
def attribute_changes(delta_df, model):
    """
    Uses model betas to explain Δperformance by ΔInteractionIndex and ΔCompetenceIndex.
    """
    bI = model.params.get('InteractionIndex', np.nan)
    bC = model.params.get('CompetenceIndex', np.nan)
    res = delta_df[['team_id','session','leader_type','Δ_performance','Δ_InteractionIndex','Δ_CompetenceIndex']].copy()
    res = res.dropna(subset=['Δ_performance'])
    res['Explained_by_Interaction'] = bI * res['Δ_InteractionIndex']
    res['Explained_by_Competence']  = bC * res['Δ_CompetenceIndex']
    res['Explained_Total'] = res['Explained_by_Interaction'] + res['Explained_by_Competence']
    res['Unexplained']     = res['Δ_performance'] - res['Explained_Total']
    with np.errstate(divide='ignore', invalid='ignore'):
        res['%Explained']  = 100 * (res['Explained_Total'] / res['Δ_performance'])
    return res

# =========================
# Leadership-split wrapper
# =========================
def leadership_split_analysis(df_sessions):
    """
    End-to-end:
      1) Build features
      2) Build deltas (global) and deltas restricted to same-leadership
      3) FE regressions overall + per-leadership
      4) Attribution overall + per-leadership
      5) Compact leadership-split summaries
    Returns a dict of DataFrames/models.
    """
    feat_df = build_features(df_sessions)
    delta_all = build_deltas(feat_df)
    delta_same = build_deltas_same_leadership(feat_df)

    # Overall FE with team FE (and leader FE as control)
    model_overall = run_fe_regression(feat_df, add_leader_fe=True)
    attr_overall_all_deltas = attribute_changes(delta_all, model_overall)
    attr_overall_samelead   = attribute_changes(delta_same, model_overall)

    # FE per leadership type
    results_by_lead = {}
    for lt in ['human','ai']:
        try:
            model_lt = run_fe_regression_by_leadership(feat_df, lt)
            # Attribute only on same-leadership deltas of this type
            deltas_lt = delta_same[delta_same['leader_type']==lt].copy()
            attr_lt   = attribute_changes(deltas_lt, model_lt)

            # Quick summary of average Δ vs explained components
            summary = (attr_lt
                       .groupby('team_id')[['Δ_performance','Explained_by_Interaction','Explained_by_Competence','Unexplained']]
                       .mean()
                       .rename(columns={
                           'Δ_performance':'avg_Δ_perf',
                           'Explained_by_Interaction':'avg_expl_by_interaction',
                           'Explained_by_Competence':'avg_expl_by_competence',
                           'Unexplained':'avg_unexplained'
                       }))
            results_by_lead[lt] = {
                'model': model_lt,
                'deltas_same_lead': deltas_lt,
                'attribution': attr_lt,
                'summary_by_team': summary
            }
        except Exception as e:
            results_by_lead[lt] = {'error': str(e)}

    # Compact leadership-wide summaries (mean over all rows)
    def compact_attr(attr_df):
        if attr_df.empty:
            return pd.Series(dtype=float)
        return pd.Series({
            'mean_Δ_perf': attr_df['Δ_performance'].mean(),
            'mean_expl_interaction': attr_df['Explained_by_Interaction'].mean(),
            'mean_expl_competence':  attr_df['Explained_by_Competence'].mean(),
            'mean_unexplained':      attr_df['Unexplained'].mean(),
            'mean_%explained':       attr_df['%Explained'].replace([np.inf,-np.inf], np.nan).mean()
        })

    overall_compact_all   = compact_attr(attr_overall_all_deltas)
    overall_compact_same  = compact_attr(attr_overall_samelead)

    return {
        'feat_df': feat_df,
        'delta_all': delta_all,
        'delta_same_lead': delta_same,
        'model_overall': model_overall,
        'attr_overall_all_deltas': attr_overall_all_deltas,
        'attr_overall_samelead': attr_overall_samelead,
        'overall_compact_all_deltas': overall_compact_all,
        'overall_compact_samelead': overall_compact_same,
        'by_leader': results_by_lead
    }
'''


def _pick_contrast_col(df: pd.DataFrame) -> str:
    prefer = ["contrast", "improved_minus_declined", "Δmean", "delta", "diff"]
    numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for name in prefer:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    if not numerics:
        raise ValueError("No numeric contrast column found.")
    return numerics[0]

def _top_k_pos_neg(df: pd.DataFrame, k: int):
    d = df.copy()
    col = _pick_contrast_col(d)
    d_pos = d.sort_values(col, ascending=False).head(k).copy()
    d_neg = d.sort_values(col, ascending=True ).head(k).copy()
    for c in d.select_dtypes(include="number").columns:
        d_pos[c] = d_pos[c].round(3)
        d_neg[c] = d_neg[c].round(3)
    return col, d_pos, d_neg

def _labelify_transition_pair(a: int, b: int) -> str:
    return f"{a}to{b}"

# ---------- main writer ----------
def write_fixed_transition_latex_with_meta(
    res_fixed_any: dict,
    out_tex: str = "fixed_transition_corr_tables.tex",
    note_excluded: str = "fixed pairs; excluding dangerous_weighted_distance",
    top_k_meta: int = 10,
    abs_thresh_single: float = None,   # filter in single-transition & ALL-transitions meta (e.g., 0.30)
    # pairwise thresholds
    abs_thresh_pair: float = 0.30,     # strong in BOTH transitions
    diff_thresh_pair: float = 0.40,    # same sign but large |Δr|
    top_k_pair: int = 10,
    # top-k improved vs declined
    top_k_contrast: int = 10,
    strength_thresh: int = 0.4
):
    """
    Works directly off res_fixed_any['delta_df'] produced by raw_delta_descriptives_flexible(..., pair_mode='fixed').
    Creates:
      1) Per-transition correlation tables (Δfeature vs Δperformance)
      2) ALL-transitions meta tables: Top shared (min|r|), Top dissimilar (range of r)
      3) Pairwise transition comparisons: Shared & Dissimilar (strong in both)
      4) Per-transition Top-k improved vs declined (from group means of Δ_* by perf_dir)
    """

    delta_df = res_fixed_any["delta_df"].copy()
    if delta_df.empty:
        with open(out_tex, "w") as f:
            f.write("% No delta rows in res_fixed_any['delta_df']\n")
        print(f"Wrote LaTeX (empty): {out_tex}")
        return

    # ensure perf_dir exists (should already from your function)
    if "perf_dir" not in delta_df.columns:
        def _label(x, eps=1e-9):
            if x > eps: return "improved"
            if x < -eps: return "declined"
            return "flat"
        delta_df["perf_dir"] = delta_df["Δ_performance"].apply(_label)

    # Define transition key strings like "0→1", "1→2", ...
    delta_df["transition_pair"] = delta_df.apply(
        lambda r: f"{int(r['session_earlier'])}→{int(r['session_later'])}", axis=1
    )

    # ---------- 1) Per-transition correlations ----------
    corr_by_transition = {}
    for tp, sub in delta_df.groupby("transition_pair"):
        if sub.empty or "Δ_performance" not in sub.columns:
            corr_by_transition[tp] = pd.DataFrame(columns=["feature","corr_with_Δperf"])
            continue
        dcols = [c for c in sub.columns if c.startswith("Δ_")]
        if not dcols:
            corr_by_transition[tp] = pd.DataFrame(columns=["feature","corr_with_Δperf"])
            continue
        # Spearman correlations
        corr_col = sub[dcols].corr(numeric_only=True, method="spearman").get("Δ_performance")
        if corr_col is None:
            corr_by_transition[tp] = pd.DataFrame(columns=["feature","corr_with_Δperf"])
            continue
        corrs = corr_col.dropna().sort_values(ascending=False)
        corr_by_transition[tp] = (
            corrs.rename("corr_with_Δperf").reset_index().rename(columns={"index":"feature"})
        )

    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: correlations with $\\Delta$performance by fixed session transitions\n\n")
        for tp, tbl in corr_by_transition.items():
            df = tbl.copy()
            if df.empty:
                continue
            col = _detect_corr_col_trans(df)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df[col].notna()].copy()
            if abs_thresh_single is not None:
                df = df.assign(abs_corr=df[col].abs())
                df = df[df["abs_corr"] >= abs_thresh_single].drop(columns="abs_corr")
            f.write(f"%% -------- Transition {tp} --------\n")
            f.write(df.to_latex(
                index=False, float_format="%.3f",
                caption=f"Fixed session transition {tp}: correlations with $\\Delta$performance ({note_excluded})",
                label=f"tab:fixed_trans_corr_{_labelify_transition_pair(*map(int, tp.split('→')))}"
            ))
            f.write("\n\n")

    # ---------- 2) ALL-transitions meta (shared/dissimilar across transitions) ----------
    long_rows = []
    for tp, tbl in corr_by_transition.items():
        df = tbl.copy()
        if df.empty: 
            continue
        col = _detect_corr_col_trans(df)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({"transition": tp, "feature": getattr(r, "feature"), "corr": getattr(r, col)})

    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return

    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()
    meta_df = longdf.copy()
    if abs_thresh_single is not None:
        meta_df = meta_df[meta_df["abs_corr"] >= abs_thresh_single]

    agg = (meta_df.groupby("feature")
                 .agg(n_trans=("transition","nunique"),
                      mean_corr=("corr","mean"),
                      mean_abs=("abs_corr","mean"),
                      min_abs=("abs_corr","min"),
                      max_abs=("abs_corr","max"),
                      min_corr=("corr","min"),
                      max_corr=("corr","max"))
                 .reset_index())

    shared_all = agg.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k_meta).copy()
    agg["range_corr"] = agg["max_corr"] - agg["min_corr"]
    dissim_all = agg.sort_values("range_corr", ascending=False).head(top_k_meta).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    with open(out_tex, "a") as f:
        f.write("%% -------- ALL TRANSITIONS: Top shared features --------\n")
        f.write(shared_all[["feature","n_trans","min_abs","mean_abs","max_abs","min_corr","max_corr"]].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k_meta} features shared across fixed transitions "
                     f"(ranked by minimum |r| across transitions; {note_excluded})."),
            label="tab:fixed_trans_shared_top"
        ))
        f.write("\n\n")

        f.write("%% -------- ALL TRANSITIONS: Top dissimilar features --------\n")
        f.write(dissim_all[["feature","n_trans","range_corr","min_corr","max_corr"]].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k_meta} most dissimilar features across fixed transitions "
                     f"(ranked by correlation range; {note_excluded})."),
            label="tab:fixed_trans_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------- 3) Pairwise transition comparisons ----------
    transitions = sorted(longdf["transition"].unique().tolist())

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN FIXED TRANSITIONS --------\n\n")

    # Pre-filter per transition for pairwise logic (strong in each)
    per_trans_df = {}
    for T in transitions:
        sub = longdf[longdf["transition"] == T].copy()
        sub = sub[sub["abs_corr"] >= abs_thresh_pair]
        per_trans_df[T] = sub.set_index("feature")[["corr","abs_corr"]]

    for A, B in itertools.combinations(transitions, 2):
        A_df = per_trans_df[A]
        B_df = per_trans_df[B]
        if A_df.empty or B_df.empty:
            continue

        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        
        inter = inter[(inter["abs_corr_A"] >= strength_thresh) | 
              (inter["abs_corr_B"] >= strength_thresh)]
        
        if inter.empty:
            continue

        inter["sign_A"] = inter["corr_A"].apply(lambda x: 1 if x >= 0 else -1)
        inter["sign_B"] = inter["corr_B"].apply(lambda x: 1 if x >= 0 else -1)
        inter["same_sign"] = (inter["sign_A"] == inter["sign_B"])
        inter["delta_r"] = (inter["corr_B"] - inter["corr_A"]).abs()
        inter["min_abs"] = inter[["abs_corr_A","abs_corr_B"]].min(axis=1)

        shared = inter[inter["same_sign"]].sort_values(
            ["delta_r", "min_abs"], ascending=[True, False]
        ).head(top_k_pair).copy()

        dissim_mask = (~inter["same_sign"]) | (inter["delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(
            ["delta_r","min_abs"], ascending=[False, False]
        ).head(top_k_pair).copy()

        shared_tbl = shared.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r"]].copy()
        shared_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|"}, inplace=True)
        dissim_tbl = dissim.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r","same_sign"]].copy()
        dissim_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|", "same_sign":"same_sign"}, inplace=True)

        for df_ in (shared_tbl, dissim_tbl):
            for c in df_.select_dtypes(include="number").columns:
                df_[c] = df_[c].round(3)

        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign, strong in both) --------\n")
            f.write(shared_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Fixed transitions {A} vs {B}: shared strong correlations "
                         f"(|r|≥{abs_thresh_pair}, same sign; ranked by min|r|; {note_excluded})."),
                label=f"tab:fixed_trans_pair_shared_{_labelify_transition_pair(*map(int, A.split('→')))}_vs_{_labelify_transition_pair(*map(int, B.split('→')))}"
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Fixed transitions {A} vs {B}: dissimilar strong correlations "
                         f"(|r|≥{abs_thresh_pair} in both; opposite sign or |Δr|≥{diff_thresh_pair}; {note_excluded})."),
                label=f"tab:fixed_trans_pair_dissimilar_{_labelify_transition_pair(*map(int, A.split('→')))}_vs_{_labelify_transition_pair(*map(int, B.split('→')))}"
            ))
            f.write("\n\n")

    # ---------- 4) Per-transition Top-k improved vs declined ----------
    # Compute means by perf_dir per transition, then contrast = improved − declined
    with open(out_tex, "a") as f:
        f.write("%% -------- PER-TRANSITION: Top-k improved vs declined (Δfeature means) --------\n\n")

    for tp, sub in delta_df.groupby("transition_pair"):
        dcols = [c for c in sub.columns if c.startswith("Δ_")]
        if not dcols or "perf_dir" not in sub.columns:
            continue
        agg = (sub.groupby("perf_dir")[dcols].mean().reset_index())
        labels = set(agg["perf_dir"].astype(str))
        if not {"improved","declined"}.issubset(labels):
            continue
        imp = agg[agg["perf_dir"]=="improved"].drop(columns=["perf_dir"], errors="ignore")
        dec = agg[agg["perf_dir"]=="declined"].drop(columns=["perf_dir"], errors="ignore")
        common = [c for c in imp.columns if c in dec.columns]
        if not common:
            continue
        contrast_df = pd.DataFrame((imp[common].values - dec[common].values), columns=common)
        contrast_df.insert(0, "contrast", "improved_minus_declined")

        col, top_impr, top_decl = _top_k_pos_neg(contrast_df, top_k_contrast)

        with open(out_tex, "a") as f:
            f.write(f"%% -------- Transition {tp}: Top-k improved vs declined --------\n")
            f.write(top_impr.to_latex(
                index=False, float_format="%.3f",
                caption=f"Transition {tp}: top {top_k_contrast} factors when performance improved (highest positive {col}).",
                label=f"tab:fixed_trans_{_labelify_transition_pair(*map(int, tp.split('→')))}_topk_improved"
            ))
            f.write("\n\n")
            f.write(top_decl.to_latex(
                index=False, float_format="%.3f",
                caption=f"Transition {tp}: top {top_k_contrast} factors when performance declined (most negative {col}).",
                label=f"tab:fixed_trans_{_labelify_transition_pair(*map(int, tp.split('→')))}_topk_declined"
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (fixed transitions: per-transition, meta, pairwise, top-k contrasts) to: {out_tex}")

def build_features(df_sessions):
    """
    Reuses your existing `session_summary(row)` to create a tidy feature table.
    Requires columns: team_id, session, leader_type, performance (+ dict fields).
    """
    if 'leader_type' not in df_sessions.columns:
        raise ValueError("df_sessions must include 'leader_type'.")
    df = df_sessions.copy()
    df['leader_type'] = df['leader_type'].astype(str).str.lower().map(
        lambda s: 'human' if 'human' in s else ('ai' if 'ai' in s else s)
    )
    feats = df.apply(session_summary, axis=1)  # <-- your function from before
    out = pd.concat([df[['team_id','session','leader_type','performance','layout','suggestions',"name"]].reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)
    return out
# -----------------------------------------------------------------------------


def summarize_up_vs_down(delta_df, by_leadership=True, same_leadership_only=False):
    """
    Produces tables of average Δfeatures for Δperformance>0 vs Δperformance<0.
    If by_leadership=True, returns dict with 'human' and 'ai' tables; else returns one table.
    """
    df = delta_df.copy()
    df = df.dropna(subset=['Δ_performance'])
    # label improves / declines
    df['perf_dir'] = np.where(df['Δ_performance']>0, 'improved',
                        np.where(df['Δ_performance']<0, 'declined', 'flat'))
    # we’ll ignore exact zeros to make contrasts cleaner
    df = df[df['perf_dir']!='flat']

    delta_cols = [c for c in df.columns if c.startswith('Δ_')]

    def _agg(tbl):
        return (tbl
                .groupby('perf_dir')[delta_cols]
                .mean()
                .rename_axis(None)
                .reset_index())

    if by_leadership:
        out = {}
        for lt in ['human','ai']:
            sub = df[df['leader_type']==lt]
            if sub.empty:
                out[lt] = pd.DataFrame(columns=['perf_dir']+delta_cols)
            else:
                out[lt] = _agg(sub)
        return out
    else:
        return _agg(df)


def diff_improved_minus_declined(summary_tbl):
    """
    Convenience: turn the 'improved vs declined' table into a single-row
    'improved - declined' contrast for easier reading per leadership.
    """
    if summary_tbl.empty or 'perf_dir' not in summary_tbl.columns:
        return pd.DataFrame()
    if not set(['improved','declined']).issubset(set(summary_tbl['perf_dir'])):
        return pd.DataFrame()
    imp = summary_tbl[summary_tbl['perf_dir']=='improved'].drop(columns=['perf_dir']).copy()
    dec = summary_tbl[summary_tbl['perf_dir']=='declined'].drop(columns=['perf_dir']).copy()
    contrast = (imp.values - dec.values)
    out = pd.DataFrame(contrast, columns=imp.columns)
    out.insert(0, 'contrast', ['improved_minus_declined'])
    return out


def correlations_with_dperf(delta_df, by_leadership=True):
    """
    Simple Pearson correlations between Δperformance and each Δfeature.
    Returns one table (overall) or dict by leadership type.
    """
    df = delta_df.dropna(subset=['Δ_performance']).copy()
    delta_cols = [c for c in df.columns if c.startswith('Δ_')]

    def _corr_tbl(tbl):
        if tbl.empty: 
            return pd.DataFrame(columns=['feature','corr_with_Δperf'])
        corrs = tbl[delta_cols].corr(numeric_only=True, method='spearman')['Δ_performance'].dropna().sort_values(ascending=False)
        return corrs.rename('corr_with_Δperf').reset_index().rename(columns={'index':'feature'})

    if by_leadership:
        out = {}
        for lt in ['human','ai']:
            sub = df[df['leader_type']==lt]
            out[lt] = _corr_tbl(sub)
        return out
    else:
        return _corr_tbl(df)


def _delta_between(later_row: pd.Series, earlier_row: pd.Series, id_cols=('team_id','session','leader_type')):
    """
    Compute Δ for all numeric columns between two rows (later - earlier).
    Returns a single-row DataFrame with id columns taken from 'later' plus Δ_* columns.
    """
    later = later_row.copy()
    earlier = earlier_row.copy()
    # Identify numeric columns to diff (exclude ids)
    num_cols = [c for c in later.index if c not in id_cols and pd.api.types.is_numeric_dtype(type(later[c]))]
    out = {k: later[k] for k in id_cols}
    for c in num_cols:
        try:
            out[f"Δ_{c}"] = float(later[c]) - float(earlier[c])
        except Exception:
            out[f"Δ_{c}"] = np.nan
    return pd.DataFrame([out])


def build_deltas_prev(feat_df: pd.DataFrame, same_leadership_only: bool = False) -> pd.DataFrame:
    """
    Non-contiguous friendly 'previous' deltas:
    For each (team[, leader]) row, compute difference against the MOST RECENT PRIOR row
    in that group (by session). If none exists, Δ_* are NaN for that row.

    same_leadership_only=False: group by team only.
    same_leadership_only=True : group by (team, leader_type).
    """
    keys = ['team_id'] if not same_leadership_only else ['team_id','leader_type']
    df = feat_df.sort_values(keys + ['session']).copy()

    # shift within group gives you the most recent prior matching those keys (non-contiguous globally is fine)
    grp = df.groupby(keys, group_keys=False)
    prev = grp.shift(1)

    # Build deltas row-wise
    delta_rows = []
    for idx in df.index:
        later_row = df.loc[idx]
        earlier_row = prev.loc[idx]
        if earlier_row.isna().all():
            # No previous in this group
            # Still return row with ids and Δ_performance (and others) = NaN
            delta_rows.append(_delta_between(later_row, later_row).assign(**{k: np.nan for k in list(_ for _ in _delta_between(later_row, later_row).columns if _.startswith('Δ_'))}).iloc[[0]])
        else:
            delta_rows.append(_delta_between(later_row, earlier_row).assign(gap_sessions = later_row['session'] - earlier_row['session']).iloc[[0]])

    delta_df = pd.concat(delta_rows, ignore_index=True)
    # keep id cols + Δ_* (and gap if present)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]


def build_deltas_all_pairs(feat_df: pd.DataFrame, same_leadership_only: bool = False) -> pd.DataFrame:
    """
    Exhaustive all-pairs deltas:
    For each team (and optionally within each leader type), compute differences for ALL pairs
    (earlier -> later), regardless of contiguity. Adds gap_sessions = later - earlier.

    Warning: grows O(n^2) per group; fine for 4 sessions but be mindful if larger.
    """
    keys = ['team_id'] if not same_leadership_only else ['team_id','leader_type']
    rows = []
    for _, g in feat_df.sort_values(keys + ['session']).groupby(keys):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        for i, j in combinations(range(len(g)), 2):
            earlier = g.loc[i]
            later   = g.loc[j]
            d = _delta_between(later, earlier)
            d['gap_sessions'] = later['session'] - earlier['session']
            rows.append(d)
    if not rows:
        # empty
        cols = ['team_id','session','leader_type','gap_sessions']
        return pd.DataFrame(columns=cols)
    delta_df = pd.concat(rows, ignore_index=True)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]


def raw_delta_descriptives_flexible(df_sessions: pd.DataFrame,
                                    pair_mode: str = "prev",
                                    same_leadership_only: bool = False,
                                    keep_flats: bool = False,
                                    eps: float = 1e-9):
    """
    Flexible wrapper:
      - pair_mode: "prev" for most-recent prior deltas (non-contiguous OK),
                   "all_pairs" for exhaustive all-pair deltas within group.
      - same_leadership_only: if True, compare only within same leader type.
      - keep_flats: if True, keep Δ_performance ~ 0 rows; else drop to sharpen contrasts.
    """
    feat_df = build_features(df_sessions)

    # ---------------------------
    # Build delta_df by pair_mode
    # ---------------------------
    if pair_mode == "prev":
        # existing behavior: previous-within team or team+leader
        delta_df = build_deltas_prev(feat_df, same_leadership_only=same_leadership_only)

    elif pair_mode == "all_pairs":
        # existing behavior: all pairs within team or team+leader
        delta_df = build_deltas_all_pairs(feat_df, same_leadership_only=same_leadership_only)

    elif pair_mode == "fixed":
        # NEW: only (0→1), (1→2), (2→3), (0→3) for each team
        rows = []
        # we will optionally check same-leadership after picking pairs
        wanted_pairs = [(0,1), (1,2), (2,3), (0,3)]
        for tid, g in feat_df.groupby("team_id"):
            g = g.sort_values("session").reset_index(drop=True)
            # build dict session_index -> row for quick access
            sess_to_row = {int(g.loc[i, "session"]): g.loc[i] for i in g.index}
            for i, j in wanted_pairs:
                if i in sess_to_row and j in sess_to_row:
                    earlier = sess_to_row[i]
                    later   = sess_to_row[j]
                    # filter by same leadership if requested
                    if same_leadership_only and (earlier["leader_type"] != later["leader_type"]):
                        continue
                    # compute deltas
                    num_cols = [c for c in feat_df.columns if c not in ["team_id","session","leader_type",'layout','suggestions',"name"]]
                    out = {
                        "team_id": earlier["team_id"],
                        "leader_type": later["leader_type"],  # leader at later session (for summaries)
                        "session_earlier": int(earlier["session"]),
                        "session_later": int(later["session"]),
                        "gap_sessions": int(later["session"] - earlier["session"]),
                    }
                    for c in num_cols:
                        try:
                            out[f"Δ_{c}"] = (later[c] - earlier[c]) if pd.notna(later[c]) and pd.notna(earlier[c]) else np.nan
                        except:
                            pdb.set_trace()
                    rows.append(out)
        delta_df = pd.DataFrame(rows)

    else:
        raise ValueError("pair_mode must be 'prev', 'all_pairs', or 'fixed'")

    # ----------------------------------
    # Label direction and (optionally) drop flats
    # ----------------------------------
    if 'Δ_performance' in delta_df.columns:
        def _label(x):
            if x > eps: return 'improved'
            if x < -eps: return 'declined'
            return 'flat'
        delta_df['perf_dir'] = delta_df['Δ_performance'].apply(_label)
        if not keep_flats:
            delta_df = delta_df[delta_df['perf_dir']!='flat'].copy()


    # ---------------------------
    # Build summaries per leader (robust)
    # ---------------------------
    def _agg(tbl):
        """Average of Δ_* by perf_dir; always returns a DataFrame with a 'perf_dir' column (may be empty)."""
        if tbl.empty or 'perf_dir' not in tbl.columns:
            return pd.DataFrame(columns=['perf_dir'])
        delta_cols = [c for c in tbl.columns if c.startswith('Δ_')]
        if not delta_cols:
            return pd.DataFrame(columns=['perf_dir'])
        out = (tbl.groupby('perf_dir')[delta_cols]
                  .mean()
                  .reset_index())
        # Ensure column present even if no groups materialize
        if 'perf_dir' not in out.columns:
            out.insert(0, 'perf_dir', pd.Series(dtype=object))
        return out

    updown_by_leader, contrast_by_leader, corr_by_leader = {}, {}, {}

    for lt in ['human','ai']:
        sub = delta_df[delta_df['leader_type'] == lt].copy()
        up = _agg(sub)
        updown_by_leader[lt] = up

        # Contrast improved - declined (guarded)
        if (not up.empty) and ('perf_dir' in up.columns):
            labels = set(up['perf_dir'].astype(str).tolist())
            if {'improved', 'declined'}.issubset(labels):
                imp = up[up['perf_dir'] == 'improved'].drop(columns=['perf_dir'], errors='ignore')
                dec = up[up['perf_dir'] == 'declined'].drop(columns=['perf_dir'], errors='ignore')
                common = [c for c in imp.columns if c in dec.columns]
                con_df = (pd.DataFrame((imp[common].values - dec[common].values), columns=common)
                          if common else pd.DataFrame())
                if not con_df.empty:
                    con_df.insert(0, 'contrast', 'improved_minus_declined')
            else:
                con_df = pd.DataFrame()
        else:
            con_df = pd.DataFrame()
        contrast_by_leader[lt] = con_df

        # Correlations with Δperformance (guarded)
        if (not sub.empty) and ('Δ_performance' in sub.columns):
            dcols = [c for c in sub.columns if c.startswith('Δ_')]
            if dcols:
                corr_col = sub[dcols].corr(numeric_only=True, method='spearman').get('Δ_performance')
                if corr_col is not None:
                    corrs = corr_col.dropna().sort_values(ascending=False)
                    corr_tbl = (corrs.rename('corr_with_Δperf')
                                     .reset_index()
                                     .rename(columns={'index': 'feature'}))
                else:
                    corr_tbl = pd.DataFrame(columns=['feature','corr_with_Δperf'])
            else:
                corr_tbl = pd.DataFrame(columns=['feature','corr_with_Δperf'])
        else:
            corr_tbl = pd.DataFrame(columns=['feature','corr_with_Δperf'])
        corr_by_leader[lt] = corr_tbl

    return {
        'feat_df': feat_df,
        'delta_df': delta_df,
        'updown_by_leader': updown_by_leader,
        'contrast_by_leader': contrast_by_leader,
        'corr_by_leader': corr_by_leader
    }
    



def build_deltas_prev_samelead(feat_df: pd.DataFrame) -> pd.DataFrame:
    keys = ['team_id','leader_type']
    df = feat_df.sort_values(keys + ['session']).copy()
    grp = df.groupby(keys, group_keys=False)
    prev = grp.shift(1)
    rows = []
    for idx in df.index:
        later = df.loc[idx]; earlier = prev.loc[idx]
        if earlier.isna().all():
            d = _delta_between(later, later);  # scaffold ids
            for c in list(d.columns):
                if c.startswith('Δ_'): d[c] = np.nan
        else:
            d = _delta_between(later, earlier)
            d['gap_sessions'] = later['session'] - earlier['session']
        rows.append(d)
    delta_df = pd.concat(rows, ignore_index=True)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]

def build_deltas_allpairs_samelead(feat_df: pd.DataFrame) -> pd.DataFrame:
    keys = ['team_id','leader_type']
    rows = []
    for _, g in feat_df.sort_values(keys + ['session']).groupby(keys):
        g = g.reset_index(drop=True)
        for i, j in combinations(range(len(g)), 2):
            earlier, later = g.loc[i], g.loc[j]
            d = _delta_between(later, earlier)
            d['gap_sessions'] = later['session'] - earlier['session']
            rows.append(d)
    if not rows:
        return pd.DataFrame(columns=['team_id','session','leader_type','gap_sessions'])
    delta_df = pd.concat(rows, ignore_index=True)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]


SIGN_HELP_BY_LEADER = {
    "human": {
        "Δ_precision": +1, "Δ_benign_ratio": -1,
        "Δ_danger_weight_coverage": +1, "Δ_dangerous_weighted_distance": +1,
        "Δ_median_sens_danger_carried": +1, "Δ_pct_danger_carried_ge2": +1,
        "Δ_misunderstanding_rate": -1, "Δ_change_rate": -1,
        "Δ_completion_rate": +1, "Δ_confirmation_rate": +1,
        "Δ_latency_norm_median": -1, "Δ_path_eff_median": -1, "Δ_mobility_gini": -1,
        "Δ_incomplete_order_rate": -1, "Δ_underspecified_rate": -1,
        # sensing mix: human-specific
        "Δ_sensed_2": -1, "Δ_sensed_3": +1,
        "Δ_sensed_4": 0, "Δ_sensed_1": 0, "Δ_sensed_0": 0,
        # leave survey/UX items neutral (0) unless you have theory
    },
    "ai": {
        "Δ_precision": +1, "Δ_benign_ratio": -1,
        "Δ_danger_weight_coverage": +1, "Δ_dangerous_weighted_distance": +1,
        "Δ_median_sens_danger_carried": +1, "Δ_pct_danger_carried_ge2": +1,
        "Δ_misunderstanding_rate": -1, "Δ_change_rate": -1,
        "Δ_completion_rate": +1, "Δ_confirmation_rate": +1,
        "Δ_latency_norm_median": -1, "Δ_path_eff_median": -1, "Δ_mobility_gini": -1,
        "Δ_incomplete_order_rate": -1, "Δ_underspecified_rate": -1,
        # sensing mix: keep neutral (low/unstable effects in report)
        "Δ_sensed_2": 0, "Δ_sensed_3": 0, "Δ_sensed_4": 0, "Δ_sensed_1": 0, "Δ_sensed_0": 0,
    }
}

# Optionally zero out tiny-importance features (absolute corr < threshold)
MIN_ABS_CORR = 0.15  # tune if you like

def importance_from_correlations(delta_df: pd.DataFrame, leader_type: str) -> dict:
    sub = delta_df[(delta_df['leader_type']==leader_type)].dropna(subset=['Δ_performance']).copy()
    if sub.empty:
        return {}
    delta_cols = [c for c in sub.columns if c.startswith('Δ_')]
    corrs = sub[delta_cols].corr(numeric_only=True, method='spearman')['Δ_performance'].dropna()
    # keep only signals above threshold; otherwise weight=0 (ignored)
    imp = {k: (abs(v) if abs(v) >= MIN_ABS_CORR else 0.0) for k,v in corrs.items()}
    return imp

def explain_transition(delta_row: pd.Series, importance: dict, top_k: int = 3, leader_type: str = None):
    items = []
    for feat, val in delta_row.items():
        if not (feat.startswith('Δ_') and feat != 'Δ_performance'): 
            continue
        w = importance.get(feat, 0.0)
        score = abs(val) * w
        if np.isfinite(val) and np.isfinite(score):
            items.append((feat, float(val), score))
    items.sort(key=lambda x: x[2], reverse=True)
    top = items[:max(top_k, 0)]

    # leader-specific sign map
    sign_map = SIGN_HELP_BY_LEADER.get(leader_type, {})
    dperf = float(delta_row.get('Δ_performance', np.nan))
    improving, worsening = [], []
    for f, v, s in top:
        pref = sign_map.get(f, 0)
        if pref == 0:
            # fall back: align with Δperformance sign
            helped = (np.sign(v) == np.sign(dperf))
        else:
            helped = (np.sign(v) == np.sign(pref))
        (improving if helped else worsening).append(f"{f} ({v:+.3f})")

    return {
        "top_contributors": [f"{f} ({v:+.3f})" for f,v,_ in top],
        "top_improving": improving,
        "top_worsening": worsening
    }

# ---------- Orchestrator: build explanations for SAME LEADERSHIP only ----------
def per_team_explanations_same_leadership(df_sessions: pd.DataFrame,
                                          pair_mode: str = "prev",
                                          top_k: int = 3,
                                          use_sign_map: bool = True):
    """
    pair_mode: "prev" (most recent prior within same leader) OR "all_pairs" (every earlier→later same leader)
    Returns a tidy DataFrame where each row is a same-leadership transition with explanation fields.
    """
    feat_df = build_features(df_sessions)
    

    if pair_mode == "prev":
        delta_df = build_deltas_prev_samelead(feat_df)
    elif pair_mode == "all_pairs":
        delta_df = build_deltas_allpairs_samelead(feat_df)
    else:
        raise ValueError("pair_mode must be 'prev' or 'all_pairs'")

    # Drop rows with no Δ_performance
    delta_df = delta_df.dropna(subset=['Δ_performance']).copy()

    # Build importance per leadership
    importances = {lt: importance_from_correlations(delta_df, lt) for lt in ['human','ai']}

    # Build explanations per row
    rows = []
    for _, r in delta_df.iterrows():
        lt = r['leader_type']
        imp = importances.get(lt, {})
        expl = explain_transition(r, imp, top_k=top_k, leader_type=lt)
        
        rows.append({
            "team_id": r['team_id'],
            "leader_type": lt,
            "session_later": r['session'],
            "gap_sessions": r.get('gap_sessions', np.nan),
            "Δ_performance": r['Δ_performance'],
            "top_contributors": "; ".join(expl["top_contributors"]),
            "top_improving": "; ".join(expl["top_improving"]),
            "top_worsening": "; ".join(expl["top_worsening"]),
        })
    return pd.DataFrame(rows)

# ---------- Minimal print helper ----------
def print_team_explanations_samelead(df_sessions, pair_mode="prev", top_k=3, export_csv=None):
    tbl = per_team_explanations_same_leadership(df_sessions, pair_mode=pair_mode, top_k=top_k)
    # Show all columns fully
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    print(tbl.sort_values(['leader_type','team_id','session_later']).reset_index(drop=True))
    if export_csv:
        tbl.to_csv(export_csv, index=False)
        print(f"\nSaved to {export_csv}")

def correlations_per_layout_no_transitions(
    df_sessions: pd.DataFrame,
    layout_col: str = 'layout',
    exclude_patterns = (
        r"^dangerous_?weighted_?distance$",
        r"^dangerous_?weithed_?distance$",  # common typo
    ),
    abs_threshold: float = None,  # e.g., 0.3 to keep only |r|>0.3
    min_rows: int = 2
):
    """
    Session-level (no deltas). For each layout, correlate performance with every numeric feature.
    Returns: {'feat_df': DataFrame, 'corr_by_layout': {layout_value: DataFrame(feature, corr_with_performance)}}
    """
    feat_df = build_features(df_sessions)
    if layout_col not in feat_df.columns:
        raise ValueError(f"Layout column '{layout_col}' not found in features table.")

    # Ensure performance is numeric
    if not pd.api.types.is_numeric_dtype(feat_df['performance']):
        feat_df['performance'] = pd.to_numeric(feat_df['performance'], errors='coerce')

    # Collect numeric features (exclude id-like cols)
    id_like = {'team_id','session','leader_type', layout_col}
    numeric_feats = [c for c in feat_df.columns
                     if c not in id_like and pd.api.types.is_numeric_dtype(feat_df[c])]

    # Exclude tautological features
    import re
    excl = re.compile("|".join(exclude_patterns), flags=re.IGNORECASE)
    numeric_feats = [c for c in numeric_feats if not excl.search(c)]
    corr_by_layout = {}
    for lv, sub in feat_df.groupby(layout_col):
        sub = sub.copy()
        if len(sub) < min_rows:
            corr_by_layout[lv] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        # Keep features that exist and are numeric in this subset AND have variance
        cols_here = []
        for c in numeric_feats:
            if c in sub.columns and pd.api.types.is_numeric_dtype(sub[c]):
                # drop zero-variance to avoid NaNs in corr
                if sub[c].nunique(dropna=True) > 1:
                    cols_here.append(c)

        if not cols_here or sub['performance'].nunique(dropna=True) <= 1:
            corr_by_layout[lv] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        # Build correlation matrix explicitly on [performance] + features
        cols = ['performance'] + cols_here
        mat = sub[cols].corr(numeric_only=True, method='spearman')

        # Guard against degenerate cases
        if 'performance' not in mat.columns or 'performance' not in mat.index:
            corr_by_layout[lv] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        # Get the single 'performance' column as a 1D Series (not a DataFrame)
        corr_s = mat.loc[cols_here, ['performance']].iloc[:, 0]  # <-- Series
        corr_s.name = 'corr_with_performance'

        # Now build a clean two-column table
        tbl = (corr_s.dropna()
                      .reset_index()
                      .rename(columns={'index': 'feature'}))

        # Rank by |r|
        tbl['abs_corr'] = tbl['corr_with_performance'].abs()
        tbl = tbl.sort_values('abs_corr', ascending=False).drop(columns=['abs_corr']).reset_index(drop=True)

        if abs_threshold is not None:
            tbl = tbl[tbl['corr_with_performance'].abs() > abs_threshold].reset_index(drop=True)

        corr_by_layout[lv] = tbl

    return {'feat_df': feat_df, 'corr_by_layout': corr_by_layout}


def write_per_layout_latex(
    corr_by_layout: dict,
    out_tex: str = "layout_session_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance"
):
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: session-level correlations with performance per layout\n\n")
        for lv, tbl in corr_by_layout.items():
            f.write(f"%% -------- Layout {lv} --------\n")
            f.write(tbl.to_latex(index=False, float_format="%.3f",
                                 caption=f"Session-level correlations with performance for layout {lv} ({note_excluded})",
                                 label=f"tab:session_corr_layout_{str(lv).replace(' ','_')}"))
            f.write("\n\n")
    print(f"Wrote LaTeX tables to: {out_tex}")
    
def _detect_corr_col(df: pd.DataFrame) -> str:
    preferred = ["corr_with_Δperf", "corr_with_perf", "corr", "r"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if c != "abs_corr" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric correlation column found.")

def write_per_layout_latex_with_meta(
    corr_by_layout: Dict[str, pd.DataFrame],
    out_tex: str = "layout_session_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance",
    top_k: int = 10,
    abs_thresh: Optional[float] = None,       # e.g., 0.30 per-layout filter for the single-layout tables & meta tables
    require_all_layouts: bool = True,
    min_layouts: Optional[int] = None,
    # NEW: pairwise comparison knobs
    do_pairwise: bool = True,
    abs_thresh_pair: float = 0.0,            # feature must be |r|>=this in BOTH layouts to be considered in a pair
    diff_thresh_pair: float = 0.40,           # if same sign but |Δr|>=this, we still call it "dissimilar"
    top_k_pair: int = 50,
    pretty_layout_names: Optional[Dict[str, str]] = None,
    strength_thresh: int = 0.4
):
    """
    Adds pairwise comparisons between layouts:
      For each pair (A,B), we produce:
        - 'Shared' table: features strong in both (|r|>=abs_thresh_pair) AND same sign; ranked by min(|r_A|,|r_B|)
        - 'Dissimilar' table: features strong in both but opposite sign OR |r_A - r_B| >= diff_thresh_pair; ranked by |r_A - r_B|

    Everything else (per-layout tables and cross-layout shared/dissimilar meta tables) stays as before.
    """
    # ---------- 1) Write the original per-layout tables ----------
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: session-level correlations with performance per layout\n\n")
        for lv, tbl in corr_by_layout.items():
            name = pretty_layout_names.get(lv, str(lv)) if pretty_layout_names else str(lv)
            df = tbl.copy()
            corr_col = _detect_corr_col(df)
            df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
            df = df[df[corr_col].notna()].copy()
            if abs_thresh is not None:
                df = df.assign(abs_corr=df[corr_col].abs())
                df = df[df["abs_corr"] >= abs_thresh].drop(columns="abs_corr")
            f.write(f"%% -------- Layout {name} --------\n")
            f.write(df.to_latex(index=False, float_format="%.3f",
                                caption=f"Session-level correlations with performance for layout {name} ({note_excluded})",
                                label=f"tab:session_corr_layout_{str(name).replace(' ','_')}"))
            f.write("\n\n")

    # ---------- 2) Build a long-form table across layouts (used for meta + pairwise) ----------
    long_rows = []
    for lv, tbl in corr_by_layout.items():
        name = pretty_layout_names.get(lv, str(lv)) if pretty_layout_names else str(lv)
        df = tbl.copy()
        corr_col = _detect_corr_col(df)
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
        df = df[df[corr_col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({
                "layout": name,
                "feature": getattr(r, "feature"),
                "corr": getattr(r, corr_col),
            })
    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return
    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()

    # ---------- 3) Cross-layout meta (shared/dissimilar across ALL layouts) ----------
    layouts = sorted(longdf["layout"].unique().tolist())
    nL = len(layouts)
    minL = nL if require_all_layouts else (min_layouts if min_layouts is not None else max(1, int(round(0.75*nL))))

    longdf_meta = longdf.copy()
    if abs_thresh is not None:
        longdf_meta = longdf_meta[longdf_meta["abs_corr"] >= abs_thresh]
    agg = (
        longdf_meta.groupby("feature")
                   .agg(n_layouts=("layout","nunique"),
                        mean_corr=("corr","mean"),
                        mean_abs=("abs_corr","mean"),
                        min_abs=("abs_corr","min"),
                        max_abs=("abs_corr","max"),
                        min_corr=("corr","min"),
                        max_corr=("corr","max"))
                   .reset_index()
    )
    kept = agg[agg["n_layouts"] >= minL].copy()
    if kept.empty:
        kept = agg.copy()

    shared_all = kept.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k).copy()
    kept["range_corr"] = kept["max_corr"] - kept["min_corr"]
    dissim_all = kept.sort_values("range_corr", ascending=False).head(top_k).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    shared_cols_all = ["feature","n_layouts","min_abs","mean_abs","max_abs","min_corr","max_corr"]
    dissim_cols_all = ["feature","n_layouts","range_corr","min_corr","max_corr"]

    with open(out_tex, "a") as f:
        f.write("%% -------- CROSS-LAYOUT (ALL): Top shared features --------\n")
        f.write(shared_all[shared_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} features shared across layouts (ranked by minimum |r| across layouts; {note_excluded})."),
            label="tab:layout_shared_top"
        ))
        f.write("\n\n")
        f.write("%% -------- CROSS-LAYOUT (ALL): Top dissimilar features --------\n")
        f.write(dissim_all[dissim_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} most dissimilar features across layouts (ranked by correlation range; {note_excluded})."),
            label="tab:layout_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------- 4) NEW: Pairwise comparisons between layouts ----------
    if not do_pairwise or len(layouts) < 2:
        print(f"Wrote LaTeX tables to: {out_tex}")
        return

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN LAYOUTS --------\n\n")

    # Convenience: pre-split per layout with abs-threshold applied for pairwise logic
    per_layout_df = {}
    for L in layouts:
        sub = longdf[longdf["layout"] == L].copy()
        sub = sub[sub["abs_corr"] >= abs_thresh_pair]
        per_layout_df[L] = sub.set_index("feature")[["corr","abs_corr"]]

    # For each pair (A,B): compute shared & dissimilar sets
    for A, B in itertools.combinations(layouts, 2):
        A_df = per_layout_df[A]
        B_df = per_layout_df[B]
        if A_df.empty or B_df.empty:
            continue
        # intersection: features strong in BOTH
        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        
        inter = inter[(inter["abs_corr_A"] >= strength_thresh) | 
              (inter["abs_corr_B"] >= strength_thresh)]
        
        if inter.empty:
            continue

        # Determine sign
        inter["sign_A"] = inter["corr_A"].apply(lambda x: 1 if x >= 0 else -1)
        inter["sign_B"] = inter["corr_B"].apply(lambda x: 1 if x >= 0 else -1)
        inter["same_sign"] = (inter["sign_A"] == inter["sign_B"])
        inter["delta_r"] = (inter["corr_B"] - inter["corr_A"]).abs()
        inter["min_abs"] = inter[["abs_corr_A","abs_corr_B"]].min(axis=1)

        # "Shared": strong in both, same sign; rank by min_abs
        shared = inter[inter["same_sign"]].sort_values(["delta_r", "min_abs"], ascending=[True, False]).head(top_k_pair).copy()

        # "Dissimilar": strong in both AND (opposite sign OR big magnitude gap)
        dissim_mask = (~inter["same_sign"]) | (inter["delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(["delta_r","min_abs"], ascending=[False, False]).head(top_k_pair).copy()

        # Select & round
        shared_tbl = shared.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r"]].copy()
        shared_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})", "min_abs":"min|r|", "delta_r":"|Δr|"}, inplace=True)
        dissim_tbl = dissim.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r","same_sign"]].copy()
        dissim_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})", "min_abs":"min|r|", "delta_r":"|Δr|", "same_sign":"same_sign"}, inplace=True)

        for df_ in (shared_tbl, dissim_tbl):
            for c in df_.select_dtypes(include="number").columns:
                df_[c] = df_[c].round(3)

        # Append to LaTeX
        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign, strong in both) --------\n")
            f.write(shared_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Layouts {A} vs {B}: shared strong correlations (|r|≥{abs_thresh_pair}, same sign; ranked by min|r|; {note_excluded})."),
                label=f"tab:layout_pair_shared_{A}_vs_{B}".replace(" ","_")
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Layouts {A} vs {B}: dissimilar strong correlations (|r|≥{abs_thresh_pair} in both; opposite sign or |Δr|≥{diff_thresh_pair}; {note_excluded})."),
                label=f"tab:layout_pair_dissimilar_{A}_vs_{B}".replace(" ","_")
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (including pairwise comparisons) to: {out_tex}")
    
def plot_layout_heatmap(corr_by_layout: dict, top_k: int = 15, out_png: str = "layout_corr_heatmap.png"):
    # Stack into one DataFrame
    stacks = []
    for lv, tbl in corr_by_layout.items():
        if tbl.empty: 
            continue
        t = tbl.copy()
        t['layout'] = lv
        stacks.append(t)
    if not stacks:
        print("No data for heatmap.")
        return
    allc = pd.concat(stacks, ignore_index=True)

    # pick top features by max |corr| across layouts
    top_feats = (allc.assign(abs_corr=lambda d: d['corr_with_performance'].abs())
                      .groupby('feature')['abs_corr'].max()
                      .sort_values(ascending=False)
                      .head(top_k).index.tolist())

    pivot = (allc[allc['feature'].isin(top_feats)]
         .pivot_table(index='feature',
                      columns='layout',
                      values='corr_with_performance',
                      aggfunc='mean')  # or 'max', 'first'
         .fillna(0.0))

    pivot.index = pivot.index.to_series().apply(lambda s: str(s).split()[0].split('_')[0])

    # simple matplotlib heatmap (no seaborn)
    fig_w = 1.0 * max(8, len(pivot.columns) * 1.5)
    fig_h = 0.6 * max(6, len(pivot.index) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect='auto', cmap='coolwarm', vmin=-1.0, vmax=1.0)

    # axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Session-level correlations with performance by layout (top {} features)".format(top_k))
    ax.set_xlabel("Layout")
    ax.set_ylabel("Feature")

    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', color='white' if abs(v) > 0.5 else 'black', fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (r)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved heatmap to: {out_png}")

def _to_binary_suggestions(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, np.integer)):
        return int(val) if val in (0,1) else np.nan
    if isinstance(val, float):
        return int(val) if val in (0.0, 1.0) else np.nan

    s = str(val).strip().lower()
    truthy = {"1","true","t","yes","y","on","with","enabled"}
    falsy  = {"0","false","f","no","n","off","without","disabled"}
    if s in truthy: return 1
    if s in falsy:  return 0

    # last-ditch attempt (e.g., "1.0")
    try:
        f = float(s)
        return int(f) if f in (0.0, 1.0) else np.nan
    except:
        return np.nan


def _detect_corr_col(df: pd.DataFrame) -> str:
    preferred = ["corr_with_Δperf", "corr_with_perf", "corr", "r"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if c != "abs_corr" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric correlation column found in provided DataFrame.")

def _key_to_label(k: Hashable) -> str:
    if isinstance(k, tuple):
        return "_".join(str(x) for x in k)
    return str(k)

def write_by_leadership_suggestions_latex_with_meta(
    corr_by_ls: Dict[Hashable, pd.DataFrame],
    out_tex: str = "lead_sugg_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance",
    top_k: int = 10,
    abs_thresh: Optional[float] = None,        # per-group filter for single-group & meta tables
    require_all_groups: bool = True,
    min_groups: Optional[int] = None,
    pretty_names: Optional[Dict[Hashable, str]] = None,
    # NEW: pairwise controls
    do_pairwise: bool = True,
    abs_thresh_pair: float = 0.40,             # strong in BOTH groups means |r|>=this in each
    diff_thresh_pair: float = 0.40,            # same sign but big magnitude gap counts as dissimilar if |Δr|>=this
    top_k_pair: int = 50,
    strength_thresh: int = 0.4
):
    """
    Writes:
      • One LaTeX table per (leadership × suggestions) group (as before)
      • Cross-group meta: top shared (min |r|) and top dissimilar (range of r) across ALL groups
      • NEW Pairwise tables for every group pair:
          - Shared strong correlations (same sign; |r|>=abs_thresh_pair in both)
          - Dissimilar strong correlations (opposite sign OR |Δr|>=diff_thresh_pair; |r|>=abs_thresh_pair in both)
    """
    # ---------- 1) Per-group tables ----------
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: correlations with performance by leadership × suggestions groups\n\n")
        for gk, tbl in corr_by_ls.items():
            label = pretty_names.get(gk, _key_to_label(gk)) if pretty_names else _key_to_label(gk)
            df = tbl.copy()
            corr_col = _detect_corr_col(df)
            df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
            df = df[df[corr_col].notna()].copy()
            if abs_thresh is not None:
                df = df.assign(abs_corr=df[corr_col].abs())
                df = df[df["abs_corr"] >= abs_thresh].drop(columns="abs_corr")
            f.write(f"%% -------- Group: {label} --------\n")
            f.write(df.to_latex(
                index=False, float_format="%.3f",
                caption=f"Correlations with performance — Group {label} ({note_excluded})",
                label=f"tab:lead_sugg_corr_{label.replace(' ','_').replace('/','-')}"
            ))
            f.write("\n\n")

    # ---------- 2) Long form across groups ----------
    long_rows = []
    for gk, tbl in corr_by_ls.items():
        label = pretty_names.get(gk, _key_to_label(gk)) if pretty_names else _key_to_label(gk)
        df = tbl.copy()
        corr_col = _detect_corr_col(df)
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
        df = df[df[corr_col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({
                "group": label,
                "feature": getattr(r, "feature"),
                "corr": getattr(r, corr_col),
            })
    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return
    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()
    groups = sorted(longdf["group"].unique().tolist())

    # ---------- 3) Cross-group meta (ALL groups) ----------
    longdf_meta = longdf.copy()
    if abs_thresh is not None:
        longdf_meta = longdf_meta[longdf_meta["abs_corr"] >= abs_thresh]
    nG = len(groups)
    minG = nG if require_all_groups else (min_groups if min_groups is not None else max(1, int(round(0.75*nG))))

    agg = (
        longdf_meta.groupby("feature")
                   .agg(n_groups=("group","nunique"),
                        mean_corr=("corr","mean"),
                        mean_abs=("abs_corr","mean"),
                        min_abs=("abs_corr","min"),
                        max_abs=("abs_corr","max"),
                        min_corr=("corr","min"),
                        max_corr=("corr","max"))
                   .reset_index()
    )
    kept = agg[agg["n_groups"] >= minG].copy()
    if kept.empty:
        kept = agg.copy()

    shared_all = kept.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k).copy()
    kept["range_corr"] = kept["max_corr"] - kept["min_corr"]
    dissim_all = kept.sort_values("range_corr", ascending=False).head(top_k).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    shared_cols_all = ["feature","n_groups","min_abs","mean_abs","max_abs","min_corr","max_corr"]
    dissim_cols_all = ["feature","n_groups","range_corr","min_corr","max_corr"]

    with open(out_tex, "a") as f:
        f.write("%% -------- CROSS-GROUP (ALL): Top shared features --------\n")
        f.write(shared_all[shared_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} features shared across leadership × suggestions groups "
                     f"(ranked by minimum |r| across groups; {note_excluded})."),
            label="tab:lead_sugg_shared_top"
        ))
        f.write("\n\n")

        f.write("%% -------- CROSS-GROUP (ALL): Top dissimilar features --------\n")
        f.write(dissim_all[dissim_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} most dissimilar features across leadership × suggestions groups "
                     f"(ranked by correlation range; {note_excluded})."),
            label="tab:lead_sugg_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------- 4) NEW: Pairwise comparisons between groups ----------
    if not do_pairwise or len(groups) < 2:
        print(f"Wrote LaTeX tables to: {out_tex}")
        return

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN GROUPS --------\n\n")

    # Pre-split per group with strength filter for pairwise logic
    per_group_df = {}
    for G in groups:
        sub = longdf[longdf["group"] == G].copy()
        sub = sub[sub["abs_corr"] >= abs_thresh_pair]
        per_group_df[G] = sub.set_index("feature")[["corr","abs_corr"]]

    for A, B in itertools.combinations(groups, 2):
        A_df = per_group_df[A]
        B_df = per_group_df[B]
        if A_df.empty or B_df.empty:
            continue

        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        
        inter = inter[(inter["abs_corr_A"] >= strength_thresh) | 
              (inter["abs_corr_B"] >= strength_thresh)]
              
        if inter.empty:
            continue

        inter["sign_A"] = inter["corr_A"].apply(lambda x: 1 if x >= 0 else -1)
        inter["sign_B"] = inter["corr_B"].apply(lambda x: 1 if x >= 0 else -1)
        inter["same_sign"] = (inter["sign_A"] == inter["sign_B"])
        inter["delta_r"] = (inter["corr_B"] - inter["corr_A"]).abs()
        inter["min_abs"] = inter[["abs_corr_A","abs_corr_B"]].min(axis=1)

        shared = inter[inter["same_sign"]].sort_values(
            ["delta_r", "min_abs"], ascending=[True, False]
        ).head(top_k_pair).copy()

        dissim_mask = (~inter["same_sign"]) | (inter["delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(
            ["delta_r","min_abs"], ascending=[False, False]
        ).head(top_k_pair).copy()

        shared_tbl = shared.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r"]].copy()
        shared_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|"}, inplace=True)
        dissim_tbl = dissim.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r","same_sign"]].copy()
        dissim_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|", "same_sign":"same_sign"}, inplace=True)

        for df_ in (shared_tbl, dissim_tbl):
            for c in df_.select_dtypes(include="number").columns:
                df_[c] = df_[c].round(3)

        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign, strong in both) --------\n")
            f.write(shared_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Groups {A} vs {B}: shared strong correlations (|r|≥{abs_thresh_pair}, same sign; ranked by min|r|; {note_excluded})."),
                label=f"tab:lead_sugg_pair_shared_{A}_vs_{B}".replace(" ","_").replace("/","-")
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Groups {A} vs {B}: dissimilar strong correlations (|r|≥{abs_thresh_pair} in both; opposite sign or |Δr|≥{diff_thresh_pair}; {note_excluded})."),
                label=f"tab:lead_sugg_pair_dissimilar_{A}_vs_{B}".replace(" ","_").replace("/","-")
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (including pairwise group comparisons) to: {out_tex}")
    
def correlations_by_leadership_and_suggestions(
    df_sessions: pd.DataFrame,
    abs_threshold: float = None,
    min_rows: int = 2
):
    feat_df = build_features(df_sessions).copy()

    # Normalize suggestions
    if 'suggestions' not in feat_df.columns:
        raise ValueError("df_sessions must include 'suggestions'.")
    feat_df['suggestions'] = feat_df['suggestions'].apply(_to_binary_suggestions)

    # Ensure performance numeric
    feat_df['performance'] = pd.to_numeric(feat_df['performance'], errors='coerce')

    # Numeric features only
    id_like = {'team_id','session','leader_type','layout','suggestions'}
    numeric_feats = [
        c for c in feat_df.columns
        if c not in id_like and pd.api.types.is_numeric_dtype(feat_df[c])
    ]

    results = {}
    for (lt, sug), sub in feat_df.groupby(['leader_type','suggestions'], dropna=False):
        if len(sub) < min_rows or sub['performance'].nunique(dropna=True) <= 1:
            results[(lt, sug)] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        cols_here = [c for c in numeric_feats if sub[c].nunique(dropna=True) > 1]
        if not cols_here:
            results[(lt, sug)] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        mat = sub[['performance'] + cols_here].corr(numeric_only=True, method='spearman')

        if 'performance' not in mat.columns:
            results[(lt, sug)] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        corrs = mat['performance'].drop('performance', errors='ignore')

        # make sure it's a Series
        if isinstance(corrs, pd.DataFrame):
            corrs = corrs.iloc[:,0]

        corrs = corrs.sort_values(ascending=False)

        if abs_threshold is not None:
            corrs = corrs[abs(corrs) >= abs_threshold]

        results[(lt, sug)] = (
            corrs.rename('corr_with_performance')
                 .reset_index()
                 .rename(columns={'index':'feature'})
        )

    return results



def top_changes_within_leader(DF, k=10):
    # 1 row per (team, leader, session)
    agg = (DF.sort_values(["team_id","leader_type","session"])
             .groupby(["team_id","leader_type","session"], as_index=False)
             .agg(performance=("performance","mean"),
                  name=("name", lambda s: s.iloc[0]),
                  layout=("layout", "first"),
                  suggestions=("suggestions","first")))
    # diffs within (team, leader)
    d = agg.sort_values(["team_id","leader_type","session"]).copy()
    d["perf_diff"] = d.groupby(["team_id","leader_type"])["performance"].diff()

    # previous fields (aligned within leader)
    for col in ["name","leader_type","session","performance","layout","suggestions"]:
        d[f"prev_{col}"] = d.groupby(["team_id","leader_type"])[col].shift(1)

    # keep real consecutive steps
    d = d[d["prev_session"].notna() & (d["session"] > d["prev_session"])]

    # build outputs
    cols = ["team_id","leader_type","prev_leader_type",
            "session","prev_session","name","prev_name",
            "performance","prev_performance","perf_diff","layout","prev_layout","suggestions","prev_suggestions"]
    inc = (d[d["perf_diff"] > 0][cols]
           .rename(columns={"leader_type":"curr_leader_type","session":"curr_session",
                            "performance":"curr_performance","name":"curr_name"})
           .sort_values("perf_diff", ascending=False).reset_index(drop=True))
    dec = (d[d["perf_diff"] < 0][cols]
           .rename(columns={"leader_type":"curr_leader_type","session":"curr_session",
                            "performance":"curr_performance","name":"curr_name"})
           .sort_values("perf_diff", ascending=True).reset_index(drop=True))
    print("\n=== WITHIN-LEADER: Top increases ===")
    print(inc.head(k).to_string(index=False))
    print("\n=== WITHIN-LEADER: Top decreases ===")
    print(dec.head(k).to_string(index=False))
    return inc, dec

# -----------------------------
# B) ACROSS-LEADER transitions
# -----------------------------
def top_changes_across_leader(DF, k=10):
    # 1 row per (team, session)
    agg = (DF.sort_values(["team_id","session"])
             .groupby(["team_id","session"], as_index=False)
             .agg(performance=("performance","mean"),
                  leader_type=("leader_type", lambda s: s.iloc[-1]),
                  name=("name", lambda s: s.iloc[-1]),
                  layout=("layout","last"),
                  suggestions=("suggestions","first")))
    d = agg.sort_values(["team_id","session"]).copy()
    d["perf_diff"] = d.groupby("team_id")["performance"].diff()

    # previous fields (aligned across any leader)
    for col in ["name","leader_type","session","performance","layout","suggestions"]:
        d[f"prev_{col}"] = d.groupby("team_id")[col].shift(1)

    # only consecutive steps; add transition label
    d = d[d["prev_session"].notna() & (d["session"] > d["prev_session"])].copy()
    d["transition"] = d["prev_leader_type"].astype(int).astype(str) + "→" + d["leader_type"].astype(int).astype(str)

    cols = ["team_id","transition",
            "leader_type","prev_leader_type",
            "session","prev_session","name","prev_name",
            "performance","prev_performance","perf_diff","layout","prev_layout","suggestions","prev_suggestions"]
    inc = (d[d["perf_diff"] > 0][cols]
           .rename(columns={"leader_type":"curr_leader_type","session":"curr_session",
                            "performance":"curr_performance","name":"curr_name"})
           .sort_values("perf_diff", ascending=False).reset_index(drop=True))
    dec = (d[d["perf_diff"] < 0][cols]
           .rename(columns={"leader_type":"curr_leader_type","session":"curr_session",
                            "performance":"curr_performance","name":"curr_name"})
           .sort_values("perf_diff", ascending=True).reset_index(drop=True))
    print("\n=== ACROSS-LEADER: Top increases ===")
    print(inc.head(k).to_string(index=False))
    print("\n=== ACROSS-LEADER: Top decreases ===")
    print(dec.head(k).to_string(index=False))
    return inc, dec


def team_perf_range_with_context(DF, k=10):
    # Aggregate at (team, leader) level to find min/max performance and keep indices
    agg = (DF.sort_values(["team_id","leader_type","session"])
             .groupby(["team_id","leader_type","session"], as_index=False)
             .agg(performance=("performance","mean"),
                  name=("name", lambda s: s.iloc[0]),
                  layout=("layout", "first"),
                  suggestions=("suggestions","first")))

    results = []

    for (tid, lid), group in agg.groupby(["team_id","leader_type"]):
        # Find row with min and row with max
        idx_min = group["performance"].idxmin()
        idx_max = group["performance"].idxmax()
        row_min = agg.loc[idx_min]
        row_max = agg.loc[idx_max]

        # Build a combined row with prev/curr semantics
        out = {
            "team_id": tid,
            "leader_type": lid,
            "prev_leader_type": row_min["leader_type"],
            "curr_leader_type": row_max["leader_type"],
            "prev_session": row_min["session"],
            "curr_session": row_max["session"],
            "prev_name": row_min["name"],
            "curr_name": row_max["name"],
            "prev_performance": row_min["performance"],
            "curr_performance": row_max["performance"],
            "perf_diff": row_max["performance"] - row_min["performance"],
            "prev_layout": row_min["layout"],
            "curr_layout": row_max["layout"],
            "prev_suggestions": row_min["suggestions"],
            "curr_suggestions": row_max["suggestions"]
        }
        results.append(out)

    df_out = pd.DataFrame(results).sort_values("perf_diff", ascending=False).reset_index(drop=True)

    print("\n=== Max−Min performance difference per team × leader ===")
    print(df_out.head(k).to_string(index=False))
    return df_out
    
def team_first_last_diff(DF, k=10):
    """
    For each team_id:
      1) Aggregate performance per session (mean if multiple rows).
      2) Take the first and last sessions (by sorted `session`).
      3) Compute diff = last_performance - first_performance.
    Returns a DataFrame sorted by diff (desc) and prints top-k.
    """
    # 1) Aggregate at (team, session) level
    agg = (
        DF.sort_values(["team_id", "session"])
          .groupby(["team_id", "session"], as_index=False)
          .agg(
              performance=("performance", "mean"),
              name=("name", lambda s: s.iloc[0]),
              # Keep helpful context; be robust to multiple values within a session
              layout=("layout", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
              leader_types=("leader_type", lambda s: ",".join(pd.Series(s).astype(str).unique())),
              suggestions=("suggestions", lambda s: pd.Series(s).mode().iloc[0] if not pd.Series(s).mode().empty else s.iloc[0]),
          )
    )

    results = []
    for tid, group in agg.groupby("team_id", as_index=False):
        group = group.sort_values("session", ascending=True)

        first = group.iloc[0]
        last  = group.iloc[-1]

        results.append({
            "team_id": tid,
            "first_session": first["session"],
            "last_session":  last["session"],
            "first_performance": first["performance"],
            "last_performance":  last["performance"],
            "perf_diff": last["performance"] - first["performance"],  # improvement (+ up)
            "first_leader_types": first["leader_types"],
            "last_leader_types":  last["leader_types"],
            "first_layout": first["layout"],
            "last_layout":  last["layout"],
            "first_suggestions": first["suggestions"],
            "last_suggestions":  last["suggestions"],
            "first_name": first["name"],
            "last_name":  last["name"],
        })

    df_out = (
        pd.DataFrame(results)
          .sort_values("perf_diff", ascending=False)
          .reset_index(drop=True)
    )

    print("\n=== Last − First performance difference per team ===")
    print(df_out.head(k).to_string(index=False))
    return df_out

def team_performance_profile(
    DF,
    k=10,
    human_label=0,
    ai_label=1,
):
    """
    Build a per-team profile including:
      - first vs last performance diff
      - mean and std across sessions
      - relative to global mean
      - plateau-drop detection
      - learning slope (linear regression of perf ~ session index)
      - NEW: human-led (2nd - 1st) performance diff
      - NEW: ai-led (2nd - 1st) performance diff

    Parameters
    ----------
    DF : pd.DataFrame with columns:
         team_id, session, performance, leader_type, layout, suggestions, name (optional)
    k : int
         number of rows to print in the headline view
    human_label, ai_label :
         values used in `leader_type` to denote human vs AI leadership
         (e.g., 0/1 or 'human'/'ai'). Must match DF['leader_type'] values.
    """

    # --- Aggregate performance per team × session
    # Keep a SINGLE representative leader_type/layout/suggestions per session (mode if available)
    def _mode_or_first(s):
        s = pd.Series(s)
        m = s.mode()
        return m.iloc[0] if not m.empty else s.iloc[0]

    agg = (
        DF.sort_values(["team_id", "session"])
          .groupby(["team_id", "session"], as_index=False)
          .agg(
              performance=("performance", "mean"),
              leader_type=("leader_type", _mode_or_first),
              layout=("layout", _mode_or_first),
              suggestions=("suggestions", _mode_or_first),
              name=("name", lambda s: s.iloc[0] if "name" in DF.columns else np.nan),
          )
    )

    global_mean = agg["performance"].mean()

    results = []
    for tid, group in agg.groupby("team_id", as_index=False):
        group = group.sort_values("session", ascending=True).reset_index(drop=True)

        # Basic first/last stats
        first = group.iloc[0]
        last  = group.iloc[-1]

        mean_perf = group["performance"].mean()
        std_perf  = group["performance"].std(ddof=0)  # population std
        net_diff  = last["performance"] - first["performance"]

        # Plateau / drop detection
        peak_idx = group["performance"].idxmax()
        peak_val = group.loc[peak_idx, "performance"]
        drop_from_peak = peak_val - last["performance"] if peak_idx != group.index[-1] else 0.0

        # Learning slope (session index → performance)
        X = np.arange(len(group)).reshape(-1, 1)
        y = group["performance"].values
        slope = LinearRegression().fit(X, y).coef_[0] if len(group) > 1 else 0.0

        # --- NEW: human-led (2nd − 1st) and AI-led (2nd − 1st) deltas
        def first_two_delta(g, leader_value):
            gg = g[g["leader_type"] == leader_value].sort_values("session")
            if len(gg) >= 2:
                s1, s2 = gg.iloc[0], gg.iloc[1]
                return {
                    "first_session": s1["session"],
                    "second_session": s2["session"],
                    "first_perf": float(s1["performance"]),
                    "second_perf": float(s2["performance"]),
                    "delta_2minus1": float(s2["performance"] - s1["performance"]),
                }
            else:
                return {
                    "first_session": np.nan,
                    "second_session": np.nan,
                    "first_perf": np.nan,
                    "second_perf": np.nan,
                    "delta_2minus1": np.nan,
                }

        human_pair = first_two_delta(group, human_label)
        ai_pair    = first_two_delta(group, ai_label)

        results.append({
            "team_id": tid,
            "first_session": first["session"],
            "last_session":  last["session"],
            "first_performance": float(first["performance"]),
            "last_performance":  float(last["performance"]),
            "net_diff": float(net_diff),
            "mean_performance": float(mean_perf),
            "std_performance": float(std_perf) if pd.notna(std_perf) else np.nan,
            "relative_to_global": float(mean_perf - global_mean),

            "peak_session": int(group.loc[peak_idx, "session"]),
            "peak_performance": float(peak_val),
            "drop_from_peak": float(drop_from_peak),
            "learning_slope": float(slope),

            "first_leader_type": first["leader_type"],
            "last_leader_type":  last["leader_type"],
            "first_layout": first["layout"],
            "last_layout":  last["layout"],
            "first_suggestions": first["suggestions"],
            "last_suggestions":  last["suggestions"],

            # --- NEW: Human-led pair
            "human_first_session": human_pair["first_session"],
            "human_second_session": human_pair["second_session"],
            "human_first_perf": human_pair["first_perf"],
            "human_second_perf": human_pair["second_perf"],
            "human_delta_2minus1": human_pair["delta_2minus1"],

            # --- NEW: AI-led pair
            "ai_first_session": ai_pair["first_session"],
            "ai_second_session": ai_pair["second_session"],
            "ai_first_perf": ai_pair["first_perf"],
            "ai_second_perf": ai_pair["second_perf"],
            "ai_delta_2minus1": ai_pair["delta_2minus1"],
        })

    df_out = (
        pd.DataFrame(results)
          .sort_values("net_diff", ascending=False)
          .reset_index(drop=True)
    )

    # Headline print
    print("\n=== Team performance profile (sorted by net_diff) ===")
    print(df_out.head(k).to_string(index=False))

    # NEW: concise summary of the two deltas
    cols = [
        "team_id",
        "human_first_session","human_second_session","human_first_perf","human_second_perf","human_delta_2minus1",
        "ai_first_session","ai_second_session","ai_first_perf","ai_second_perf","ai_delta_2minus1",
    ]
    print("\n--- Δ (2nd − 1st) within leader type ---")
    print(df_out[cols].head(k).to_string(index=False))

    return df_out

def first_diff_table(
    DF: pd.DataFrame,
    team_col: str = "team_id",
    session_col: str = "session",
    perf_col: str = "performance",
    complexity_col: str = "difficulty",
    leader_col: str  = "leader_type",
    layout_col: str  = "layout",
    tol: float = 1e-6,
    fit_model: bool = True,
):
    """
    Build a session-to-session transitions table within each team showing:
      - ΔPerformance and ΔComplexity with ↑/↓ arrows
      - Optional leader/layout change flags
      - If fit_model=True, fit Δperf ~ Δcomplexity + leader/layout changes
        and add:
          * pred_task = beta_complexity * Δcomplexity
          * resid_after_complexity = Δperf - pred_task
          * (optional) pred_leader = beta_leader * leader_change
          * pred_total = model prediction
          * resid_total = Δperf - pred_total
    Expects one row per (team, session). If multiple rows exist, aggregates by mean.
    """

    # --- 1) Collapse to one row per team × session (robust to duplicates)
    agg_dict = {
        perf_col: (perf_col, "mean"),
        complexity_col: (complexity_col, "mean"),
    }
    if leader_col and leader_col in DF.columns:
        agg_dict[leader_col] = (leader_col, "first")
    if layout_col and layout_col in DF.columns:
        agg_dict[layout_col] = (layout_col, "first")

    agg = (
        DF.groupby([team_col, session_col], as_index=False)
          .agg(**agg_dict)
          .sort_values([team_col, session_col])
          .reset_index(drop=True)
    )

    # --- 2) Within-team first differences
    agg["d_perf"] = agg.groupby(team_col)[perf_col].diff()
    agg["d_complexity"] = agg.groupby(team_col)[complexity_col].diff()

    # Optional change flags
    if leader_col and leader_col in agg.columns:
        agg["leader_change"] = (agg.groupby(team_col)[leader_col].diff().fillna(0) != 0).astype(int)
    else:
        agg["leader_change"] = 0

    if layout_col and layout_col in agg.columns:
        agg["layout_change"] = (agg.groupby(team_col)[layout_col].diff().fillna(0) != 0).astype(int)
    else:
        agg["layout_change"] = 0

    # Keep only rows that actually represent transitions
    T = agg.dropna(subset=["d_perf", "d_complexity"]).copy()

    # --- 3) Friendly arrows + quick interpretation
    T["perf_change"] = np.where(T["d_perf"] > 0, "↑", np.where(T["d_perf"] < 0, "↓", "—"))
    T["complexity_change"] = np.where(T["d_complexity"] > 0, "↑", np.where(T["d_complexity"] < 0, "↓", "—"))

    def _interpret(dp, dc):
        if pd.isna(dp) or pd.isna(dc):
            return ""
        if abs(dc) <= tol:
            return "Δperf with ~no Δcomplexity → interaction-driven"
        same = (dp > 0 and dc > 0) or (dp < 0 and dc < 0)
        return "Aligned with complexity" if same else "Counter to complexity → interaction-driven"

    T["interpretation"] = [_interpret(dp, dc) for dp, dc in zip(T["d_perf"], T["d_complexity"])]

    # --- 4) First-differences regression (optional)
    if fit_model:
        # Build formula with whatever covariates we have
        rhs = ["d_complexity"]
        if "leader_change" in T.columns:
            rhs.append("leader_change")
        if "layout_change" in T.columns:
            rhs.append("C(layout_change)")  # treat as categorical (0/1)

        formula = "d_perf ~ " + " + ".join(rhs)
        model = smf.ols(formula, data=T).fit(
            cov_type="cluster", cov_kwds={"groups": T[team_col]}
        )

        # Extract key coefficients safely (default 0 if absent)
        beta_complex = float(model.params.get("d_complexity", 0.0))
        beta_leader  = float(model.params.get("leader_change", 0.0))

        # Task-only predicted change
        T["pred_task"] = beta_complex * T["d_complexity"]
        T["resid_after_complexity"] = T["d_perf"] - T["pred_task"]

        # Optional component from leader change (useful to report separately)
        if "leader_change" in rhs:
            T["pred_leader"] = beta_leader * T["leader_change"]
        else:
            T["pred_leader"] = 0.0

        # Full model prediction + residual
        T["pred_total"] = model.predict(T)
        T["resid_total"] = T["d_perf"] - T["pred_total"]

        return T[[team_col, session_col, perf_col, complexity_col,
                  "d_perf", "d_complexity", "perf_change", "complexity_change",
                  "leader_change", "layout_change", "pred_task", "pred_leader",
                  "pred_total", "resid_after_complexity", "resid_total", "interpretation"]], model

    # If not fitting, just return the transitions table
    return T[[team_col, session_col, perf_col, complexity_col,
              "d_perf", "d_complexity", "perf_change", "complexity_change",
              "leader_change", "layout_change", "interpretation"]], None

def format_messages_llm(scenario_options):

    global session_log,json_session,different_agents,conversations,messages_present,distances,conversation_groups,last_time,total_sessions,good_sessions,good_names,all_messages,history_conversation_groups

    if session_log:
        json_session["content"][session_num] = session_log.copy()
        #tutorial = not all(so[1] == "human" for so in scenario_options["robots_type"])
        
        #json_session["tutorial"][session_num] = tutorial
        
        print(session_num)
        
        '''
        if tutorial:
            print("tutorial")
        else:
            print("not tutorial")
        '''
        
        robots_str = ""
        for r_idx,r in enumerate(scenario_options["robots_type"]):
            if r[1] == "ai":
                robot_type = "AI"
            elif r[1] == "human":
                robot_type = "Human"
            
            if r_idx:
                robots_str += ", "
                
            robots_str += "Agent " + r[0] + " is " + robot_type
            
            
                
        
        print(robots_str)
        for m_idx,m in enumerate(json_session["content"][session_num]):
            print(m_idx+1, m)
            
        #print(json_session["content"][session_num])
        
        if len(session_log) > 10 and len(different_agents) <= 4:
            total_sessions += 1
            if len(different_agents) not in good_sessions.keys():
                good_sessions[len(different_agents)] = 0
                good_names[len(different_agents)] = []
            good_sessions[len(different_agents)] += 1
            good_names[len(different_agents)].append(json_session["file"])
        

    #if messages_present:
    #    print(session_log)

    session_log = []
    
    distances = {}
    
    #if messages_present:
    #    pdb.set_trace()
    if conversations[0] and conversations[0][0]:
        message_log = {"messages":[preamble], "preambles":[0], "conversation_groups":[[]]}
        message_count = 1
        for c_key in conversations.keys():
            
            
            for d_idx,d in enumerate(conversations[c_key]):
                history_messages = []
                for t_idx,t in enumerate(d):
                    #print(t)
                    
                     if t_idx:
                        messg = "" 
                        
                        messg += "History of messages: " + str(history_messages) + "\n"
                        messg += "New message: " + str(t) + "\n"
                        messg += "Who is " + t["Sender"] + " replying to? Output a JSON.\n"
                        message_log["messages"].append(messg)
                        message_log["conversation_groups"].append(history_conversation_groups[c_key][d_idx])
                        
                        message_count += 1
                        
                     history_messages.append(t)
                     
                     
                     if message_count == 20:
                        message_log["preambles"].append(len(message_log["messages"]))
                        message_log["conversation_groups"].append([])
                        message_count = 0
                        message_log["messages"].append(preamble)
                    
            #print(json.dumps(message_log))
                
            #print("")
        
        if len(message_log["messages"]) > 1:
            all_messages["messages"].append(message_log["messages"])
            all_messages["preambles"].append(message_log["preambles"])
            all_messages["conversation_groups"].append(message_log["conversation_groups"])
            all_messages["agents"].append(different_agents)
            
    conversations = {0: [[]]} #{}
    history_conversation_groups = {0: [[]]}
    conversation_groups = {}
    last_time = {0: [[]]} #{}
    messages_present = False
    different_agents = []
                    
def get_settings(log_state_file):

    
    settings = log_state_file.readline()
    
    if not settings:
        return None,None
        
    settings = eval(settings.strip())
    arguments = log_state_file.readline()
    if not arguments:
        return None,None
    arguments = arguments.strip()
    scenario_options_line = log_state_file.readline()
    if scenario_options_line:
        scenario_options = eval(scenario_options_line.strip())
    else:
        scenario_options = None

    return scenario_options,settings



def match_survey(json_session, survey_results):

    for us in json_session["users"].keys():
        for sr in survey_results:
            try:
                if json_session["users"][us]["token"] == sr["token"]:
                    json_session["users"][us]["survey"] = {"questions": sr["questions"], "answers": sr["answers"]}
            except:
                pdb.set_trace()
                
                
    return json_session
                
                
def reset_json_session(file_d):
    json_session = {}
    
    json_session["file"] = file_d
    json_session["content"] = {}
    json_session["users"] = {}
    
    return json_session
    

def identify_who_brought_object(metadata, heavy_objects_in_goal, goal_radius):
    
    carried_by = {ob:"" for ob in heavy_objects_in_goal}
    
    try:
        for m in metadata:
            to_delete = []
            for ob in heavy_objects_in_goal:
                if np.linalg.norm(np.array(m["metadata"][int(ob)][2:4])) < goal_radius:
                    for ag in m["metadata"][20:]:
                        if ob in ag[5:7]:
                            carried_by[ob] = ag[1]
                            to_delete.append(ob)
                            break
                            
            for td in to_delete:
                heavy_objects_in_goal.remove(td)
            
    except:
        pdb.set_trace()       
    return carried_by 
 
def corr_one_feature(df, feature):
    rows = []
    for layout, g in df.dropna(subset=['performance', feature]).groupby('layout'):
        if len(g) < 5:
            rows.append((layout, float('nan'), float('nan'), len(g)))
            continue
        r, p = spearmanr(g['performance'], g[feature])
        rows.append((layout, r, p, len(g)))
    return pd.DataFrame(rows, columns=['layout','spearman_r','p_value','n']) \
             .sort_values('spearman_r', key=lambda s: s.abs(), ascending=False)   

def pandas_process(filtering_column, condition_columns, metric_column):

    performance = {}

    for a in df.groupby(filtering_column)[[*condition_columns, metric_column]]:
    
        condition = a[1].iloc[0][condition_columns].to_string(header=False, index=False).strip().replace('\n',' ')

        try:
            if condition not in performance:
                performance[condition] = []
        except:
            pdb.set_trace()
        performance[condition].append(a[1].iloc[0][metric_column])
        
        #print(a)
    
    for ps in performance.keys():
        print(ps, len(performance[ps]), np.nanmean(performance[ps]), np.nanstd(performance[ps]))
        
     
    return performance
    
    
def heuristic(a_row, a_col, b_row, b_col):
    # Manhattan distance
    return abs(a_row - b_row) + abs(a_col - b_col)

def reconstruct_path(parent, dest_row, dest_col):
    path = []
    r, c = dest_row, dest_col
    while True:
        path.append((r, c))
        pr, pc = parent[r][c]
        if pr == r and pc == c:
            break
        r, c = pr, pc
    path.reverse()
    return path

def findPath(start, goal, occ):
    """
    Pure-Python A* on a grid.
    
    start, goal: numpy arrays of shape (2,), e.g. np.array([r, c])
    occ: 2D numpy array of ints; passable when occ[r,c] in {0,3,-2}
    
    Returns list of (row, col) tuples from start (exclusive) to goal (inclusive),
    or [] if no path.
    """
    # unpack into Python ints
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])
    
    # trivial case
    if sr == gr and sc == gc:
        return []
    
    nrows, ncols = occ.shape
    INF = 10**9

    # preallocate grids
    g = [[INF]*ncols for _ in range(nrows)]
    f = [[INF]*ncols for _ in range(nrows)]
    parent = [[None]*ncols for _ in range(nrows)]
    closed = [[False]*ncols for _ in range(nrows)]

    # initialize start node
    g[sr][sc] = 0
    f[sr][sc] = heuristic(sr, sc, gr, gc)
    parent[sr][sc] = (sr, sc)

    # heap elements: (f_score, g_score, row, col)
    open_heap = [(f[sr][sc], 0, sr, sc)]

    # 4-connectivity
    neighbors = [(1,0),(-1,0),(0,1),(0,-1)]

    while open_heap:
        _, curr_g, r, c = heapq.heappop(open_heap)
        if closed[r][c]:
            continue
        if (r, c) == (gr, gc):
            return reconstruct_path(parent, gr, gc)

        closed[r][c] = True

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            # bounds check
            if not (0 <= nr < nrows and 0 <= nc < ncols):
                continue
            # closed check
            if closed[nr][nc]:
                continue
            # occupancy check
            if (nr, nc) != (gr, gc) and occ[nr, nc] not in (0, 3, -2):
                continue

            tentative_g = curr_g + 1
            if tentative_g < g[nr][nc]:
                g[nr][nc] = tentative_g
                h = heuristic(nr, nc, gr, gc)
                f[nr][nc] = tentative_g + h
                parent[nr][nc] = (r, c)
                heapq.heappush(open_heap, (f[nr][nc], tentative_g, nr, nc))

    # no path found
    return []

def manual_occupancy_map(scenario_options):
    
    walls = scenario_options["walls"]
    map_size = scenario_options["scenario_size"]
    occupancy_map = np.zeros((map_size,map_size))
    occupancy_map[:,:] = 0
    occupancy_map[0,:] = 1
    occupancy_map[-1,:] = 1
    occupancy_map[:,0] = 1
    occupancy_map[:,-1] = 1
    
    min_pos = [-9.5, -9.5]
    multiple = 1
    
    for wall in walls:
        first_point = [wall[0][0], wall[0][1]]
        last_point = [wall[-1][0], wall[-1][1]]
        
        first_cell = [round((first_point[0] + abs(min_pos[0])) / multiple),
               round((first_point[1] + abs(min_pos[1])) / multiple)]
               
        last_cell = [round((last_point[0] + abs(min_pos[0])) / multiple),
               round((last_point[1] + abs(min_pos[1])) / multiple)]
               
               
        if first_cell[0] == last_cell[0]:
            if first_cell[1] > last_cell[1]:
                for loc in range(last_cell[1],first_cell[1]+1):
                    occupancy_map[first_cell[0],loc] = 1
            else:
                for loc in range(first_cell[1],last_cell[1]+1):
                    occupancy_map[first_cell[0],loc] = 1
                
        elif first_cell[1] == last_cell[1]:
            if first_cell[0] > last_cell[0]:
                for loc in range(last_cell[0],first_cell[0]+1):
                    occupancy_map[loc,first_cell[1]] = 1
            else:
                for loc in range(first_cell[0],last_cell[0]+1):
                    occupancy_map[loc,first_cell[1]] = 1    

    return occupancy_map   

def copy_files(src,processed_folder,subfolder,date,identifier):


    #subfolder = ["videos/", "initial_state/", "traces/"]
    new_folder = processed_folder + subfolder
    if not os.path.isdir(new_folder):
        os.mkdir(new_folder)

    #src = ["../../simulator/videos/","../../simulator/log/","./"]
    glob_files = glob.glob(src + date + "*")
    for f in glob_files:
        name = os.path.basename(f).replace(date,identifier)
        shutil.copy(f, new_folder + name)

def convert_to_grid_coordinates(location):

    if not location:
        pos_new = [-1,-1]
    else:
        min_pos = [-9.5, -9.5]
        multiple = 1
        pos_new = [round((location[0] + abs(min_pos[0])) / multiple),
                   round((location[1] + abs(min_pos[1])) / multiple)]

    return pos_new
    

def H2(q):
    return -q*math.log2(q)-(1-q)*math.log2(1-q) if 0<q<1 else 0

def expected_posterior_entropy(r, p1_list, p2_list, repeat):
    E = 0.0
    for obs in itertools.product([0,1], repeat=repeat):  # 1=D,0=B
        # likelihoods
        lD = math.prod(p1_list[i]    if obs[i] else (1-p1_list[i]) 
                       for i in range(repeat))
        lB = math.prod((1-p2_list[i]) if obs[i] else p2_list[i] 
                       for i in range(repeat))
        p_obs = r*lD + (1-r)*lB
        if p_obs == 0: 
            continue
        post = (r*lD)/(p_obs)
        E += p_obs * H2(post)
    return E
    
def follow_to_carry(follow_order, current_orders):

    agent_to_follow = follow_order.split("_")[1]
    if agent_to_follow in current_orders.keys() and "carry" in current_orders[agent_to_follow][0]:
        return current_orders[agent_to_follow][0]
        
    return ""

def parse_sensing_message(rematch):    
        
    object_id = rematch.group(1)
    
    item = {"id":object_id, "sender":[]}
    
    item["item_danger_level"] = []
    item["item_danger_confidence"] = []
    
    
    if rematch.group(5):
        danger_list = rematch.group(6).strip('][').split(',')
        prob_list = rematch.group(8).strip('][').split(',')
        from_list = rematch.group(10).strip('][').split(',')
        
        for lidx in range(len(danger_list)):
        
            if "benign" in danger_list[lidx]:
                danger_level = 1
                
            else:
                danger_level = 2
        
            item["item_danger_level"].append(danger_level)
            item["item_danger_confidence"].append(float(prob_list[lidx].strip('%'))/100)
            item["sender"].append(from_list[lidx])


    return item
    
def translate_item_message(message,robot_id):
    
    message_text = ""
    
    for rematch in re.finditer(item_regex_full_alt,message):

        if rematch.group(5):
            old_string = rematch.group()
            new_string = " Status: [" + rematch.group(6) + "], Prob. Correct: [" + rematch.group(7) + "%], From: [" + robot_id + "]"
            message_text += old_string.replace(rematch.group(5), new_string)
            
    
    return message_text
    



rooms = {}
rooms_limits = {}
rooms[0] = [[x,y] for x in range(15,19) for y in range(11,19)]
rooms[1] = [[x,y] for x in range(1,6) for y in range(15,19)]
rooms[2] = [[x,y] for x in range(1,5) for y in range(7,14)]
rooms[3] = [[x,y] for x in range(1,6) for y in range(1,6)]
rooms[4] = [[x,y] for x in range(7,14) for y in range(1,6)]
rooms[5] = [[x,y] for x in range(15,19) for y in range(1,6)]
rooms[6] = [[x,y] for x in range(5,17) for y in range(7,10)]
rooms[7] = [[x,y] for x in range(17,19) for y in range(7,10)]
scenario_size = 20
cell_size = 1
for r_key in rooms.keys():
    if r_key == 6:
        multipliers = [0.5,0.5,1.2,1.5]
    else:
        multipliers = [0.5,0.5,1.5,1.5]
        
    rooms_limits[r_key] = [[loc[0]-scenario_size/2-cell_size*multipliers[0],loc[1]-scenario_size/2-cell_size*multipliers[1]] if loc_idx < len(rooms[r_key])-1 else [loc[0]-scenario_size/2+cell_size*multipliers[2],loc[1]-scenario_size/2+cell_size*multipliers[3]] for loc_idx,loc in enumerate(rooms[r_key])]
    

def get_room(position):
    ego_room = -1
    for r in rooms_limits.keys():
        if position[0] >= rooms_limits[r][0][0] and position[1] >= rooms_limits[r][0][1] and position[0] <= rooms_limits[r][-1][0] and position[1] <= rooms_limits[r][-1][1]:
            ego_room = r
            break
    return ego_room


agent_info = []
agent_info_columns = []
message_logs = {}
human_names = {}
message_to_df = [["Date", "Time", "AI Agents", "Sender", "Received by", "Message", "Sent by AI?", "Received by AI?"]]

extra_stats_questions = ["team_quality_no_dropped", "productivity", "quality_choosing", "quality_communication", "quality_carrying", "quality_moving", "team_quality_choosing", "team_quality_moving", "team_quality_communication", "team_quality_carrying", "total_distance_traveled", "team_objects_in_goal", "percentage_carried_heavy_objects", "percentage_carried_heavy_objects_mixed", "number_carried_heavy_objects", "total_messages_exchanged", "mixed_message_exchanges", "percentage_mixed_messages", "percentage_mixed_messages_AI_sender", "mixed_carrying_leader", "carrying_leader", "complete_coverage", "overlapping_coverage_mixed", "percentage_carried_objects_dangerous_ai", "percentage_carried_objects_dangerous_human", "times_played_simulated_environment", "informational_exchanges_mixed", "average_grabbed", "average_sensed", "percentage_grabbed_ai", "percentage_sensed_ai", "avg_num_messages_human", "avg_message_len_human", "avg_num_messages_ai", "avg_message_len_ai", "avg_messages_sent", "avg_messages_directed_to_ai", "avg_messages_directed_to_human", "avg_human_message_len_compared", "percentage_collective_sensed","percentage_ai_human_sensed","percentage_all_sensed", "percentage_sensed_both_types", "freq_sense"]

stats_questions = ['distance_traveled', 'grabbed_objects', 'grab_attempts', 'dropped_outside_goal', 'objects_sensed', 'sensor_activation', 'objects_in_goal', 'dangerous_objects_in_goal', 'num_messages_sent', 'average_message_length', 'time_with_teammates_A', 'time_with_teammates_B','time_with_teammates_C','time_with_teammates_D', 'end_time', 'team_dangerous_objects_in_goal', 'total_dangerous_objects', 'quality_work', 'effort', 'human_team_effort', 'team_end_time', 'team_quality_work', 'team_speed_work', 'team_achievement', "task_completion", "team_completion", *extra_stats_questions]

final_questions = ["Dependable","Reliable","Unresponsive","Predictable","Act consistently","Function successfully","Malfunction","Have errors","Provide feedback","Meet the needs of the mission/task","Provide appropriate information","Communicate with people","Perform exactly as instructed", "Follow directions"]


item_regex_full = "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status: (\[\w+(,\w+)*\]), Prob. Correct: (\[\d+\.\d+%(,\d+\.\d+%)*\]), From: (\[\w+(,\w+)*\]))?"
item_regex_full_alt = "Object (\d+) \(weight: (\d+)\) Last seen in (\(-?\d+\.\d+,-?\d+\.\d+\)) at (\d+:\d+).( Status: (\w+), Prob. Correct: (\d+\.\d+)%)?"

legacy = True

if not legacy:
    log_dir = "log"
else:
    log_dir = "log_HAexperiments" #"log_january2025" #"log_HAexperiments"

ai_only = False

ai_end = True

if ai_only:
    log_dir = "log_AIOnly"


files = glob.glob(log_dir + "/20*.txt")
files.sort(key=lambda x: datetime.strptime(os.path.basename(x)[:19], "%Y_%m_%d_%H_%M_%S"))

print(files)

processed_folder = "processed_traces/"

if os.path.isdir(processed_folder):
    shutil.rmtree(processed_folder)

os.mkdir(processed_folder)

#files = ["2024_09_24_11_40_52.txt"]

#df_message = pd.read_csv('message_check_processed.csv', header=None, encoding='latin-1')

obfuscated_date = 0

track_experience = {}

together_files = [["2025_07_10_17_12_54", "2025_07_10_17_28_39"], ["2025_07_15_14_59_31", "2025_07_15_15_13_10"], ['2025_08_11_20_02_46','2025_08_11_20_26_58'],["2025_08_18_20_43_08", "2025_08_18_21_00_51"], ["2025_08_19_07_15_44", "2025_08_19_07_45_09"], ["2025_08_23_15_29_41", "2025_08_23_15_44_24"], ["2025_08_25_19_53_56","2025_08_25_20_09_45"], ["2025_09_03_15_15_13", "2025_09_03_15_22_30"], ["2025_09_17_14_53_55","2025_09_17_15_04_01"]] #Watchout 2025_08_11_20_26_58

together_files_dict = {tf[1]:tf[0] for tf in together_files}

previous_file = ""
final_average_survey_results = {}



tutorial = ["2025_07_10_15_24_22","2025_07_15_13_56_19", "2025_07_20_12_53_53", "2025_08_01_13_52_35", "2025_08_05_06_55_45", "2025_08_07_18_52_22", '2025_08_11_18_52_39', "2025_08_18_18_50_56", "2025_08_19_05_52_17", "2025_08_20_14_53_11", "2025_08_21_09_42_16", "2025_08_23_14_52_08", "2025_08_25_18_52_00", "2025_09_03_13_21_49", "2025_09_10_09_54_41", "2025_09_11_05_51_32", "2025_09_17_13_50_57", "2025_12_29_08_54_42", "2026_01_19_12_12_17"]
skip = ["2025_07_20_13_05_58", "2025_07_20_13_26_40", "2025_08_05_08_45_52", "2025_08_05_08_59_53", "2025_08_05_07_19_39", "2025_08_07_19_23_14", '2025_08_11_19_26_23', '2025_08_11_20_23_20', '2025_08_11_20_24_29', '2025_08_11_20_33_22', "2025_08_18_20_53_20", "2025_08_19_07_37_53", "2025_08_19_07_40_49", "2025_08_20_15_24_56", "2025_08_20_15_28_45", "2025_08_23_17_17_47", "2025_08_25_14_53_35", "2025_08_25_14_54_47", "2025_08_25_19_24_32", "2025_08_25_20_02_51", "2025_08_25_20_08_10", "2025_09_03_15_21_25", "2025_09_10_11_08_22", "2025_09_10_11_12_52", "2025_09_10_12_05_13", "2025_09_10_12_33_36", "2025_09_11_06_34_32", "2025_09_11_06_35_22", "2025_09_11_07_26_23", "2025_09_17_15_01_53", "2026_01_19_14_17_27", "2025_12_28_17_30_10", "2025_12_28_20_58_09"] #2025_08_05_07_19_39 is also tutorial
skip_first = ["2025_07_10_15_24_22","2025_07_10_15_56_34","2025_07_10_16_21_36","2025_07_10_16_48_09","2025_07_10_17_12_54", "2025_07_10_17_28_39"]

skip_substitute = ["2025_09_17_13_50_57", "2025_09_17_14_27_25", "2025_09_17_14_53_55", "2025_09_17_15_01_53", "2025_09_17_15_04_01", "2025_09_17_15_21_50", "2025_09_17_15_40_51", "2025_07_20_12_53_53",  "2025_07_20_13_28_59","2025_07_20_13_53_26", "2025_07_20_14_19_52", "2025_07_20_14_46_00"]

skip.extend(skip_substitute)

skip.extend(skip_first)

layouts = [[1,2,3,4],[2,4,1,3],[4,3,2,1],[3,1,4,2]]

layout_ids = [1,2,2,2,1,1,1,3,3,3,3,4,4,4,4,2] #[1,2,2,2,2,1,1,1,3,3,3,3,4,4,4,4] #add for each new experiment

empty_rooms = [[3,2,6],[0,2,6],[7,2,6],[5,2,6]]

ai_layouts = [2,4,3,1,1,4,2,3]
counter_ai_only = 0

skip_sessions_ai_only= {
"2025_09_18_15_02_17": [0,7,8, 14, 15],
"2025_09_18_20_03_38": [0, 5, 10, 14],
"2025_09_19_09_50_44": [0, 6],
"2025_09_18_00_10_43": [0,2,3,6,7,8,9,10,11,12,13,14,15,16],
"2025_09_18_09_39_10": [0,2,7,9,10],
"2025_09_19_16_20_28": [0,5],
"2025_09_17_19_14_03": [0,10,12,13,14,15,16],
"2025_09_25_21_07_10": [0,2,3,5,7,11,13,15,16]
}
ai_layout_tracker = {1:16,2:16,3:16,4:16}

extra_resets = {"2025_07_15_14_30_47": 2, "2025_07_15_15_51_02":3, "2025_08_18_19_26_28": 2}

current_layout_id = 0
ai_layout_idx = 0

if legacy:
    skip = ["2024_09_03_19_55_37", "2024_09_04_20_31_42"]

if ai_only:
    skip = ["2025_09_17_19_11_50", "2025_09_17_19_13_06", "2025_09_19_16_18_31"]

debug_var = False

BOLD = "\033[32m"
RESET = "\033[0m"

all_final_metrics = []
df_sessions = []
payment = {"A":[],"B":[]}
occ_map = np.array([])
final_metrics = {"total_distance_traveled":[], "distribution_sensed":[], "objects_carried":[], "dangerous_objects":[], "weight_distribution_carried":[], "weight_sum":[], "complexity": [], "complexity_reduction":[], "name":[], "strategy":[], "suggestions_taken":[], "suggestions_given":[], "sensing_complexity":[], 'experience_videogame':[], 'experience_teamcollab':[], 'experience_llm':[], "accidental_drop":[], "idle_time":[], "object_carry_speed": [], "order_completed": [], "object_dangerous_time": [], "object_benign_time":[], "help_carry_dialogue":[], "objects_carried_distribution":[], "objects_carried_distribution_danger":[], "objects_carried_distribution_weight":[], "idle_time_distribution": [], "travel_distance_distribution":[], 'orders_parsed':[], "carry_order_to_first_carry":[], "num_times_sensed_before_carry":[], "frac_dangerous_ambiguous": [], "frac_benign_as_dangerous": [], "lost_track":[], "NASA":[], "PTT":[]}

tutorial_results = {"questions":[], "answers": []}
demographic_qs = ['Name', 'Age', 'How often do you play first-person video games on average?', 'How many times have you interacted with the TeamCollab simulation environment in the past?', 'How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?', 'Generally, I trust automated agents', 'Automated agents help me solve many problems', 'I think it’s a good idea to rely on automated agents for help', 'I don’t trust the information I get from automated agents', 'Automated agents are reliable', 'I rely on automated agents']
average_demographic_qs = {}
participants = {
 '1': 'A',  '2': 'B',  '3': 'A',  '4': 'B',
 '5': 'A',  '6': 'B',  '7': 'A',  '8': 'B',
 '9': 'A',  '10': 'B', '11': 'A', '12': 'B',
 '13': 'A', '14': 'B', '15': 'A', '16': 'B',
 '17': 'A', '18': 'B', '19': 'A', '20': 'B',
 '21': 'A', '22': 'B', '23': 'A', '24': 'B',
 '25': 'A', '26': 'B', '27': 'A', '28': 'B',
 '29': 'A', '30': 'B', '31': 'A', '32': 'B'
} #TODO DELETE THIS -----------------------------

participants_metadata = []

BASE = Path("chat_logs")
carrying_df = pd.read_csv(BASE / "orders_grab_drop_summary.csv")

for col in ["first_order_time","first_grab_time","first_drop_in_goal_time","last_drop_time"]:
    carrying_df[col] = pd.to_numeric(carrying_df[col], errors="coerce")
    
carrying_df["carrying_time"] = 0
for index, row in carrying_df.iterrows():

    if not pd.isna(row["first_grab_time"]):
        if not pd.isna(row["first_order_time"]) and row["first_order_time"] < row["first_grab_time"]:
            first_time = row["first_order_time"]
        else:
            first_time = row["first_grab_time"]
            
        if not pd.isna(row["first_drop_in_goal_time"]):
            second_time = row["first_drop_in_goal_time"]
        else:
            if pd.isna(row["last_drop_time"]):
                second_time = 1200 #If drop not registered, put the maximum time
            else:
                second_time = row["last_drop_time"]
        

        carrying_df.loc[index,"carrying_time"] = second_time - first_time

dfile_idx = 0
dfile_idx_prev = 0
while dfile_idx < len(files):

    
    #if dfile_idx > dfile_idx_prev:
    #    pdb.set_trace()
    d = files[dfile_idx]

    dfile_idx += 1

    
    if debug_var:
        #print("\n".join(message_history))
        quit()

    #d = "log/2025_07_10_15_56_34.txt"
    
    #d = "log/2025_07_10_16_21_36.txt"
    #debug_var = True
    
    dtime = os.path.basename(d)[:-4]
    
    #d = "log_test/2025_07_23_10_39_07.txt"
    #dtime = "2025_07_23_10_39_07"
    #log_dir = "log_test"
    
    if dtime in skip:
        continue
    
    if "events" not in d:
    
        if not ai_only or (ai_only and ai_end):

            if any(t in d for t in tutorial):
                number_surveys = 0
                log_file = open(d)
                new_line = log_file.readline()
                while new_line:
                    split_line = new_line.strip().split(',')
                    if int(split_line[1]) == 7:
                        
                        first_bracket = new_line.find('[')
                        second_bracket = new_line.find(']')
                        third_bracket = new_line.rfind('[')
                        fourth_bracket = new_line.rfind(']')
                        
                        last_comma = new_line.rfind(',')
                        
                        questions = eval(new_line[first_bracket:second_bracket+1])
                        answers = eval(new_line[third_bracket:fourth_bracket+1])
                        
                        if answers[0]:
                            number_surveys += 1
                            
                            if not tutorial_results["questions"]:
                                tutorial_results["questions"] = questions.copy()
                            tutorial_results["answers"].append(answers)
                            #print(questions, answers)
                
                    if number_surveys == 2:
                        break
                        
                    new_line = log_file.readline()
                    
                            
                PTT = [0,0]
                tres = tutorial_results["answers"][-2:]
                for q_idx in range(len(demographic_qs)):
                    if not q_idx: #Question is Name
                        continue

                    if demographic_qs[q_idx] not in average_demographic_qs.keys():
                        average_demographic_qs[demographic_qs[q_idx]] = []

                    var = 0
                    for ag in range(len(tres)):
                        if q_idx > 4: #This is where PTT starts
                            if q_idx == 8: #Reverse
                                PTT[ag] += 6-int(tres[ag][q_idx])
                            else:
                                PTT[ag] += int(tres[ag][q_idx])
                        var += int(tres[ag][q_idx])
                        
                    average_demographic_qs[demographic_qs[q_idx]].append(var/len(tres))
                
                for ag in range(len(tres)):
                    PTT[ag] /= 6
                    
                continue
        
            log_file = open(d)
            
            try:
                log_state_file = open("../simulator/" + log_dir + "/" + dtime + "_state.txt")
            except:
                print("Could not open file", d)
                continue
        
        new_line = log_file.readline()
        just_reset = []
        
        num_humans = 0
        robots_ais = 0
        
        
        
        

        continue_file = False
        if any(dtime in t for t in together_files) and not previous_file:
            previous_file = dtime
        elif any(dtime in t for t in together_files) and previous_file:
            continue_file = True
            previous_file = ""
            print("Continue", dtime)
        else:
            previous_file = ""
            
        if not continue_file:
        
            if not ai_only or (ai_only and ai_end):
                scenario_options,settings = get_settings(log_state_file)
            
            sensor_qualities = {}
            
            robots_type = {}
            for agent in scenario_options["robots_type"]:
                
                if agent[1] == "human":
                    num_humans += 1
                    robots_type[agent[0]] = 'human'
                else:
                    robots_ais += 1
                    robots_type[agent[0]] = 'ai'
                    
                
                sensor_qualities[agent[0]] = {"p11": agent[2],"p22": agent[3]}
        
            strategy = ''
            team_composition = ""
            
            if not (ai_only and not ai_end):
                leader = ''
            agent_model = {}
            agent_reliability = {}
            metadata = ""
            last_time = 0
            previous_last_time = 0
            latest_item_info = {}
            objects_grabbed = {}
            objects_grabbed_total = []
            objects_fully_sensed = {}
            objects_partially_sensed = {}
            
            metrics = {"goal_objects":[]}
            object_agent_locations = []
            agents = {}
            objects_metadata = {}
            survey_results = []
            average_survey_results = {}
            time_spent_carrying = {}
            message_history = []
            suggestions_taken = 0
            suggestions_given = 0
            accidentally_dropped_objects = []
            complexity = 0
            idle_time = 0
            order_completed = 0
            grabbed_objects_tracker = {}
            chat_log = []
            help_carry_dialogue = 0
            idle_time_distribution = {'A':0, 'B':0, 'C':0, 'D':0}
            travel_distance_distribution = {'A':0, 'B':0, 'C':0, 'D':0}
            carry_order_to_first_carry = {}
            num_times_sensed_before_carry = {}
            lost_track = 0
            num_resets = 0
            NASA = []
            where_is_agent = {"C":False,"D":False}
            extra_features = {"changes_in_carrying_teams":[], "carried_by_AI":[], "teammates_dispersal_carrying_order":[], "drop_per_distance":{}, "reliability_score":0, "communication_range_error":0, "times_reprimanded":0, "result_before_carrying":[], "team_human_ai_composition":[], "orders_AI_human_fraction":[], "completed_AI_human_fraction":[], "team_dispersion":[], "dispersion_to_leader":[], "repeated_order":[], "probability_at_grabbing":[], "uncertainty_at_grabbing":[]}
            sense_track = {}
 
            
 
            tokens = {}
            known_objects_to_leader = {}
            
            current_orders = {}
            orders_parsed2 = {}
            team_tracker = TeamTracker()
            frequency_sample = {"period": 1, "current_period": 0}
            for ag in ['A','B','C','D']:
                orders_parsed2[ag] = {'completed':[], 'incomplete':[], 'preempt': [], 'unique_carry':[], 'unique_sense':[], 'unspecified':[], 'completed_actions':[], "all":[], "orders_changed":[]}
            
            orders_parsed = {}
            for ag in ['C','D']:
                orders_parsed[ag] = {'previous_order':"", 'changes_of_order':0, 'sensing_orders':[], 'carrying_orders':[], 'follow_orders':[], 'unique_sensing_orders':[], 'unique_carrying_orders':[], 'order_completes':0, 'unspecified_team':0, 'incomplete_orders':0, "misunderstood":0}
            
            if not legacy:
                start = convert_to_grid_coordinates([0,5])
            else:
                start = convert_to_grid_coordinates([0,0])
            
        
        if occ_map.size == 0:
            occ_map = manual_occupancy_map(scenario_options)
        
        #print(d)
        
        while new_line:
            split_line = new_line.strip().split(',')
            
            #print(new_line, current_layout_id)
            
            if not legacy and int(split_line[1]) == 3 and not dtime in extra_resets.keys():
                
                agent_id = split_line[2]
                #print(new_line)
                
                first_bracket = new_line.find('{')
                second_bracket = new_line.rfind('}')
                third_bracket = new_line[second_bracket:].find('[') + second_bracket
                fourth_bracket = new_line[second_bracket:].find(']') + second_bracket
                fifth_bracket = new_line[fourth_bracket:].find('[') + fourth_bracket
                sixth_bracket = new_line.rfind(']')
                
                item_info = eval(new_line[first_bracket:second_bracket+1])
                
                if item_info:
                    if agent_id not in latest_item_info.keys():
                        latest_item_info[agent_id] = {}
                    latest_item_info[agent_id].update(item_info)
                
                                
                object_grabbed = eval(new_line[third_bracket:fourth_bracket+1])
                try:
                    objects_dropped = eval(new_line[fifth_bracket:sixth_bracket+1])
                except:
                    pdb.set_trace()
                
                accidentally_dropped = False
                alt_object_id = ""
                for ob in objects_dropped:
                    if ob[0] == agent_id:
                        accidentally_dropped = True
                        alt_object_id = ob[1]
                        break
                
                #if objects_dropped:
                #    pdb.set_trace()
                
                if any(object_grabbed) and not (agent_id in objects_grabbed and objects_grabbed[agent_id]):
                
                    object_id = ""
                    for ob in object_grabbed:
                        if ob:
                            object_id = ob
                            break
                
                    objects_grabbed[agent_id] = object_id
                    
                    if object_id not in objects_grabbed_total:
                        objects_grabbed_total.append(object_id)
                        
                    #print("Object " + object_id + " grabbed at " + split_line[0] + " by " + agent_id, object_grabbed)
                    
                    chat_log.append({"type":"action", "time":split_line[0], "description": "Object " + object_id + " grabbed by "  + agent_id})
                    
                    c_order = 'carry_' + object_id
                    if c_order in carry_order_to_first_carry.keys() and not carry_order_to_first_carry[c_order]["end"]:
                        carry_order_to_first_carry[c_order]["end"] = float(split_line[0])
                    
                    if object_id not in grabbed_objects_tracker.keys():
                        grabbed_objects_tracker[object_id] = {"start_time":float(split_line[0]), "end_time":0, "goal":False, "agent_id": agent_id}
                        groundtruth = {}              
                        for m in metadata["metadata"]:
                            if not m[0]:
                                groundtruth[m[1]] = m[5]
                        try: 
                            if object_id in known_objects_to_leader.keys():
                                results = evaluate_objects(
                                    {object_id:known_objects_to_leader[object_id]},
                                    groundtruth,
                                    max_readings=4,
                                    entropy_epistemic_thresh_bits=0.9
                                )
                                extra_features["uncertainty_at_grabbing"].append(results["per_object"][object_id]["entropy_bits"])
                                extra_features["probability_at_grabbing"].append(results["per_object"][object_id]["q_danger"])
                    
                        except:
                            pdb.set_trace()
                        

                    if object_id not in num_times_sensed_before_carry.keys():
                        num_times_sensed_before_carry[object_id] = sum(object_id in objects_fully_sensed[ag] for ag in objects_fully_sensed.keys())                    
                    
                elif not any(object_grabbed) and agent_id in objects_grabbed and objects_grabbed[agent_id]:
                    
                    object_id = objects_grabbed[agent_id]
                    
                    objects_grabbed[agent_id] = ""
                    
                    grabbed_objects_tracker[object_id]["agent_id"] = agent_id
                    
                    if not accidentally_dropped:
                        ob_str = "Object " + object_id + " dropped at " + split_line[0] + " by " + agent_id
                        chat_log.append({"type":"action", "time":split_line[0], "description": "Object " + object_id + " dropped normally by "  + agent_id})
                        
                    else:
                        ob_str = "Object " + object_id + " accidentally dropped at " + split_line[0] + " by " + agent_id
                        accidentally_dropped_objects.append(object_id)
                        chat_log.append({"type":"action", "time":split_line[0], "description": "Object " + object_id + " accidentally dropped by "  + agent_id})
                    
                    '''
                    if dtime == "2025_08_25_20_28_32" and object_id == "13":
                        pdb.set_trace()
                    '''
                    if not grabbed_objects_tracker[object_id]["goal"]:
                        grabbed_objects_tracker[object_id]["end_time"] = float(split_line[0])
                    
                    if object_id not in metrics["goal_objects"]:
                        for m in metadata["metadata"]:
                            if not m[0] and m[1] == object_id and np.linalg.norm(np.array(m[2:4])-np.array([0,5])) < settings["goal_radius"]:
                                ob_str += ". Object in goal"
                                chat_log[-1]["description"] += " in the goal area"
                                metrics["goal_objects"].append(object_id)
                                grabbed_objects_tracker[object_id]["goal"] = True
                                
                                '''
                                if dtime == "2025_08_25_20_28_32":
                                    pdb.set_trace()
                                '''
                                c_order = 'carry_' + object_id
                                if c_order not in orders_parsed2[agent_id]['completed']:
                                    orders_parsed2[agent_id]['completed'].append(c_order)
                                    
                                    
                                break
                            
                    #print(ob_str, object_grabbed)
                #else:
                #    print("Accidentally droped object " + alt_object_id + " by team with agent " + agent_id)
                    
                    
                if agent_id not in objects_fully_sensed.keys():
                    objects_fully_sensed[agent_id] = []
                if agent_id not in objects_partially_sensed.keys():
                   objects_partially_sensed[agent_id] = []
                   
                for item_key in item_info.keys():
                    if "sensor" in item_info[item_key] and agent_id in item_info[item_key]["sensor"] and item_info[item_key]["sensor"][agent_id]:
                        if item_key not in objects_fully_sensed[agent_id]:
                            objects_fully_sensed[agent_id].append(item_key)
                            if agent_id == leader:
                                if item_key not in known_objects_to_leader.keys():
                                    known_objects_to_leader[item_key] = {"danger_level":[], "prob":[], "from":[]}
                                known_objects_to_leader[item_key]["from"].append(agent_id)
                                try:
                                    known_objects_to_leader[item_key]["danger_level"].append(item_info[item_key]["sensor"][leader]["value"])
                                except:
                                    pdb.set_trace()
                                known_objects_to_leader[item_key]["prob"].append(item_info[item_key]["sensor"][leader]["confidence"])
                        
                        
                        room = get_room(agents[agent_id]["last_position"])
                        
                        if room == -1: # and dtime == "2025_07_15_15_27_48":
                            #pdb.set_trace()
                            continue
                        c_order = 'sense_room' + str(room)
                        if c_order not in orders_parsed2[agent_id]['completed']:
                            orders_parsed2[agent_id]['completed'].append(c_order)
                        
                            #print("Fully sensed object " + item_key + " at time " + split_line[0] + " by " + agent_id, item_info[item_key])
                    elif "weight" in item_info[item_key] and item_info[item_key]["weight"]:
                        if item_key not in objects_partially_sensed[agent_id]:
                            objects_partially_sensed[agent_id].append(item_key)
                        
                            #print("Partially sensed object " + item_key + " at time " + split_line[0] + " by " + agent_id, item_info[item_key])
        
        
            elif int(split_line[1]) == 4:
                
                    
                log_line_agent = split_line[2]
        
                just_reset.append(log_line_agent)

                if all(robot[0] in just_reset for robot in scenario_options["robots_type"]): 
                    print("Resetting")
                    if dtime in extra_resets.keys():
                        extra_resets[dtime] -= 1
                        if not extra_resets[dtime]:
                            del extra_resets[dtime]
                    
                    #print("NUMBER OF OBJECTS IN GOAL", metrics["goal_objects"])
                    
                    
                    if not continue_file:
                        try:
                            scenario_options_tmp,settings_tmp = get_settings(log_state_file)
                        except:
                            pdb.set_trace()
                        
                        if scenario_options_tmp:
                            scenario_options = scenario_options_tmp
                            settings = settings_tmp

                        
                    just_reset = []
                    
                    first_bracket = new_line.find('{')
                    second_bracket = new_line.rfind('}')
                    
                    newer_line = new_line[first_bracket:second_bracket+1].replace("false", "False").replace("true","True")
                    
                    team_strategy = eval(newer_line)
                    
                    if legacy:
                    
                        type_subrole = [[0,0],[0,0]]
                        
                        team_strategy_agents = {}
                        
                        
                        if settings["num_humans"] > settings["num_ais"]:
                            team_composition = "Majority Human"
                        elif settings["num_ais"] > settings["num_humans"]:
                            team_composition = "Majority AI"
                        else:
                            team_composition = "Equal"
                        
                        for agent in scenario_options["robots_type"]:
                            if agent[1] == "human":
                                if team_strategy["role"][agent[0]] == "sensing" or team_strategy["hierarchy"][agent[0]] == "order" or team_strategy["interdependency"][log_line_agent] == "followed":
                                    type_subrole[0][0] += 1
                                else:
                                    type_subrole[0][1] += 1
                            else:
                                if team_strategy["role"][agent[0]] == "sensing" or team_strategy["hierarchy"][agent[0]] == "order" or team_strategy["interdependency"][log_line_agent] == "followed":
                                    type_subrole[1][0] += 1
                                else:
                                    type_subrole[1][1] += 1
                        
                        if team_strategy["role"][log_line_agent] != "equal":
                            type_strategy = "complementary"
                            
                            if type_subrole[0][0] == sum(type_subrole[0]) and type_subrole[1][1] == sum(type_subrole[1]):
                                substrategy = "Human-only sensing"
                            elif type_subrole[0][1] == sum(type_subrole[0]) and type_subrole[1][0] == sum(type_subrole[1]):
                                substrategy = "Human-only lifting"
                            else:
                                substrategy = "Mixed Roles"
                            
                            particular_role = team_strategy["role"][log_line_agent]
                        elif team_strategy["hierarchy"][log_line_agent] != "equal":
                            type_strategy = "hierarchy"    
                            
                            if type_subrole[0][0] == sum(type_subrole[0]) and type_subrole[1][1] == sum(type_subrole[1]):
                                substrategy = "Human-only leaders"
                            elif type_subrole[0][1] == sum(type_subrole[0]) and type_subrole[1][0] == sum(type_subrole[1]):
                                substrategy = "AI-only leaders"
                            else:
                                substrategy = "Mixed Leaders"
                            
                            particular_role = team_strategy["hierarchy"][log_line_agent]
                        elif team_strategy["interdependency"][log_line_agent] != "equal":
                            type_strategy = "independent"
                            
                            if not team_strategy["return"][log_line_agent]:
                                substrategy = "Single unit"
                            else:
                                if type_subrole[0][0] + type_subrole[1][0] == len(scenario_options["robots_type"]):
                                    substrategy = "Individuals"
                                else:
                                    substrategy = "Multiple units"
                            
                            particular_role = "independent"   
                        
                        
                        
                        for robot in scenario_options["robots_type"]: 
                        
                            if True: #robot[1] == "human":
                        
                                if team_strategy["role"][robot[0]] != "equal":
                                    particular_role = team_strategy["role"][robot[0]]
                                elif team_strategy["hierarchy"][robot[0]] != "equal":
                                    particular_role = team_strategy["hierarchy"][robot[0]]
                                elif team_strategy["interdependency"][robot[0]] != "equal":
                                    particular_role = "independent"
                                
                                team_strategy_agents[robot[0]] = [type_strategy, substrategy, particular_role]
                                strategy = type_strategy + " " +  substrategy
                    
                    else:
                        for agent in scenario_options["robots_type"]:
                            if team_strategy["hierarchy"][agent[0]] == "order":
                                if agent[1] == "human":
                                    strategy = "human leader"
                                else:
                                    strategy = "AI leader"
                                    
                                leader = agent[0]
                                    
                                break
                 
                        if team_strategy["suggestions"]:
                            strategy += " with suggestions"
                        else:
                            strategy += " without suggestions"
                     
                    if not complexity or "2025_07_15" in dtime:   
                        complexity = 0
                    
                        for ob in scenario_options["objects"]:
                            if ob[2] == 2:
                                goal = convert_to_grid_coordinates([ob[3]["x"],ob[3]["z"]])
                                path = findPath(start, goal, occ_map)
                                complexity += len(path)*ob[1] #len(path)*2*ob[1]
                        
                        #print("Complexity:",complexity)
                        
                    if "human" in strategy:
                        strat_str = "A is the leader"
                        if "with suggestions" in strategy:
                            strat_str += ", AI agents are going to be giving suggestions"
                    else:
                        strat_str = "C is the leader"
                        if "with suggestions" in strategy:
                            strat_str += ", human agents are going to be giving suggestions"
                        
                    chat_log.append({"type":"setup", "strategy":strat_str, "team": "A and B are human, C and D are AI"})
                    
                    if ai_only:
                        average_survey_results = {"None":None}
                        new_line = log_file.readline()
                        break

            elif not legacy and int(split_line[1]) == 2 and not dtime in extra_resets.keys():
                
                first_bracket = new_line.find('{')
                second_bracket = new_line.rfind('}')
                
                
                messages = eval(new_line[first_bracket:second_bracket+1])
                
                regex_str = r"My goal is (\w+) (\(-?\d+\.\d+,-?\d+\.\d+\)), I'm moving towards (\(-?\d+\.\d+,-?\d+\.\d+\)). My current location is (\(-?\d+\.\d+,-?\d+\.\d+\)).( Carrying object (\w+).)?( Helping (\w+).)?"
                    
                rematch = re.search(regex_str,messages["whole"])

                if rematch:
                    messages["whole"] = messages["whole"].replace(rematch.group(0), "").strip()
                
                if "AI" in strategy:
                    if "C" in messages.keys() and "Order completed" in messages["C"]:
                        order_completed += 1
                else:
                    try:
                        if "A" in messages.keys() and "Order completed" in messages["A"]:
                            order_completed += 1
                    except:
                        pdb.set_trace()
                        
                if messages["whole"]:
                    mess = "Message sent by " + split_line[2] + " at time " + split_line[0] + ": " + messages["whole"]
                    
                    chat_log.append({"type":"communication", "time":split_line[0], "sender": split_line[2], "receivers":[], "message": messages["whole"]})
                    csvwriter.writerow([dtime,split_line[0],split_line[2],messages["whole"],split_line[2] == leader])
                    #print(mess)
                    
                    if leader in messages.keys() and messages[leader]:
                        if re.search(item_regex_full,messages[leader]) or re.search(item_regex_full_alt,messages[leader]):
                            new_rm = messages[leader]
                            new_rm += translate_item_message(messages[leader],split_line[2])
                            for ridx,rematch in enumerate(re.finditer(item_regex_full,new_rm)):
                                item_dict = parse_sensing_message(rematch)
                                if item_dict["id"] not in known_objects_to_leader.keys():
                                    known_objects_to_leader[item_dict["id"]] = {"danger_level":[], "prob":[], "from":[]}
                                
                                for ag_idx in range(len(item_dict["sender"])):
                                    ag = item_dict["sender"][ag_idx]
                                    if ag != leader and ag not in known_objects_to_leader[item_dict["id"]]["from"]:
                                        known_objects_to_leader[item_dict["id"]]["from"].append(ag)
                                        known_objects_to_leader[item_dict["id"]]["danger_level"].append(item_dict["item_danger_level"][ag_idx])
                                        known_objects_to_leader[item_dict["id"]]["prob"].append(item_dict["item_danger_confidence"][ag_idx])
                    
                    
                    '''
                    if "steering" in messages["whole"]:
                        print("Message sent by " + split_line[2] + " at time " + split_line[0], messages["whole"])
                        #debug_var = True
                        mess = BOLD + mess + RESET
                    '''
                    
                    if split_line[2] == "C":
                        if "Your plan doesn't seem good:" in messages["whole"] or "Your plan seems good" in messages["whole"]: #or "We are already doing the plan you are proposing!" in messages['whole'] 
                            suggestions_taken += 1
                        
                        if "Your plan seems good" in messages["whole"]:
                            suggestions_given += 1
                        
                        if "You are carrying the wrong object" in messages["whole"] or "You should go towards" in messages["whole"]:
                            extra_features["times_reprimanded"] += 1
                    
                    if split_line[2] in ["C","D"]:
                        if not where_is_agent[split_line[2]] and "Where is agent" in messages["whole"]:
                            where_is_agent[split_line[2]] = True
                            lost_track += 1
                        elif not "I don't know where is agent" in messages["whole"]:
                            where_is_agent[split_line[2]] = False
                    
                    rematch = re.search(r"Room (\d+) is empty",messages["whole"])
                    if split_line[2] in current_orders.keys() and ("Room is empty" in messages["whole"] or rematch):

                        current_ord = current_orders[split_line[2]][0]
                        
                        if "Room is empty" in messages["whole"] and "sense_room" in current_ord:
                            room = current_ord.split("room")[1]
                        elif rematch:
                            room = rematch.group(1)    
                            
                        if ai_only:
                            layout_num = ai_layouts[ai_layout_idx]
                        else:
                            layout_num = layouts[layout_ids[current_layout_id // 4]-1][current_layout_id % 4]
                            
                        if int(room) in empty_rooms[layout_num-1]:
                            c_order = 'sense_room' + room 
                            if c_order not in orders_parsed2[split_line[2]]['completed']:
                                orders_parsed2[split_line[2]]['completed'].append(c_order)
                                

                    if "Order completed" in messages["whole"] and split_line[2] in current_orders.keys() and current_orders[split_line[2]]:
                        c_order = current_orders[split_line[2]][0]
                        
                        if "follow" in c_order:
                            c_order = follow_to_carry(c_order, current_orders)
                            
                            if c_order:
                                if not any(c_order in orders_parsed2[ag]['completed'] for ag in orders_parsed2.keys()):
                                    orders_parsed2[split_line[2]]['incomplete'].append(current_orders[split_line[2]][0])
                        else:
                        
                            already_sensed_object = False
                            if "sense" in c_order and "room" not in c_order:
                                if c_order.split("_")[1] in objects_fully_sensed[split_line[2]]:
                                    already_sensed_object = True
                        
                            if c_order not in orders_parsed2[split_line[2]]['completed'] and not already_sensed_object:
                                orders_parsed2[split_line[2]]['incomplete'].append(c_order)

                    orders_to_agent = {}
                    orders_leader = ""
                    if split_line[2] == leader:
                        if leader == 'A':
                            if dtime in orders_by_human_agent.keys() and orders_by_human_agent[dtime] and orders_by_human_agent[dtime][0][0] == float(split_line[0]):
                                orders_to_agent = orders_by_human_agent[dtime].pop(0)[1]

                        elif leader == 'C':
                            
                            orders_to_agent, orders_leader = process_AI_orders_pd(messages["whole"])
                            #print(orders_to_agent)
                            
                    if orders_to_agent:
                        orders_given = True
                    else:
                        orders_given = False
                    for oa in orders_to_agent.keys():
                        if orders_to_agent[oa]:
                            
                            if oa != leader and (oa not in messages or not messages[oa]):
                                extra_features["communication_range_error"] += 1
                                
                            for ota in orders_to_agent[oa]:
                                if "sense" in ota:
                                    key_sense_track = ota + oa
                                    if key_sense_track not in sense_track:
                                        sense_track[key_sense_track] = 0
                                    sense_track[key_sense_track] += 1
                                
                            
                            if oa in current_orders and orders_to_agent[oa][0] not in current_orders[oa]:
                            
                                     
                            
                                orders_parsed2[oa]['orders_changed'].append(orders_to_agent[oa][0])
                            
                                if "follow" in current_orders[oa]:
                                    
                                    c_order = follow_to_carry(current_orders[oa][0], current_orders)
                                    if c_order:
                                        #obj_id = c_order.split("_")[!]
                                        #if obj_id in current_team.keys() and oa in current_team[obj_id]["followers"]:
                                            
                                        if not any(c_order in orders_parsed2[ag]['completed'] for ag in orders_parsed2.keys()):
                                            orders_parsed2[oa]['preempt'].append(orders_to_agent[oa][0])
                                else:         
                                    if orders_to_agent[oa][0] not in orders_parsed2[oa]['completed']:
                                        orders_parsed2[oa]['preempt'].append(orders_to_agent[oa][0])
                            
                            '''
                            if "follow" in orders_to_agent[oa][0]:  
                                agent_to_follow = orders_to_agent[oa][0].split("_")[1]
                                c_order = follow_to_carry(current_orders[oa][0], current_orders)
                                if c_order:
                                    obj_id = c_order.split("_")[1]
                                    if obj_id in current_team.keys() and agent_to_follow == current_team[obj_id]["leader"] and oa not in current_team[obj_id]["followers"]:
                                        if oa == current_team[obj_id]["leader"]:
                                        elif oa not in current_team[obj_id]["followers"]:
                                            current_team[obj_id]["followers"].append(oa)
                            ''' 
                            
                            if "carry" in orders_to_agent[oa][0] and orders_to_agent[oa][0] not in carry_order_to_first_carry.keys():
                                carry_order_to_first_carry[orders_to_agent[oa][0]] = {"start":float(split_line[0]) , "end":0}
                                #ob_id = orders_to_agent[oa][0].split("_")[1]
                                #current_team[ob_id] = {"leader": oa, "followers":[f for f in orders_to_agent.keys() if orders_to_agent[f][0] == "follow_" + f]}}
                                
                            current_orders[oa] = orders_to_agent[oa]
                            
                            #if dtime == "2025_07_15_15_51_02" and oa == "A":
                            #    print(split_line[0],orders_to_agent[oa], messages["whole"])
                            #    print(orders_parsed2[oa]["all"])
                            #    pdb.set_trace()

                            orders_parsed2[oa]["all"].append(orders_to_agent[oa][0])
                            if 'sense' in orders_to_agent[oa][0] and orders_to_agent[oa][0] not in orders_parsed2[oa]['unique_sense']:
                                orders_parsed2[oa]['unique_sense'].append(orders_to_agent[oa][0])
                            elif 'carry' in orders_to_agent[oa][0] and orders_to_agent[oa][0] not in orders_parsed2[oa]['unique_carry']:
                                orders_parsed2[oa]['unique_carry'].append(orders_to_agent[oa][0])
                    
                    if orders_leader:
                        current_orders["C"] = []
                    
                    #if leader == 'C':
                    #    for 

                    find_and_remove_carry_overlaps(orders_to_agent, current_orders)
                    
                    team_tracker.refresh(float(split_line[0]), current_orders, orders_to_agent, agents, objects_metadata)   
                       
                    #if dtime == "2025_08_11_20_35_54":
                    #    print(orders_to_agent, current_orders, {obj_id:team_tracker.current_team[obj_id]["timeline"] for obj_id in team_tracker.current_team.keys()})
                    #    pdb.set_trace()
                       
                    #if orders_given and dtime == "2025_08_11_20_35_54":
                    #    print(current_orders)
                    #    print(team_tracker.unique_team_configurations())
                        
                    if split_line[2] in orders_parsed.keys():
                        regex_str = r"I'm going to ask for help to carry object (\d+)"
                        rematch = re.search(regex_str,messages["whole"])
                        function_action = ""
                        if rematch:
                            help_carry_dialogue += 1
                            orders_parsed[split_line[2]]["unspecified_team"] += 1
                            function_action = "carry_" + rematch.group(1)
                            
                        
                        regex_str = r"I'm going to collect object (\d+)"
                        rematch = re.search(regex_str,messages["whole"])
                        if rematch:
                            function_action = "carry_" + rematch.group(1)
                            
                        regex_str = r"I'm going to sense all objects in room (\d+)"
                        rematch = re.search(regex_str,messages["whole"])
                        if rematch:
                            function_action = "sense_room" + rematch.group(1)
                            
                        regex_str = r"I'm going to sense object (\d+)"
                        rematch = re.search(regex_str,messages["whole"])
                        if rematch:
                            function_action = "sense_" + rematch.group(1)
                            
                        regex_str = r"I'll follow you (\w)"
                        rematch = re.search(regex_str,messages["whole"])
                        if rematch and rematch.group(1) in ['A','B','C','D']:
                            function_action = "follow_" + rematch.group(1)    
                        
                            
                        regex_str = r"I will follow your lead, (\w)"
                        rematch = re.search(regex_str,messages["whole"])
                        if rematch:
                            function_action = "follow_" + rematch.group(1)    
                            
                            
                        if function_action and current_orders and split_line[2] in current_orders and function_action not in current_orders[split_line[2]]:
                            orders_parsed[split_line[2]]["misunderstood"] += 1
                            
                        if "I don't know where that object is" in messages["whole"]:
                            orders_parsed[split_line[2]]["incomplete_orders"] += 1
                        
                        if 'Order completed' in messages["whole"]:
                            orders_parsed[split_line[2]]["order_completes"] += 1
                        
                        if function_action and function_action != orders_parsed[split_line[2]]["previous_order"]:
                            orders_parsed[split_line[2]]["previous_order"] = function_action
                            if orders_parsed[split_line[2]]["previous_order"]:
                                orders_parsed[split_line[2]]["changes_of_order"] += 1
                            if "carry" in function_action:
                                if function_action not in orders_parsed[split_line[2]]["unique_carrying_orders"]:
                                    orders_parsed[split_line[2]]["unique_carrying_orders"].append(function_action)
                                orders_parsed[split_line[2]]["carrying_orders"].append(function_action)
                            elif "sense" in function_action:
                                if function_action not in orders_parsed[split_line[2]]["unique_sensing_orders"]:
                                    orders_parsed[split_line[2]]["unique_sensing_orders"].append(function_action)
                                orders_parsed[split_line[2]]["sensing_orders"].append(function_action)
                            elif "follow" in function_action:
                                orders_parsed[split_line[2]]["follow_orders"].append(function_action)

                    
                    
                        
                    
                    message_history.append(mess)
                    for m_key in messages.keys():
                        if m_key != "whole":
                            rematch = re.search(regex_str,messages[m_key])
                            if rematch:
                                messages[m_key] = messages[m_key].replace(rematch.group(0), "").strip()
                            
                            if messages[m_key]:
                                pass
                                #if messages["whole"] not in messages[m_key]:
                                #    pdb.set_trace()
                                if "All" not in m_key:
                                    chat_log[-1]["receivers"].append(m_key)
                                #print("Received by " + m_key, messages[m_key])
                
            elif int(split_line[1]) == 0 and not dtime in extra_resets.keys():
            

                first_bracket = new_line.find('{')
                second_bracket = new_line.rfind('}')
                
                metadata = eval(new_line[first_bracket:second_bracket+1].replace('\\','').replace("false","False").replace("true","True"))


                last_time = float(split_line[0])
                
                for ob in metadata["metadata"]:
                    if ob[0]:
                    
                        new_position = np.array(ob[2:4])
                    
                        if ob[1] not in agents.keys():
                            agents[ob[1]] = {"last_position": np.array([]), "distance_traveled": 0.0}
                            
                        if agents[ob[1]]["last_position"].size > 0:
                            distance_travel = float(np.linalg.norm(new_position - agents[ob[1]]["last_position"]))
                            agents[ob[1]]["distance_traveled"] += distance_travel
                            
                            try:
                                if distance_travel < 0.01: # and robots_type[ob[1]] == "human":
                                    idle_time += last_time - previous_last_time
                                    idle_time_distribution[ob[1]] += round(last_time - previous_last_time,2)
                                    
                            except:
                                pdb.set_trace()
                            
                        agents[ob[1]]["last_position"] = np.array(new_position)
                    
                    else:
                        new_position = np.array(ob[2:4])
                        
                        if ob[1] not in objects_metadata.keys():
                            objects_metadata[ob[1]] = {"last_position": np.array([])}
                        objects_metadata[ob[1]]["last_position"] = np.array(new_position)
        
                frequency_sample["current_period"] += last_time - previous_last_time
                
                previous_last_time = last_time
                
                
                if frequency_sample["current_period"] > frequency_sample["period"]:
                    #if counter_ai_only:
                    #    pdb.set_trace()
                    extra_features["team_dispersion"].append(dispersion_scalar(agents))
                    extra_features["dispersion_to_leader"].append(dispersion_to_leader(agents, leader))
                    frequency_sample["current_period"] = 0
                
            elif int(split_line[1]) == 8 and not dtime in extra_resets.keys():
                
                first_bracket = new_line.find('[')
                second_bracket = new_line.find(']')
                third_bracket = new_line.rfind('[')
                fourth_bracket = new_line.rfind(']')
                
                last_comma = new_line.rfind(',')
                
                questions = eval(new_line[first_bracket:second_bracket+1])
                answers = eval(new_line[third_bracket:fourth_bracket+1])
                
                if not average_survey_results:
                    average_survey_results = {q:None for q in questions}
                
                NASA.append(0)
                for q_idx in range(len(questions)):
                    
                    if answers[q_idx]:
                        val = float(answers[q_idx])
                        if not (val > 100 and q_idx <= 13):
                        
                            if q_idx > 13:
                                value = (int(answers[q_idx])-1)*5
                                if "Physical Demand" not in questions[q_idx]:
                                    NASA[-1] += value
                            else:
                                value = int(answers[q_idx])
                                if "Unresponsive" in questions[q_idx] or "Malfunction" in questions[q_idx] or "Have errors" in questions[q_idx]:
                                    value = 100 - value
                            
                                
                                
                            if average_survey_results[questions[q_idx]] is None:
                                average_survey_results[questions[q_idx]] = [value]
                            else:
                                average_survey_results[questions[q_idx]].append(value)
                
                NASA[-1] = NASA[-1]/5
                
                token_here = new_line[last_comma+1:].strip()
                
                try:
                    survey_results.append({"agent_id": tokens[token_here], "token":token_here, "questions": questions, "answers": [int(a) if a else None for a in answers]})
                except:
                    pdb.set_trace()
                
             
            elif int(split_line[1]) == 5 and not dtime in extra_resets.keys():
                first_bracket = new_line.find('{')
                second_bracket = new_line.rfind('}')
                
                results = eval(new_line[first_bracket:second_bracket+1])
                
                agent_id = split_line[2]
                
                if results["token"]:
                    tokens[results["token"]] = agent_id
                
                if agent_id in payment.keys() and results["team_end_time"]:
                
                    payment[agent_id].append(results["individual_payment"])
               
        
            new_line = log_file.readline()
     
            
        '''
        if metadata:
            print("LAST METADATA", metadata)
            print(scenario_options["robots_type"])
            print(last_time)
            print(latest_item_info)
        '''        
        
        if leader == 'C' and not ai_only:
            
            d3 = "../ai_controller/log/" + dtime + "_" + leader + "_monitor.txt"
            if os.path.exists(d3):
            
                log_file = open(d3)
                new_line = log_file.readline()
                while new_line:
                
                    split_line = new_line.strip().split(',')
                    agent_id = split_line[2]
                    
                    if int(split_line[1]) == 0:
                    
                        first_bracket = new_line.find('{')
                        second_bracket = new_line.rfind('}')
                
                        info = eval(new_line[first_bracket:second_bracket+1])
                        
                        if agent_id not in agent_reliability.keys():
                            agent_reliability[agent_id] = info
                            #print("Reliability for agent " + agent_id + " at time " + str(split_line[0]), info)
                        else:
                            changed_reliability = []
                            for i in info.keys():
                            
                                if agent_reliability[agent_id][i] != info[i]:
                                    changed_reliability.append((i,info[i]))
                                    
                            #print("Changed reliability for agent " + agent_id + " at time " + str(split_line[0]), changed_reliability)
                            agent_reliability[agent_id] = info

                    new_line = log_file.readline()
            
            try:       
                reliability_per_agent = sum([sum(agent_reliability[agent_id][ar] for ar in agent_reliability[agent_id].keys())/len(agent_reliability[agent_id]) for agent_id in ["A","B"]])
            except:
                pdb.set_trace()
            extra_features["reliability_score"] = reliability_per_agent/2       
              
    
        if False:
            if strategy == 'AI leader':
                analyzed_agents = [leader]
            else:
                analyzed_agents = [a for a in robots_type.keys() if robots_type[a] == "ai"]
            
            for analyzed_a in analyzed_agents:
            
                
                d2 = "../ai_controller/log/" + dtime + "_" + analyzed_a + "_strategy.txt"
                
                if os.path.exists(d2):
                    try:
                        log_file = open(d2)
                    except:
                        pdb.set_trace()
                    new_line = log_file.readline()
                    while new_line:
                    
                        info_decision = eval(new_line.replace("false", "False").replace("true","True"))
                        #pdb.set_trace()
                        if not info_decision["type"]:
                            #print("Assignments by " + analyzed_a + " at time " + str(info_decision["time"]))
                            for i in info_decision["assignments"]:
                                #print(i[0], i[1])
                                pass
                                
                            #print("Rooms: ", info_decision["rooms"])
                            
                            s_agent = {}
                            for s in info_decision["sensed_objects"]:
                                if s[0] not in s_agent.keys():
                                    s_agent[s[0]] = []
                                s_agent[s[0]].append(s[1])
                            
                            #print("Sensed objects", s_agent)
                            
                        else:
                            suggestions_taken += 1
                            #print("Changed time for recs at time " + str(info_decision["time"]) + ":", info_decision["time_last_suggestion_interval"])
                    
                        new_line = log_file.readline()
                        
                
                d3 = "../ai_controller/log/" + dtime + "_" + analyzed_a + "_monitor.txt"
                
                if os.path.exists(d3):
                
                    log_file = open(d3)
                    new_line = log_file.readline()
                    while new_line:
                    
                        split_line = new_line.strip().split(',')
                        agent_id = split_line[2]
                        
                        if int(split_line[1]) == 0:
                        
                            first_bracket = new_line.find('{')
                            second_bracket = new_line.rfind('}')
                    
                            info = eval(new_line[first_bracket:second_bracket+1])
                            
                            if agent_id not in agent_reliability.keys():
                                agent_reliability[agent_id] = info
                                #print("Reliability for agent " + agent_id + " at time " + str(split_line[0]), info)
                            else:
                                changed_reliability = []
                                for i in info.keys():
                                
                                    if agent_reliability[agent_id][i] != info[i]:
                                        changed_reliability.append((i,info[i]))
                                        
                                #print("Changed reliability for agent " + agent_id + " at time " + str(split_line[0]), changed_reliability)
                                agent_reliability[agent_id] = info


                        
                        elif int(split_line[1]) == 1:
                        
                            first_bracket = new_line.find('{')
                            second_bracket = new_line.rfind('}')
                    
                            info = eval(new_line[first_bracket:second_bracket+1])
                            
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
                                
                        
                        elif int(split_line[1]) == 2:
                        
                            if split_line[3] == 1:
                                pass
                                #print("Agent " + agent_id + " moved in the wrong direction")
                            elif split_line[3] == 2:
                                pass
                                #print("Agent " + agent_id + " arrived to its destination")
                            
                        new_line = log_file.readline()
                    
                
                d4 = "../ai_controller/log/" + dtime + "_" + analyzed_a + "_LLM.txt"
                
                if os.path.exists(d4):
                
                    log_file = open(d4)
                    new_line = log_file.readline()
                    while new_line:
                        
                        llm_response = eval(new_line)
                        
                        #print("Used LLM at time " + llm_response["time"])
                        
                        new_line = log_file.readline()
        
        if not previous_file and metadata and average_survey_results and not (ai_only and counter_ai_only in skip_sessions_ai_only[dtime]):
            print("NUMBER OF OBJECTS IN GOAL", metrics["goal_objects"])  
            dangerous = []
            total_dangerous_locations = []
            benign = []
            total_dangerous = []
            total_objects = []
            all_dangerous_weights = []
            weights = []
            dangerous_plus_weight = []
            all_objects_locations = {}
            all_objects_weights_dist = {}
            all_objects_danger_status_dist = {}
            groundtruth = {}
            total_benign = []
            
            
            for m in metadata["metadata"]:
                #For those objects that somehow ended being dropped just before the goal area
                if not m[0] and np.linalg.norm(np.array(m[2:4])-np.array([0,5])) < settings["goal_radius"] and not legacy:
                    object_id = m[1]
                    
                    #if grabbed_objects_tracker[object_id]["start_time"] > grabbed_objects_tracker[object_id]["end_time"]:
                    #    grabbed_objects_tracker[object_id]["end_time"] = 1200
                    
                    agent_id = grabbed_objects_tracker[object_id]["agent_id"]
                    metrics["goal_objects"].append(object_id)
                    grabbed_objects_tracker[object_id]["goal"] = True
                    

                    c_order = 'carry_' + object_id
                    try:
                        if c_order not in orders_parsed2[agent_id]['completed']:
                            orders_parsed2[agent_id]['completed'].append(c_order)
                    except:
                        pdb.set_trace()
            
                if not m[0] and any(m[1] == object_id for object_id in metrics["goal_objects"]):
                    if m[5] == 1:
                        benign.append(m[1])
                    else:
                        dangerous.append(m[1])
                        dangerous_plus_weight.append(m[4])
                        
                        
                    weights.append(m[4])
                
                if not m[0]:
                    total_objects.append(m[1])
                    all_objects_locations[m[1]] = convert_to_grid_coordinates(m[2:4])
                    all_objects_danger_status_dist[m[1]] = m[5]-1 #Careful with this 2 -> 1, 1 -> 0
                    groundtruth[m[1]] = m[5]
                    all_objects_weights_dist[m[1]] = m[4]
                    if m[5] == 2:
                        total_dangerous.append(m[1])
                        total_dangerous_locations.append(convert_to_grid_coordinates(m[2:4]))
                        all_dangerous_weights.append(m[4])
                    else:
                        total_benign.append(m[1])
                            
            
            print("Dangerous:", len(dangerous), "Benign:", len(benign), "Total dangerous:", len(total_dangerous), "Total objects", len(total_objects))    
            print("Weights:", weights)    
            
            chat_log.append({"type":"final", "description": "Only the objects in \"dangerous_objects\" are truly dangerous, the others are benign", "dangerous_objects": total_dangerous})
            
            final_metrics["dangerous_objects"].append(len(dangerous))
            #final_metrics["dangerous_objects"].append(sum(dangerous_plus_weight))
            final_metrics["objects_carried"].append(len(dangerous)+len(benign))
            print({obj_id:team_tracker.current_team[obj_id]["timeline"] for obj_id in team_tracker.current_team.keys()})
            print(team_tracker.ordered_formation_counts())
            print(team_tracker.get_distance_log())
            
            print({agent_id:orders_parsed2[agent_id]['orders_changed'] for agent_id in orders_parsed2.keys()})
            #print({agent_id:orders_parsed2[agent_id]['incomplete'] for agent_id in orders_parsed2.keys()})
            #print({agent_id:",".join(orders_parsed2[agent_id]['unique_carry']) + "," + ",".join(orders_parsed2[agent_id]['unique_sense']) for agent_id in orders_parsed2.keys()})
            print(dtime)
            #if dtime == "2025_08_11_20_35_54":
            #    print(orders_to_agent, current_orders, {obj_id:team_tracker.current_team[obj_id]["timeline"] for obj_id in team_tracker.current_team.keys()})
            #    pdb.set_trace()
            changes_per_obj = count_team_changes(team_tracker)
            total_changes_in_team = sum(changes_per_obj.values())
            print(total_changes_in_team)
            
            carry_global = {}
            sense_local = {}
            repeated_orders = 0
            for agent_id in orders_parsed2.keys():
                sense_local[agent_id] = {}
                for order in orders_parsed2[agent_id]['all']:
                    if "carry" in order:
                        if order not in carry_global.keys():
                            carry_global[order] = 0
                        else:
                            carry_global[order] += 1
                            repeated_orders += 1
                    elif "sense" in order:
                        if order not in sense_local[agent_id].keys():
                            sense_local[agent_id][order] = 0
                        else:
                            sense_local[agent_id][order] += 1
                            repeated_orders += 1 
            
            
            
            final_metrics["weight_distribution_carried"].append(weights)
            final_metrics["weight_sum"].append(sum([scenario_options["objects"][int(ob)][1] for ob in objects_grabbed_total if ob in total_dangerous]))
            
            object_sensed_per_agent = {}
            
            for ob_id in total_objects:
                object_sensed_per_agent[ob_id] = 0
            
            for agent_id in objects_fully_sensed.keys():
                for ob_id in objects_fully_sensed[agent_id]:
                    object_sensed_per_agent[ob_id] += 1
            
            num_sensed = {i: 0 for i in range(5)}
            for ob_id in object_sensed_per_agent.keys():
                num_sensed[object_sensed_per_agent[ob_id]] += 1
               
            for time_sensed in num_sensed:
                num_sensed[time_sensed] /= len(total_objects)
                num_sensed[time_sensed] = round(num_sensed[time_sensed],2)
                
            print("Number of times an object was sensed:", num_sensed)
            
            final_metrics["distribution_sensed"].append(num_sensed)
            
            total_distance_traveled = 0
            for ag_key in agents.keys():
                total_distance_traveled += agents[ag_key]["distance_traveled"]
                travel_distance_distribution[ag_key] = round(agents[ag_key]["distance_traveled"],2)
            
            print("Total distance traveled:", total_distance_traveled)
            
            final_metrics["total_distance_traveled"].append(total_distance_traveled)
            final_metrics["accidental_drop"].append(len(accidentally_dropped_objects))
            final_metrics["order_completed"].append(order_completed)
            
            
            for a_key in average_survey_results.keys():
                if average_survey_results[a_key] is not None:
                    try:
                        if a_key not in final_average_survey_results:
                            final_average_survey_results[a_key] = []
                        final_average_survey_results[a_key].append(sum(average_survey_results[a_key])/len(average_survey_results[a_key]))
                    except:
                        pdb.set_trace()
            
     
            ambiguous_results = evaluate_objects(
                known_objects_to_leader,
                groundtruth,
                max_readings=4,
                entropy_epistemic_thresh_bits=0.9
            )
            from pprint import pprint
            pprint(ambiguous_results["summary"])
            
            print({obj_id:ambiguous_results["per_object"][obj_id] for obj_id in ambiguous_results["per_object"].keys() if ambiguous_results["per_object"][obj_id]["n_readings"] > 0})
            #pdb.set_trace()
            
            try:
                frac_dangerous, ambigous_dangerous = analyze_ambiguous_cases(ambiguous_results["per_object"], entropy_thresh=0.9)
            except:
                pdb.set_trace()
            
            number_ambigous_dangerous = len(ambigous_dangerous)
            
            res = find_high_conf_misclassifications_fully_sensed(
                ambiguous_results["per_object"],
                q_low=0.10,             # confident-benign cutoff
                q_high=0.90,            # confident-dangerous cutoff
                use_margin=False        # set True to use margin instead: |q-0.5| >= margin_delta
            )

            
            benign_as_dangerous = []
            dangerous_as_benign = []
            if res["false_negatives"] or res["false_positives"]:
                actual_misclassifications = False
                print("Summary:", res["summary"])
                print("\nHigh-confidence FALSE NEGATIVES (dangerous classified as benign):")
                for r in res["false_negatives"]:
                    if r['n_readings'] > 2:
                        print(r)
                        if r['obj'] not in ambigous_dangerous:
                            ambigous_dangerous.append(r['obj'])
                        actual_misclassifications = True
                        dangerous_as_benign.append(r['obj'])

                print("\nHigh-confidence FALSE POSITIVES (benign classified as dangerous):")
                for r in res["false_positives"]:
                    if r['n_readings'] > 2:
                        print(r)
                        benign_as_dangerous.append(r['obj'])
                        actual_misclassifications = True
                        
                #if actual_misclassifications:
                #    pdb.set_trace()
            
            number_false_negatives = len(dangerous_as_benign)
            
            #if benign_as_dangerous or ambigous_dangerous:
            #    print(dtime)
            #    print(benign_as_dangerous)
            #    print(ambigous_dangerous)
            #    pdb.set_trace()
            final_metrics["frac_dangerous_ambiguous"].append(frac_dangerous)
            final_metrics["frac_benign_as_dangerous"].append(len(benign_as_dangerous)/len(total_benign))
     
            all_dangerous_weights_complexity = []
            all_dangerous_weights_complexity_no_filter = []
     
            for t in total_dangerous:
                all_dangerous_weights_complexity_no_filter.append(all_objects_weights_dist[t])
                if t not in ambigous_dangerous:
                    all_dangerous_weights_complexity.append(all_objects_weights_dist[t])
     
            complexity_reduction = 0
            complexity = 0
            
            complexity_reduction_no_filter = 0
            complexity_no_filter = 0
            
            object_carried_distance_distribution = {}
            object_carried_distance_distribution_no_filter = {}
            for d_idx in range(len(total_dangerous)):
                
                for ob in scenario_options["objects"]:
                    if total_dangerous[d_idx] == ob[0]:
                        goal = convert_to_grid_coordinates([ob[3]["x"],ob[3]["z"]])
                        path = findPath(start, goal, occ_map)
                        
                        if ob[0] not in ambigous_dangerous: 
                            complexity += len(path)*ob[1]
                        
                        complexity_no_filter += len(path)*ob[1]
                        
                        if ob[0] in dangerous:
                        
                            if ob[0] not in ambigous_dangerous:
                                complexity_reduction += len(path)*ob[1] #len(path)*2*ob[1]
                                object_carried_distance_distribution[ob[0]] = len(path)
                            
                            
                            complexity_reduction_no_filter += len(path)*ob[1] #len(path)*2*ob[1]
                            object_carried_distance_distribution_no_filter[ob[0]] = len(path)
                        else:
                            path2 = findPath(start, total_dangerous_locations[d_idx], occ_map)
                            
                            if ob[0] not in ambigous_dangerous:
                                object_carried_distance_distribution[ob[0]] = 0
                            
                            object_carried_distance_distribution_no_filter[ob[0]] = 0
                            if len(path) - len(path2) > 0:
                                if ob[0] not in ambigous_dangerous:
                                    complexity_reduction += (len(path) - len(path2))*ob[1] #len(path)*ob[1] + (len(path) - len(path2))*ob[1]
                                    object_carried_distance_distribution[ob[0]] = (len(path) - len(path2))
                                
                                complexity_reduction_no_filter += (len(path) - len(path2))*ob[1] #len(path)*ob[1] + (len(path) - len(path2))*ob[1]
                                object_carried_distance_distribution_no_filter[ob[0]] = (len(path) - len(path2))
            

                
            total_speed_carrying_dangerous = []
            total_speed_carrying_benign = []
            total_speed_carrying_distribution = {}
            object_time_dict = {}
            object_danger_dict = {}
            object_weight_dict = {}
            weights_carrying_dangerous = 0
            
            weights_carrying_dangerous_goal = 0
            
            false_negative = [t for t in total_dangerous if object_sensed_per_agent[t]]
            false_positive = []
            true_positive = []
            
            false_negative_goal = [t for t in total_dangerous if object_sensed_per_agent[t]]
            false_positive_goal = []
            true_positive_goal = []
            
            #For speed of carrying
            for ob in scenario_options["objects"]:
                if ob[0] in grabbed_objects_tracker.keys():
                
                    
                    if ob[0] in benign_as_dangerous:
                        pass #Only occurs when AI are leaders
                        
                    object_row = carrying_df[(carrying_df["file"] == dtime + ".json") & (carrying_df["object_id"] == int(ob[0]))]
                    
                    if not any(carrying_df["file"] == dtime + ".json"): #Check process this files
                        print("Need to process file: " + dtime)
                        carrying_time = grabbed_objects_tracker[ob[0]]["end_time"] - grabbed_objects_tracker[ob[0]]["start_time"]
                        object_row = pd.DataFrame({"carrying_time":[carrying_time]})
                    
                    
                    
                    
                    goal = convert_to_grid_coordinates([ob[3]["x"],ob[3]["z"]])
                    path = findPath(start, goal, occ_map)
                    
                    distance = 0
                    if ob[0] in dangerous or ob[0] in benign:
                        distance = len(path)
                        
                        if ob[0] in dangerous:
                            if ob[0] in false_negative:
                                false_negative_goal.remove(ob[0])
                                true_positive_goal.append(ob[0])
                            weights_carrying_dangerous_goal += ob[1]
                        else:
                            false_positive_goal.append(ob[0])
                            
                        
                    else:
                        
                        d_location = all_objects_locations[ob[0]]
                    
                        path2 = findPath(start, d_location, occ_map)
                            
                        if len(path) - len(path2) > 0:
                            distance = (len(path) - len(path2))
                            
                    '''
                    path3 = findPath(goal, all_objects_locations[ob[0]], occ_map)
                    path2 = (len(path) - len(path3))
                    '''
                    
                    if grabbed_objects_tracker[ob[0]]["start_time"] > grabbed_objects_tracker[ob[0]]["end_time"]:
                        grabbed_objects_tracker[ob[0]]["end_time"] = 1200
                    
                    carrying_time = grabbed_objects_tracker[ob[0]]["end_time"] - grabbed_objects_tracker[ob[0]]["start_time"]
                    object_row = pd.DataFrame({"carrying_time":[carrying_time]})
                    
                    #if ob[1] > 1:
                    #total_speed_carrying.append(path2/(grabbed_objects_tracker[ob[0]]["end_time"] - grabbed_objects_tracker[ob[0]]["start_time"]))
                    
                    if distance:
                        try:
                            if pd.isna(object_row["carrying_time"].iloc[0]):
                                pdb.set_trace()
                        except:
                            pdb.set_trace()
                        speed_value = object_row["carrying_time"].iloc[0]/distance #*ob[1]/distance
                        
                        if not speed_value == float("inf"):
                            if speed_value < 0 or math.isnan(speed_value):
                                pdb.set_trace()
                            
                            object_time_dict[ob[0]] = round(speed_value,2)
                            object_danger_dict[ob[0]] = ob[2]
                            object_weight_dict[ob[0]] = ob[1]
                            
                            total_speed_carrying_distribution[ob[0]] = speed_value
                            
                            if ob[0] in accidentally_dropped_objects:
                                extra_features["drop_per_distance"][ob[0]] = accidentally_dropped_objects.count(ob[0])/distance
                            
                            if ob[2] == 2:
                                total_speed_carrying_dangerous.append(speed_value)
                                #if ob[1] > 1:
                                weights_carrying_dangerous += ob[1]
                                
                                if ob[0] in false_negative:
                                    false_negative.remove(ob[0])
                                    true_positive.append(ob[0])
                                
                            else:
                                total_speed_carrying_benign.append(speed_value)
                                false_positive.append(ob[0])
                        #total_speed_carrying.append((grabbed_objects_tracker[ob[0]]["end_time"] - grabbed_objects_tracker[ob[0]]["start_time"]))
                    

            final_metrics["weight_sum"][-1] = weights_carrying_dangerous
            average_speed_carrying_dangerous = 0 #float("nan")
            if total_speed_carrying_dangerous:
                average_speed_carrying_dangerous = sum(total_speed_carrying_dangerous)/len(total_speed_carrying_dangerous)
            average_speed_carrying_benign = 0 #float("nan")
            if total_speed_carrying_benign:
                average_speed_carrying_benign = sum(total_speed_carrying_benign)/len(total_speed_carrying_benign)
                
            p1_list = [p[2] for p in scenario_options["robots_type"]]
            p2_list = [p[3] for p in scenario_options["robots_type"]]
            
            r = len(total_dangerous)/len(scenario_options["objects"])
            
            H_res = expected_posterior_entropy(r, p1_list, p2_list, len(scenario_options["robots_type"]))

            H_prior = -r*math.log2(r) - (1-r)*math.log2(1-r) if 0<r<1 else 0
            H_norm = H_res / H_prior if H_prior>0 else 0
            
           
            
            final_metrics["complexity"].append(complexity/sum(all_dangerous_weights))
            #final_metrics["complexity"].append(complexity)
            final_metrics["complexity_reduction"].append(complexity_reduction/sum(all_dangerous_weights_complexity))
            #final_metrics["complexity_reduction"].append(complexity_reduction)
            final_metrics["sensing_complexity"].append(H_norm)
            final_metrics["name"].append(dtime)   
            final_metrics["strategy"].append(strategy)
            final_metrics["suggestions_taken"].append(suggestions_taken)
            final_metrics["suggestions_given"].append(suggestions_given)
            final_metrics["idle_time"].append(idle_time)
            final_metrics["object_carry_speed"].append(0) #average_speed_carrying)
            
            final_metrics["object_dangerous_time"].append(average_speed_carrying_dangerous) #sum(total_speed_carrying_dangerous))
            final_metrics["object_benign_time"].append(sum(total_speed_carrying_benign))
            
            final_metrics["help_carry_dialogue"].append(help_carry_dialogue)
            final_metrics["objects_carried_distribution"].append(object_time_dict)
            final_metrics["objects_carried_distribution_danger"].append(object_danger_dict)
            final_metrics["objects_carried_distribution_weight"].append(object_weight_dict)
            final_metrics["idle_time_distribution"].append(idle_time_distribution)
            final_metrics["travel_distance_distribution"].append(travel_distance_distribution)
            final_metrics["carry_order_to_first_carry"].append({cc.split('_')[1]:round(carry_order_to_first_carry[cc]["end"] - carry_order_to_first_carry[cc]["start"],2) for cc in carry_order_to_first_carry.keys() if carry_order_to_first_carry[cc]["end"]})
            final_metrics["num_times_sensed_before_carry"].append(num_times_sensed_before_carry)
            if ai_only or legacy:
                final_metrics["NASA"].append(0)
                final_metrics["PTT"].append(0)
            else:
                final_metrics["NASA"].append(sum(NASA)/len(NASA))
                final_metrics["PTT"].append(sum(PTT)/len(PTT))
            final_metrics["lost_track"].append(lost_track)
            
            if leader == 'A':
                leader_type = 'human'
            else:
                leader_type = 'ai'
            
            
            if not legacy and team_strategy["suggestions"]:
                suggestions = "Yes"
            else:
                suggestions = "No"
            
            
            if ai_only:
                layout_num = str(ai_layouts[ai_layout_idx])
            elif legacy:
                layout_num = "0"
            else:
                layout_num = str(layouts[layout_ids[current_layout_id // 4]-1][current_layout_id % 4])
                
             
            if leader_type == "human":
                orders_to_other_type = sum(len(orders_parsed2[oa]["all"]) for oa in orders_parsed2.keys() if oa in ["C","D"])
                completed_other_type = sum(len(orders_parsed2[oa]["completed"]) for oa in orders_parsed2.keys() if oa in ["C","D"])
            else:
                orders_to_other_type = sum(len(orders_parsed2[oa]["all"]) for oa in orders_parsed2.keys() if oa in ["A","B"])
                completed_other_type = sum(len(orders_parsed2[oa]["completed"]) for oa in orders_parsed2.keys() if oa in ["A","B"])
            
            drop_per_distance = 0
            if extra_features["drop_per_distance"]:
                drop_per_distance = sum(extra_features["drop_per_distance"][e] for e in extra_features["drop_per_distance"].keys())/len(extra_features["drop_per_distance"].keys())
            
            
            
            if extra_features["uncertainty_at_grabbing"]:
                uncertainty_at_grabbing = sum(extra_features["uncertainty_at_grabbing"])/len(extra_features["uncertainty_at_grabbing"])
            else:
                uncertainty_at_grabbing = float("nan")
                
            if extra_features["probability_at_grabbing"]:
                probability_at_grabbing = sum(extra_features["probability_at_grabbing"])/len(extra_features["probability_at_grabbing"])
            else:
                probability_at_grabbing = float("nan")
            
            
            performance_no_filter = complexity_reduction_no_filter/sum(all_dangerous_weights_complexity_no_filter)
            
            session_dict = {"team_id": current_layout_id // 4, "session": current_layout_id % 4, "leader_type": leader_type, "performance": final_metrics["complexity_reduction"][-1], "total_orders": sum(len(orders_parsed2[oa]["all"]) for oa in orders_parsed2.keys() if oa != leader), "misunderstood_orders": sum(orders_parsed[oa]["misunderstood"] for oa in orders_parsed.keys() if oa in ['C','D']), "orders_changed": sum(len(orders_parsed2[oa]["orders_changed"]) for oa in orders_parsed2.keys() if oa != leader), "completed_orders": sum(len(orders_parsed2[oa]["completed"]) for oa in orders_parsed2.keys() if oa != leader), "orders_to_AI": sum(len(orders_parsed2[oa]["all"]) for oa in orders_parsed2.keys() if oa in ['C','D']), "incomplete_orders": sum(len(orders_parsed2[oa]["incomplete"]) for oa in orders_parsed2.keys() if oa != leader), "underspecified": sum(orders_parsed[oa]["unspecified_team"]+orders_parsed[oa]["incomplete_orders"] for oa in orders_parsed2.keys() if oa in ['C','D']), "dist_by_agent": final_metrics["travel_distance_distribution"][-1], "weight_by_object": all_objects_weights_dist, "danger_level_by_object": all_objects_danger_status_dist, "carried_objects": list(final_metrics["objects_carried_distribution_weight"][-1].keys()), "carried_distance_by_object": object_carried_distance_distribution, "time_over_distance": total_speed_carrying_distribution, "time_first_order_to_carry": final_metrics["carry_order_to_first_carry"][-1], "sensings_per_object": object_sensed_per_agent, "layout": layout_num, "frac_dangerous_ambiguous": final_metrics["frac_dangerous_ambiguous"][-1], "frac_benign_as_dangerous": final_metrics["frac_benign_as_dangerous"][-1], "suggestions": suggestions, "suggestions_exchanged":suggestions_taken, "suggestions_accepted": suggestions_given, "lost_track": lost_track, "total_messages": len(message_history), "NASA": final_metrics["NASA"][-1], "PTT": final_metrics["PTT"][-1], "team_dispersion": sum(extra_features["team_dispersion"])/len(extra_features["team_dispersion"]), "dispersion_to_leader": sum(extra_features["dispersion_to_leader"])/len(extra_features["dispersion_to_leader"]), "total_changes_in_team": total_changes_in_team, "mean_distance_order_carry": overall_mean_distance(team_tracker), "team_carrying_composition_AI": average_team_composition_fraction_global(team_tracker,{r:True if robots_type[r] == "human" else False for r in robots_type.keys()}, all_objects_weights_dist)["avg_ai_fraction"], "times_reprimanded":extra_features["times_reprimanded"], "communication_range_error":extra_features["communication_range_error"], "drop_per_distance":drop_per_distance, "completed_other_type": completed_other_type, "uncertainty_at_grabbing": uncertainty_at_grabbing, "probability_at_grabbing": probability_at_grabbing, "reliability_score": extra_features["reliability_score"], "name": dtime, "repeated_orders": repeated_orders, "dangerous_weighted_work": complexity_reduction, "relative_performance": complexity_reduction/complexity, "complexity": complexity, "normalized_complexity": final_metrics["complexity"][-1], "complexity_no_filter": complexity_no_filter, "performance_no_filter": performance_no_filter, "carried_distance_by_object_no_filter": object_carried_distance_distribution_no_filter, "dangerous_weighted_work_no_filter": complexity_reduction_no_filter, "relative_performance_no_filter": complexity_reduction_no_filter/complexity_no_filter, "true_positive": len(true_positive), "false_positive":len(false_positive), "false_negative":len(false_negative), "sensitivity_filter": final_metrics["complexity_reduction"][-1] - performance_no_filter, "weight_sum": weights_carrying_dangerous, "true_positive_goal": len(true_positive_goal), "false_positive_goal":len(false_positive_goal), "false_negative_goal":len(false_negative_goal), "weight_sum_goal": weights_carrying_dangerous_goal, "number_false_negatives": number_false_negatives, "number_ambigous_dangerous": number_ambigous_dangerous, "total_dangerous": len(total_dangerous), "total_benign": len(total_benign), "number_ambiguous": len(ambigous_dangerous), "strategy": strategy, "team_composition": team_composition}


            
            subjective_dict = {}
            for agp in ["A","B"]:
                #subjective_dict = {"agent_id": agp, "session":session_dict["session"], "team_id":session_dict["team_id"]}
                if agp == "A":
                    prefix = "leader_"
                else:
                    prefix = "nonleader_"
                
                for surv in survey_results[-2:]:
                    if surv["agent_id"] == agp:
                        NASA_calc = 0
                        for s_idx in range(len(surv["questions"])):
                            val = surv["answers"][s_idx]
                            question = surv["questions"][s_idx].strip()
                            if val is not None and not (val > 100 and s_idx <= 13):
                        
                                if s_idx > 13:
                                    value = (int(val)-1)*5
                                    if "Physical Demand" not in question:
                                        NASA_calc += value
                                else:
                                    value = int(val)
                                    if "Unresponsive" in question or "Malfunction" in question or "Have errors" in question:
                                        value = 100 - value
                            elif val is None or (val > 100 and s_idx <= 13):
                                value = float("nan")
                                print(session_dict["team_id"], session_dict["session"], question, val)
                            subjective_dict[prefix + question] = value
                        
                        subjective_dict[prefix + "NASA"] = NASA_calc/5
                         
                for tut in tutorial_results['answers'][-2:]:
                    if participants[tut[0]] == agp:
                        PTT_calc = 0
                        for s_idx in range(len(tutorial_results["questions"])):
                            if s_idx > 0:
                                if s_idx > 4: #This is where PTT starts
                                    if s_idx == 8: #Reverse
                                        value = 6-int(tut[s_idx])
                                        PTT_calc += value
                                    else:
                                        value = int(tut[s_idx])
                                        PTT_calc += value
                                        
                                else:
                                    value = int(tut[s_idx])
                            
                                
                                subjective_dict[prefix + tutorial_results["questions"][s_idx].strip()] = value
                        subjective_dict[prefix + "PTT"] = PTT_calc / 6
                
                    
            for f_key in subjective_dict.keys():
                session_dict[f_key] = subjective_dict[f_key]
            
            
            for f_key in final_average_survey_results.keys():
                session_dict[f_key] = final_average_survey_results[f_key][-1]
                
            
            for f_key in average_demographic_qs.keys():
                session_dict[f_key] = average_demographic_qs[f_key][-1]
            
            df_sessions.append(session_dict)

            print(orders_parsed)
            aggregated_orders_parsed = {"sense_complete":0, "carry_complete":0}
            for a in orders_parsed.keys():
                for k in orders_parsed[a].keys():
                    if k != 'previous_order' and k not in aggregated_orders_parsed.keys():
                        aggregated_orders_parsed[k] = 0
                
                    if k == "sensing_orders" or k == "carrying_orders" or k == 'follow_orders' or k == 'unique_sensing_orders' or k == 'unique_carrying_orders':
                        orders_parsed[a][k] = len(orders_parsed[a][k])
                    
                    if k != 'previous_order':   
                        aggregated_orders_parsed[k] += orders_parsed[a][k]
                        
                del orders_parsed[a]["previous_order"]
            
            for a in orders_parsed2.keys():
                for k in orders_parsed2[a].keys():
                    
                    if k == 'unique_sense' or k == 'unique_carry' or k == "preempt" or k == "incomplete" or k == 'orders_changed':
                    
                        if k not in aggregated_orders_parsed.keys():
                            aggregated_orders_parsed[k] = 0
                    
                        orders_parsed2[a][k] = len(orders_parsed2[a][k])
                        
                        aggregated_orders_parsed[k] += orders_parsed2[a][k]
                    elif k == "completed":
                        for o in orders_parsed2[a][k]:
                            if o in orders_parsed2[a]["all"]:
                                if "sense" in o:
                                    aggregated_orders_parsed["sense_complete"] += 1
                                elif "carry" in o:
                                    aggregated_orders_parsed["carry_complete"] += 1
            #print(aggregated_orders_parsed)
            

            final_metrics["orders_parsed"].append(aggregated_orders_parsed)

            try:
                final_metrics['experience_videogame'].append([int(p[2]) for p in tutorial_results['answers'][-2:]])
                final_metrics['experience_teamcollab'].append([int(p[3]) for p in tutorial_results['answers'][-2:]])
                final_metrics['experience_llm'].append([int(p[4]) for p in tutorial_results['answers'][-2:]])
                
            except:
                pdb.set_trace()  
                
            chat_log.append({"type":"score", "score": final_metrics["complexity_reduction"]})
            with open("chat_logs/" + dtime + ".json", "w") as json_file:
                json.dump(chat_log, json_file)           
            #print(final_average_survey_results)
            #print(final_metrics)
            #all_final_metrics.append(final_metrics)
            #pdb.set_trace()
            

            
            current_layout_id += 1
            
            if ai_only:
                ai_end = False
                dfile_idx -= 1
                counter_ai_only += 1
                ai_layout_tracker[ai_layouts[ai_layout_idx]] -= 1
                if new_line and not ai_layout_tracker[ai_layouts[ai_layout_idx]]:
                    ai_end = True  
                    counter_ai_only = 0 
                    ai_layout_idx += 1
                    dfile_idx += 1
                
        elif ai_only:
            counter_ai_only += 1  
            dfile_idx -= 1  
            ai_end = False
            
        if not new_line and ai_only:

            ai_end = True  
            counter_ai_only = 0 
            ai_layout_idx += 1
            dfile_idx += 1
            
                      
          
#pdb.set_trace()  
#df_sessions[df_sessions["performance"] > 2][["performance","layout"]].groupby("layout").mean()
#print(final_metrics["frac_dangerous_ambiguous"])
'''
        performance
layout             
1         12.438272
2         11.503968
3         14.019481
4          8.084211

'''
df_sessions = pd.DataFrame(df_sessions)
pdb.set_trace()

fe_df = build_features(df_sessions)

if ai_only:
    fe_df.to_csv("team_stats_ai.csv", index=False)
elif legacy:
    fe_df.to_csv("team_stats_legacy.csv", index=False)
else:
    fe_df.to_csv("team_stats.csv", index=False)

#participants_metadata = pd.DataFrame(participants_metadata)

#participants_metadata.to_csv("participants.csv",index=False)


pdb.set_trace()
df_outliers = detect_outliers(df_sessions, perf_col="performance", method="iqr")

'''
results = leadership_split_analysis(df_sessions)
print(results['model_overall'].summary())
print("Overall compact (all deltas):\n", results['overall_compact_all_deltas'])
print("Overall compact (same-leadership deltas):\n", results['overall_compact_samelead'])
print("HUMAN model:\n", results['by_leader']['human']['model'].summary())
#print("AI model:\n", results['by_leader']['ai']['model'].summary())
print(results['by_leader']['human']['attribution'].sort_values(['team_id','session']))
print(results['by_leader']['ai']['summary_by_team'])
print(results['by_leader']['human']['summary_by_team'])
'''

# 1) Non-contiguous, previous-with-same-leader (recommended)
res_same_prev = raw_delta_descriptives_flexible(
    df_sessions,
    pair_mode="prev",
    same_leadership_only=True,   # compare only within same leader type
    keep_flats=False             # drop ~0 changes for cleaner contrasts
)
res_same_prev['delta_df'].to_csv("team_flexible_transitions_samelead.csv", index=False)

# 2) previous
res_fixed_any = raw_delta_descriptives_flexible(
    df_sessions,
    pair_mode="fixed",
    same_leadership_only=False,   # compare only within same leader type
    keep_flats=False             # drop ~0 changes for cleaner contrasts
)


write_fixed_transition_latex_with_meta(
    res_fixed_any,
    out_tex="fixed_transition_corr_tables.tex",
    note_excluded="fixed pairs; excluding dangerous_weighted_distance",
    top_k_meta=50,
    abs_thresh_pair=0.0,
    diff_thresh_pair=0.40,
    top_k_pair=50,
    top_k_contrast=50
)


res_same_prev['delta_df'].to_csv("team_flexible_transitions_samelead.csv", index=False)
res_fixed_any['delta_df'].to_csv("team_fixed_transitions_anylead.csv", index=False)

with open("samelead_report.txt", "w") as f:
    f.write("=== HUMAN up vs down (same-leadership, non-contiguous) ===\n")
    f.write(res_same_prev['updown_by_leader']['human'].to_string())
    f.write("\n\n=== AI up vs down (same-leadership, non-contiguous) ===\n")
    f.write(res_same_prev['updown_by_leader']['ai'].to_string())

    f.write("\n\n=== HUMAN improved − declined contrast ===\n")
    f.write(res_same_prev['contrast_by_leader']['human'].to_string())
    f.write("\n\n=== AI improved − declined contrast ===\n")
    f.write(res_same_prev['contrast_by_leader']['ai'].to_string())

    f.write("\n\n=== HUMAN correlations with Δperformance ===\n")
    f.write(res_same_prev['corr_by_leader']['human'].to_string())
    f.write("\n\n=== AI correlations with Δperformance ===\n")
    f.write(res_same_prev['corr_by_leader']['ai'].to_string())

    f.write("\n\n=== Raw delta rows (head) ===\n")
    f.write(res_same_prev['delta_df'].head().to_string())

#LATEX STARTS
# --- inputs from your pipeline ---
# res_same_prev = raw_delta_descriptives_flexible(..., same_leadership_only=True, ...)
human_df = res_same_prev['corr_by_leader']['human'].copy()
ai_df    = res_same_prev['corr_by_leader']['ai'].copy()

# Optional: exclude tautological or noisy items (uncomment if desired)
EXCLUDE = set([
    "Δ_dangerous_weighted_distance",   # basically the metric itself
    # add more to exclude if you want
])

THRESH = 0.40        # abs correlation threshold for the first 2 tables
TOP_DIFF_K = 10      # how many most-different features to show in the third table
OUT_TEX = "samelead_corr_tables.tex"

# --- small helpers ---
def _prep_corr_df(df):
    out = df.copy()
    out = out[~out['feature'].isin(EXCLUDE)]
    # ensure numeric
    out['corr_with_Δperf'] = pd.to_numeric(out['corr_with_Δperf'], errors='coerce')
    # sort by absolute correlation (desc)
    out = out.assign(abs_corr=lambda d: d['corr_with_Δperf'].abs()) \
             .sort_values('abs_corr', ascending=False)
    return out

human_prepped = _prep_corr_df(human_df)
ai_prepped    = _prep_corr_df(ai_df)

# Filter by |r| > THRESH for the first two tables
human_filtered = human_prepped #[human_prepped['abs_corr'] > THRESH].drop(columns=['abs_corr'])
ai_filtered    = ai_prepped #[ai_prepped['abs_corr'] > THRESH].drop(columns=['abs_corr'])

# --- Build the "most different correlations" table ---
# Merge on feature (inner join so we compare same feature across leaderships)
diff_tbl = (human_prepped[['feature','corr_with_Δperf']].rename(columns={'corr_with_Δperf':'human_corr'})
            .merge(ai_prepped[['feature','corr_with_Δperf']].rename(columns={'corr_with_Δperf':'ai_corr'}),
                   on='feature', how='inner'))

# Compute differences and sort by absolute difference
diff_tbl['Δcorr_ai_minus_human'] = diff_tbl['ai_corr'] - diff_tbl['human_corr']
diff_tbl['|Δcorr|'] = diff_tbl['Δcorr_ai_minus_human'].abs()
diff_tbl = diff_tbl.sort_values('|Δcorr|', ascending=False).head(TOP_DIFF_K)

# Optional: nice rounding for display (we'll keep full precision in data)
human_out = human_filtered.copy()
human_out['corr_with_Δperf'] = human_out['corr_with_Δperf'].round(3)

ai_out = ai_filtered.copy()
ai_out['corr_with_Δperf'] = ai_out['corr_with_Δperf'].round(3)

diff_out = diff_tbl.copy()
for c in ['human_corr','ai_corr','Δcorr_ai_minus_human','|Δcorr|']:
    diff_out[c] = diff_out[c].round(3)

# --- Write LaTeX file ---
with open(OUT_TEX, "w") as f:
    f.write("%% Auto-generated tables: correlations by leadership (same-leadership contrast analysis)\n\n")

    f.write("%% -------- HUMAN-LED (|r| > %.2f) --------\n" % THRESH)
    f.write(human_out.to_latex(index=False, float_format="%.3f",
                               caption="Human-led: correlations with $\\Delta$performance (|r| > %.2f)" % THRESH,
                               label="tab:humn_corr_gt_%.0f" % (THRESH*100)))
    f.write("\n\n")

    f.write("%% -------- AI-LED (|r| > %.2f) --------\n" % THRESH)
    f.write(ai_out.to_latex(index=False, float_format="%.3f",
                            caption="AI-led: correlations with $\\Delta$performance (|r| > %.2f)" % THRESH,
                            label="tab:ai_corr_gt_%.0f" % (THRESH*100)))
    f.write("\n\n")

    f.write("%% -------- MOST DIFFERENT BETWEEN HUMAN vs AI --------\n")
    # Reorder columns for readability
    cols = ['feature', 'human_corr', 'ai_corr', 'Δcorr_ai_minus_human', '|Δcorr|']
    f.write(diff_out[cols].to_latex(index=False, float_format="%.3f",
                                    caption="Largest differences in correlations between AI-led and Human-led transitions",
                                    label="tab:largest_diff_corr"))
                                    
                                   
    f.write("\n\n")
    
    f.write("%% -------- MOST DIFFERENT (absolute correlation values) --------\n")
    diff_tbl_abs = diff_tbl.copy()
    diff_tbl_abs['Δ|corr|'] = diff_tbl_abs['ai_corr'].abs() - diff_tbl_abs['human_corr'].abs()
    diff_tbl_abs['|Δ|corr||'] = diff_tbl_abs['Δ|corr|'].abs()
    diff_tbl_abs = diff_tbl_abs.sort_values('|Δ|corr||', ascending=False).head(TOP_DIFF_K)

    cols_abs = ['feature', 'human_corr', 'ai_corr', 'Δ|corr|', '|Δ|corr||']
    f.write(diff_tbl_abs[cols_abs].to_latex(index=False, float_format="%.3f",
                                            caption="Largest differences in *absolute* correlations between AI-led and Human-led transitions",
                                            label="tab:largest_absdiff_corr"))
    f.write("\n")
    

TOP_K = 10


# Pull your existing contrast frames
human_contrast = res_same_prev['contrast_by_leader']['human'].copy()
ai_contrast    = res_same_prev['contrast_by_leader']['ai'].copy()

# Build top-k tables
h_col, human_top_impr, human_top_decl = _top_k_pos_neg(human_contrast, TOP_K)
a_col, ai_top_impr,    ai_top_decl    = _top_k_pos_neg(ai_contrast, TOP_K)

with open(OUT_TEX, "a") as f:  # append to your samelead_corr_tables.tex
    f.write("\n%% -------- HUMAN: Top-k improved vs declined (same-leadership) --------\n")
    f.write(human_top_impr.to_latex(
        index=False, float_format="%.3f",
        caption=f"Human-led: top {TOP_K} factors when performance improved (highest positive {h_col}).",
        label="tab:human_topk_improved"
    ))
    f.write("\n\n")
    f.write(human_top_decl.to_latex(
        index=False, float_format="%.3f",
        caption=f"Human-led: top {TOP_K} factors when performance declined (most negative {h_col}).",
        label="tab:human_topk_declined"
    ))
    f.write("\n\n")

    f.write("%% -------- AI: Top-k improved vs declined (same-leadership) --------\n")
    f.write(ai_top_impr.to_latex(
        index=False, float_format="%.3f",
        caption=f"AI-led: top {TOP_K} factors when performance improved (highest positive {a_col}).",
        label="tab:ai_topk_improved"
    ))
    f.write("\n\n")
    f.write(ai_top_decl.to_latex(
        index=False, float_format="%.3f",
        caption=f"AI-led: top {TOP_K} factors when performance declined (most negative {a_col}).",
        label="tab:ai_topk_declined"
    ))
    f.write("\n")
print(f"LaTeX tables written to: {OUT_TEX}")

#LATEX ENDs


# df_sessions already built in your script (list of dicts -> DataFrame)
res = correlations_per_layout_no_transitions(
    df_sessions=df_sessions,
    layout_col='layout',            # or 'map' if that’s your column
)
write_per_layout_latex(res['corr_by_layout'], out_tex="layout_session_corr_tables.tex")

write_per_layout_latex_with_meta(res['corr_by_layout'], out_tex="layout_session_corr_tables.tex")

# optional heatmap
plot_layout_heatmap(res['corr_by_layout'], top_k=10, out_png="layout_corr_heatmap.png")


corrs = correlations_by_leadership_and_suggestions(df_sessions)

for (lt, sug), tbl in corrs.items():
    print(f"\nLeader={lt}, Suggestions={sug}")
    print(tbl.head(10))
    

write_by_leadership_suggestions_latex_with_meta(
    corrs,        # dict like {('human','on'): df, ('ai','off'): df, ...}
    out_tex="lead_sugg_corr_tables.tex",
    note_excluded="excluding dangerous_weighted_distance",
    top_k=50,
    require_all_groups=True,
    pretty_names={
        ('human','on'):  "Human, Suggestions ON",
        ('human','off'): "Human, Suggestions OFF",
        ('ai','on'):     "AI, Suggestions ON",
        ('ai','off'):    "AI, Suggestions OFF",
    },
    do_pairwise=True,
    abs_thresh_pair=0.0,
    diff_thresh_pair=0.40,
    top_k_pair=50
)

results_delta = correlations_by_transition(df_sessions, use_delta=True)  # correlations to Δperformance
for trans, corr_df in results_delta.items():
    print(f"\n=== {trans} — correlations w.r.t. Δperformance ===")
    print(corr_df.head(20))
    corr_df.to_csv(f"corr_{trans.replace('->','to')}_delta_performance.csv", index=False)

save_latex_tables(results_delta, filename_prefix="corr_delta", caption_suffix="$\Delta$performance")

write_by_transition_latex_with_meta(
    results_delta,
    out_tex="transition_corr_tables_raw.tex",
    note_excluded="excluding dangerous_weighted_distance",
    top_k=50,
    do_pairwise=True,
    abs_thresh_pair=0.0,
    diff_thresh_pair=0.40,
    top_k_pair=50,
    # (optional prettier names)
    pretty_names={
        "AI->Human":   "AI → Human",
        "Human->AI":   "Human → AI",
        "AI->AI":      "AI → AI",
        "Human->Human":"Human → Human",
    }
)


explanations_tbl = per_team_explanations_same_leadership(
    df_sessions,
    pair_mode="prev",   # or "all_pairs"
    top_k=3,            # number of top contributors
    use_sign_map=True   # use the leader-specific SIGN_HELP_BY_LEADER
)

explanations_tbl.to_csv("team_explanations_samelead.csv", index=False)

if not legacy:
    for p_key in payment.keys():
        print(len(payment[p_key]))
        print(p_key, sum(payment[p_key][-4:]), payment[p_key][-4:])
        print(p_key, sum(payment[p_key][-8:-4]), payment[p_key][-8:-4])
    #quit()

    
    write_grouped_sums(payment, "payment.csv")
    
    for a_key in final_average_survey_results.keys():
        print(a_key,final_average_survey_results[a_key])


    legends = ['Human leader, no suggestions', 'Human leader, with suggestions', 'AI leader, with suggestions', 'AI leader, no suggestions']

    #print(all_final_metrics)
    print("")

    number_total_scenarios = len(final_metrics["distribution_sensed"])
    number_scenarios = 4
    series = 0
    group_by_layout = [{l+1:[] for l in range(len(layouts))},{l+1:[] for l in range(len(layouts))}, {l+1:[] for l in range(len(layouts))}, {l+1:[] for l in range(len(layouts))}]
    layout_list = []

    for n in range(number_total_scenarios):

        if n and not (n % number_scenarios):
            series += 1

        try:
            group_by_layout[0][layouts[layout_ids[series]-1][n % number_scenarios]].append(final_metrics["complexity_reduction"][n])
            
            if "human" in final_metrics["strategy"][n]:
                strat = "human leader"
            else:
                strat = "AI leader"
            
            group_by_layout[1][layouts[layout_ids[series]-1][n % number_scenarios]].append(strat)
            group_by_layout[2][layouts[layout_ids[series]-1][n % number_scenarios]].append(final_metrics["name"][n])
            group_by_layout[3][layouts[layout_ids[series]-1][n % number_scenarios]].append(n % number_scenarios)
            layout_list.append(layouts[layout_ids[series]-1][n % number_scenarios])
        except:
            pdb.set_trace() #error here because of layout
        
        for a_key in final_metrics.keys():

            #print(a_key + ": ", final_metrics[a_key][n])
            pass
                    
        #print("")


    for l in group_by_layout[0].keys():
        mx = np.argmax(group_by_layout[0][l])
        mn = np.argmin(group_by_layout[0][l])
        
        print("Layout", l, "Min:", group_by_layout[2][l][mn], group_by_layout[1][l][mn], "Max:", group_by_layout[2][l][mx], group_by_layout[1][l][mx])
        

    multiples_final_metrics = int(len(final_metrics["distribution_sensed"])/number_scenarios)

    group_final = [q for q in final_average_survey_results if q in final_questions]
    group_other = [q for q in final_average_survey_results if q not in final_questions]
     
    def plot_group(ax, keys, title, legends, m):
        n_questions = len(keys)
        n_bars = len(legends)
        indices = np.arange(n_questions)
        bar_width = 0.8 / n_bars
        # Build one series per legend entry
        values_matrix = [
            [final_average_survey_results[q][i] for q in keys]
            for i in range(m*n_bars,n_bars*(m+1))
        ]

        for i, vals in enumerate(values_matrix):
            ax.bar(
                indices + i * bar_width,
                vals,
                width=bar_width,
                label=legends[i]
            )

        # Optional key cleanup for the “Other Questions” plot
        xtick_labels = keys
        if title == 'Other Questions':
            xtick_labels = [k.split(":")[0] for k in keys]

        ax.set_xticks(indices + bar_width * (n_bars - 1) / 2)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
        ax.set_ylabel('Average Score')
        ax.set_title(title)


    '''
    #Block3--------------
    dist_list    = final_metrics["objects_carried_distribution"]
    danger_list  = final_metrics["objects_carried_distribution_danger"]
    weight_list  = final_metrics["objects_carried_distribution_weight"]
    perf_list    = final_metrics["complexity_reduction"]
    layouts      = layout_list
    leaders = [1 if "AI" in s else 0 for s in final_metrics["strategy"]]  # [0/1] per session
    session_labels = final_metrics["name"]  # optional

    # =========================
    # 1) Build per-object (long) dataframe
    # =========================
    records = []
    for sess_id, (dist, dang, wdict, lay, lt) in enumerate(zip(dist_list, danger_list, weight_list, layouts, leaders)):
        for obj, tod in dist.items():
            # danger flag: 2 = dangerous, 1 = benign
            dflag = dang.get(obj, 1)
            # weight if present (may be NaN)
            w = wdict.get(obj, np.nan)
            records.append({
                "session": sess_id,
                "session_label": session_labels[sess_id] if session_labels else f"S{sess_id}",
                "layout": lay,
                "leader_type": lt,
                "object": obj,
                "dangerous": (dflag == 2),
                "time_over_distance": float(tod),
                "weight": float(w) if pd.notna(w) else np.nan,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No per-object data found. Check final_metrics lists.")

    # Efficiency (higher is better). Guard against div-by-zero or negative values.
    tod = pd.to_numeric(df["time_over_distance"], errors="coerce")
    tod = tod.where(tod > 0, np.nan)  # set non-positive to NaN
    df["efficiency"] = 1.0 / tod

    # =========================
    # 2) Aggregate to session-level
    # =========================
    def agg_session(g: pd.DataFrame) -> pd.Series:
        # Basic counts
        n_objects = len(g)
        n_danger  = int(g["dangerous"].sum())
        n_benign  = int((~g["dangerous"]).sum())

        # Efficiency aggregates
        overall_eff = g["efficiency"].mean(skipna=True)

        # Weights
        total_weight         = g["weight"].sum(skipna=True)
        total_danger_weight  = g.loc[g["dangerous"], "weight"].sum(skipna=True)
        total_benign_weight  = g.loc[~g["dangerous"], "weight"].sum(skipna=True)

        # Time (optional, sometimes helpful)
        total_time           = g["time_over_distance"].sum(skipna=True)
        benign_time          = g.loc[~g["dangerous"], "time_over_distance"].sum(skipna=True)
        dangerous_time       = g.loc[g["dangerous"], "time_over_distance"].sum(skipna=True)

        return pd.Series({
            "n_objects": n_objects,
            "n_danger": n_danger,
            "n_benign": n_benign,
            "overall_eff": overall_eff,
            "total_weight": total_weight,
            "total_danger_weight": total_danger_weight,
            "total_benign_weight": total_benign_weight,
            "total_time": total_time,
            "dangerous_time": dangerous_time,
            "benign_time": benign_time,
        })

    agg = (df
           .groupby(["session", "session_label", "layout", "leader_type"], dropna=False)
           .apply(agg_session)
           .reset_index())

    # =========================
    # 3) Merge performance (complexity_reduction)
    # =========================
    perf_df = pd.DataFrame({
        "session": np.arange(len(perf_list), dtype=int),
        "complexity_reduction": pd.to_numeric(perf_list, errors="coerce")
    })
    summary2 = agg.merge(perf_df, on="session", how="left")

    # Optional: sanity check
    # print(summary2.head())

    # =========================
    # 4) Analyses
    # =========================

    # ---- A) Does overall efficiency translate into more objects carried? ----
    # Correlation (Spearman is robust to non-normality)
    corr_spearman = summary2[["overall_eff", "n_objects"]].corr(method="spearman")
    print("\n[Analysis A] Spearman correlation: overall_eff vs n_objects")
    print(corr_spearman)

    # (Optional) Simple regression: n_objects ~ overall_eff + leader/layout controls
    model_eff_to_count = smf.ols(
        "n_objects ~ overall_eff + C(leader_type) + C(layout)",
        data=summary2
    ).fit(cov_type="HC3")
    print("\n[Analysis A] OLS: n_objects ~ overall_eff + controls (robust SEs)")
    print(model_eff_to_count.summary())

    # ---- B) Do dangerous vs benign counts matter differently for performance? ----
    model_counts = smf.ols(
        "complexity_reduction ~ n_danger + n_benign + C(leader_type) + C(layout)",
        data=summary2
    ).fit(cov_type="HC3")
    print("\n[Analysis B] OLS: performance ~ n_danger + n_benign + controls (robust SEs)")
    print(model_counts.summary())

    # ---- C) Does total weight predict performance beyond counts? ----
    model_weight = smf.ols(
        "complexity_reduction ~ total_weight + n_objects + C(leader_type) + C(layout)",
        data=summary2
    ).fit(cov_type="HC3")
    print("\n[Analysis C] OLS: performance ~ total_weight + n_objects + controls (robust SEs)")
    print(model_weight.summary())

    # ---- D) Do dangerous vs benign weight matter differently? ----
    model_split_weight = smf.ols(
        "complexity_reduction ~ total_danger_weight + total_benign_weight + C(leader_type) + C(layout)",
        data=summary2
    ).fit(cov_type="HC3")
    print("\n[Analysis D] OLS: performance ~ danger_weight + benign_weight + controls (robust SEs)")
    print(model_split_weight.summary())

    # =========================
    # Optional: quick effect tables for each model
    # =========================
    def coef_table(model):
        est = model.params.rename("coef")
        se  = model.bse.rename("robust_se")
        ci  = model.conf_int()
        ci.columns = ["ci_low","ci_high"]
        tab = pd.concat([est, se, ci, model.pvalues.rename("pval")], axis=1)
        return tab.sort_values("pval")

    print("\n[Analysis A] Coef table:\n", coef_table(model_eff_to_count))
    print("\n[Analysis B] Coef table:\n", coef_table(model_counts))
    print("\n[Analysis C] Coef table:\n", coef_table(model_weight))
    print("\n[Analysis D] Coef table:\n", coef_table(model_split_weight))
    
    # Integrated model: performance as complexity_reduction
    model_integrated = smf.ols(
        "complexity_reduction ~ total_danger_weight + n_benign + overall_eff + C(leader_type) + C(layout)",
        data=summary2
    ).fit(cov_type="HC3")

    print("\n[Integrated Model] OLS: performance ~ danger_weight + n_benign + efficiency + controls (robust SEs)")
    print(model_integrated.summary())

    # Optional: nice coef table
    def coef_table(model):
        est = model.params.rename("coef")
        se  = model.bse.rename("robust_se")
        ci  = model.conf_int()
        ci.columns = ["ci_low","ci_high"]
        return pd.concat([est, se, ci, model.pvalues.rename("pval")], axis=1).sort_values("pval")

    print("\n[Integrated Model] Coef table:\n", coef_table(model_integrated))
    
    
    # Integrated model with interaction: efficiency × dangerous weight
    formula = """
    complexity_reduction ~ total_danger_weight * overall_eff 
                          + n_benign 
                          + C(leader_type) 
                          + C(layout)
    """

    model_integrated_inter = smf.ols(formula, data=summary2).fit(cov_type="HC3")

    print("\n[Integrated Model with Interaction] OLS: performance ~ danger_weight * efficiency + n_benign + controls")
    print(model_integrated_inter.summary())

    # Optional: tidy coef table
    def coef_table(model):
        est = model.params.rename("coef")
        se  = model.bse.rename("robust_se")
        ci  = model.conf_int()
        ci.columns = ["ci_low","ci_high"]
        return pd.concat([est, se, ci, model.pvalues.rename("pval")], axis=1).sort_values("pval")

    print("\n[Integrated Model with Interaction] Coef table:\n", coef_table(model_integrated_inter))
    
    

    
    #Block3-------------
    '''
    '''
    #Block------------------
    # --- Inputs you already have ---
    # final_metrics["objects_carried_distribution"] -> list[dict[obj -> time_over_distance]]
    # layout_list -> list[str] same length
    # leader_type -> list[int] 0=human, 1=AI (same length)
    objects_carried_distribution = final_metrics["objects_carried_distribution"]
    layouts_for_session = layout_list
    leader_type_list = [1 if "AI" in s else 0 for s in final_metrics["strategy"]]
    session_labels = final_metrics.get("name", None)

    # --- Provide/derive your team list here ---
    team_list = final_metrics.get("team_id", None)
    if team_list is None:
        # Fallback: each session as its own team (change this to your real team list)
        team_list = list(range(len(objects_carried_distribution)))

    objects_carried_distribution_danger = final_metrics["objects_carried_distribution_danger"]

    # -----------------------------
    # Config
    # -----------------------------
    INSIDE_LABELS = True
    LABEL_ROTATION = 90
    FONT_SIZE = 7
    GROUP_WIDTH = 0.85

    # Outline colors to encode leader type
    EDGE_HUMAN = "#1f2937"  # dark gray
    EDGE_AI    = "#b45309"  # dark orange

    # Danger colors (for x tick labels)
    DANGER_COLOR = "red"
    BENIGN_COLOR = "blue"

    # Label shortening
    MAX_LABEL_CHARS = 16
    ELLIPSIZE = True
    INSIDE_MIN_HEIGHT_FRAC = 0.12

    # -----------------------------
    # Helpers
    # -----------------------------
    _num_re = re.compile(r'\d+')
    def natural_key(s):
        if isinstance(s, (int, float)):
            return (0, s)
        nums = _num_re.findall(str(s))
        return (0, int(nums[-1])) if nums else (1, str(s).lower())

    def fmt_session(i):
        if session_labels and i < len(session_labels):
            s = session_labels[i]
        else:
            s = f"S{i}"
        if ELLIPSIZE and len(s) > MAX_LABEL_CHARS:
            return s[:MAX_LABEL_CHARS-1] + "…"
        return s

    # Build a big, distinct team palette
    def build_team_palette(team_ids):
        # concatenate several tab palettes for many distinct colors
        cmap_list = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c]
        uniq = list(dict.fromkeys(team_ids))  # stable unique
        colors = []
        for cmap in cmap_list:
            colors.extend([cmap(i) for i in range(cmap.N)])
        # repeat if needed
        while len(colors) < len(uniq):
            colors = colors + colors
        return {t: colors[idx] for idx, t in enumerate(uniq)}

    team_color_map_global = build_team_palette(team_list)

    # -----------------------------
    # Group sessions by layout
    # -----------------------------
    by_layout = defaultdict(list)
    for idx, layout in enumerate(layouts_for_session):
        by_layout[layout].append(idx)

    # -----------------------------
    # Plot one figure per layout
    # -----------------------------
    for layout, idxs in by_layout.items():
        # Union of objects for this layout
        all_objs = set()
        for i in idxs:
            all_objs |= set(objects_carried_distribution[i].keys())
        objs = sorted(list(all_objs), key=natural_key)
        n_obj = len(objs)
        n_sess = len(idxs)

        # Values matrix
        values = np.zeros((n_obj, n_sess), dtype=float)
        for j, obj in enumerate(objs):
            for k, i in enumerate(idxs):
                values[j, k] = objects_carried_distribution[i].get(obj, 0.0)

        # Danger labels for x ticks
        danger_labels = []
        for obj in objs:
            status = 1
            for i in idxs:
                if objects_carried_distribution_danger[i].get(obj, 1) == 2:
                    status = 2
                    break
            danger_labels.append(DANGER_COLOR if status == 2 else BENIGN_COLOR)

        # Scale / geometry
        y_max = values.max() if values.size else 1.0
        if y_max <= 0:
            y_max = 1.0
        x = np.arange(n_obj)
        bar_w = GROUP_WIDTH / max(n_sess, 1)
        offsets = (np.arange(n_sess) - (n_sess - 1) / 2) * bar_w

        fig_w = max(8, 0.6 * n_obj + 4)
        fig, ax = plt.subplots(figsize=(fig_w, 5.8))

        # Plot bars + annotate
        team_handles_seen = {}  # to build legend for teams (fills)
        for k, i in enumerate(idxs):
            lt = leader_type_list[i]
            team_id = team_list[i]

            fill_color = team_color_map_global[team_id]
            edge_color = EDGE_AI if lt == 1 else EDGE_HUMAN

            bars = ax.bar(
                x + offsets[k],
                values[:, k],
                width=bar_w,
                color=fill_color,          # FILL encodes TEAM
                edgecolor=edge_color,      # OUTLINE encodes LEADER TYPE
                linewidth=1.2
            )

            # Keep one handle per team for legend
            if team_id not in team_handles_seen:
                team_handles_seen[team_id] = Patch(facecolor=fill_color, edgecolor="#222222", label=f"Team {team_id}")

            sess_txt = fmt_session(i)
            for j, bar in enumerate(bars):
                h = bar.get_height()
                if h <= 0:
                    continue
                if INSIDE_LABELS and h >= INSIDE_MIN_HEIGHT_FRAC * y_max:
                    y_text = h / 2.0
                    va = "center"
                else:
                    y_text = h + 0.03 * y_max
                    va = "bottom"

                # Keep text black for readability (outline already encodes leader)
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_text,
                    sess_txt,
                    ha="center",
                    va=va,
                    rotation=LABEL_ROTATION,
                    fontsize=FONT_SIZE,
                    color="black",
                    clip_on=True
                )

        # Axes formatting
        ax.set_title(f"Objects Carried – Time/Distance by Session (Layout: {layout})", pad=10)
        ax.set_ylabel("Time / Distance")
        ax.set_xticks(x)
        ax.set_xticklabels(objs, rotation=45, ha="right")
        for tick_label, color in zip(ax.get_xticklabels(), danger_labels):
            tick_label.set_color(color)
        ax.set_ylim(0, y_max * 1.12)
        ax.margins(x=0.01)

        # Vertical separators between objects
        for pos in x[:-1]:
            ax.axvline(pos + 0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)

        # --- Legends ---
        # Team legend (fills)
        team_handles = list(team_handles_seen.values())
        leg_teams = ax.legend(handles=team_handles, title="Teams (fill color)", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        ax.add_artist(leg_teams)

        # Leader legend (outlines)
        leader_handles = [
            Patch(facecolor="white", edgecolor=EDGE_HUMAN, linewidth=1.6, label="Human-led"),
            Patch(facecolor="white", edgecolor=EDGE_AI,    linewidth=1.6, label="AI-led"),
        ]
        ax.legend(handles=leader_handles, title="Leader (outline color)", loc="upper left", bbox_to_anchor=(1.01, 0.35))

        fig.tight_layout()
        plt.show()

    #Block-------------------
    '''
    '''
    #Block2-------------------
    dist_list = final_metrics["objects_carried_distribution"]
    danger_list = final_metrics["objects_carried_distribution_danger"]
    comp_red    = final_metrics["complexity_reduction"]
    layouts = layout_list
    leaders = [1 if "AI" in s else 0 for s in final_metrics["strategy"]]  # [0/1] per session
    session_labels = final_metrics["name"]  # optional
    # -----------------
    # Helper functions
    # -----------------
    def natural_key(s):
        _num_re = re.compile(r'\d+')
        if isinstance(s, (int, float)):
            return (0, s)
        nums = _num_re.findall(str(s))
        return (0, int(nums[-1])) if nums else (1, str(s).lower())

    def sess_name(i):
        if session_labels and i < len(session_labels):
            return session_labels[i]
        return f"S{i}"

    # ----------------------
    # Build a long dataframe
    # ----------------------
    records = []
    for sess_id, (dist, dang, lay, lt) in enumerate(zip(dist_list, danger_list, layouts, leaders)):
        for obj, tod in dist.items():
            dflag = dang.get(obj, 1)  # default benign(1)
            records.append({
                "session": sess_id,
                "session_label": sess_name(sess_id),
                "layout": lay,
                "leader_type": lt,
                "object": obj,
                "dangerous": (dflag == 2),
                "time_over_distance": float(tod),
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No data found in objects_carried_distribution.* inputs.")

    # Efficiency: higher is better (avoid div by zero)
    df["efficiency"] = df["time_over_distance"].replace(0, np.nan)
    df["efficiency"] = 1.0 / df["efficiency"]
    df["efficiency"] = df["efficiency"].fillna(0.0)

    # Attach per-session complexity reduction
    comp_df = pd.DataFrame({
        "session": np.arange(len(comp_red), dtype=int),
        "complexity_reduction": pd.to_numeric(comp_red, errors="coerce")
    })

    # --------------------------
    # Session-level summary stats
    # --------------------------
    def agg_session(g):
        is_dan = g["dangerous"].values
        eff    = g["efficiency"].values
        tod    = g["time_over_distance"].values

        dan_eff = np.nanmean(eff[is_dan]) if np.any(is_dan) else np.nan
        ben_eff = np.nanmean(eff[~is_dan]) if np.any(~is_dan) else np.nan
        danger_focus = float(np.mean(is_dan)) if len(is_dan) else np.nan

        total_time  = tod.sum()
        benign_time = tod[~is_dan].sum() if np.any(~is_dan) else 0.0
        benign_share = benign_time / total_time if total_time > 0 else np.nan

        return pd.Series({
            "dangerous_eff": dan_eff,
            "benign_eff": ben_eff,
            "danger_focus": danger_focus,
            "benign_time_share": benign_share,
            "n_objects": len(g),
            "n_danger": int(is_dan.sum()),
            "n_benign": int((~is_dan).sum())
        })

    summary = (df
        .groupby(["session", "session_label", "layout", "leader_type"], dropna=False)
        .apply(agg_session)
        .reset_index())

    # Merge complexity reduction
    summary = summary.merge(comp_df, on="session", how="left")

    # (Optional) per-dangerous-object relative comparison (z-score) — unchanged
    danger_df = df[df["dangerous"]].copy()
    if not danger_df.empty:
        obj_stats = danger_df.groupby("object")["efficiency"].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sigma"})
        danger_df = danger_df.merge(obj_stats, on="object", how="left")
        danger_df["z_eff"] = (danger_df["efficiency"] - danger_df["mu"]) / danger_df["sigma"].replace(0, np.nan)
        danger_df["z_eff"] = danger_df["z_eff"].fillna(0.0)
        z_by_session = danger_df.groupby("session")["z_eff"].mean().rename("mean_z_danger_eff")
        summary = summary.merge(z_by_session, left_on="session", right_index=True, how="left")

    # -------------
    # Plotting setup
    # -------------
    HUMAN_COLOR = "#4C78A8"
    AI_COLOR    = "#F58518"
    EDGE        = "black"

    def color_for_leader(lt):  # lt in {0,1}
        return AI_COLOR if lt == 1 else HUMAN_COLOR

    # -------------
    # Make figures per layout
    # -------------
    for layout, sub in summary.groupby("layout"):
        # ---- NEW: Rank within this layout by complexity_reduction (best=1) ----
        # If lower is better in your metric, set ascending=True instead.
        sub = sub.copy()
        sub["rank_in_layout"] = sub["complexity_reduction"].rank(ascending=False, method="dense").astype(int)

        # -------- Figure 1: Focus vs Efficiency (dangerous) with ranks inside points --------
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        x = sub["danger_focus"].values
        y = sub["dangerous_eff"].values
        sizes  = np.clip(sub["n_objects"].values, 1, None) * 25  # bubble size
        colors = [color_for_leader(lt) for lt in sub["leader_type"].values]

        sc = ax1.scatter(x, y, s=sizes, c=colors, alpha=0.9, edgecolors=EDGE, linewidths=0.7, zorder=2)
        ax1.set_title(f"Focus vs Efficiency (dangerous) — Layout: {layout}")
        ax1.set_xlabel("Danger Focus Ratio (#danger / #carried)")
        ax1.set_ylabel("Dangerous Efficiency (mean of 1 / time-distance)")
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Put RANK number inside each circle; keep a tiny session label next to it (optional)
        for _, row in sub.iterrows():
            # Inside marker text (rank)
            ax1.text(
                row["danger_focus"], row["dangerous_eff"],
                str(row["rank_in_layout"]),
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if row["leader_type"] == 1 else "black",  # contrast vs fill
                zorder=3
            )
            # Optional tiny label near the point (comment this block out if you want only ranks)
            ax1.annotate(
                row["session_label"],
                (row["danger_focus"], row["dangerous_eff"]),
                textcoords="offset points", xytext=(6, 5),
                fontsize=7, color="black", zorder=3
            )

        # Leader legend
        leader_handles = [
            plt.Line2D([0],[0], marker='o', linestyle='none', markersize=8, markerfacecolor=HUMAN_COLOR,
                       markeredgecolor=EDGE, label="Human-led"),
            plt.Line2D([0],[0], marker='o', linestyle='none', markersize=8, markerfacecolor=AI_COLOR,
                       markeredgecolor=EDGE, label="AI-led"),
        ]
        ax1.legend(handles=leader_handles, title="Leader Type", loc="best")

        fig1.tight_layout()
        plt.show()
        
        
    # Make a working copy and keep only needed columns
    cols_needed = [
        "complexity_reduction",   # outcome (performance)
        "danger_focus",
        "dangerous_eff",
        "benign_time_share",
        "eff_gap",                # if you created: dangerous_eff - benign_eff
        "mean_z_danger_eff",      # optional; may be NaN if not computed
        "n_objects",              # control (workload)
        "leader_type",
        "layout",
    ]
    df_reg = summary.copy()

    # If eff_gap isn't present, create it (harmless if already there)
    if "eff_gap" not in df_reg.columns and {"dangerous_eff","benign_eff"}.issubset(df_reg.columns):
        df_reg["eff_gap"] = df_reg["dangerous_eff"] - df_reg["benign_eff"]

    # Keep only columns that exist in your summary
    cols_exist = [c for c in cols_needed if c in df_reg.columns]
    df_reg = df_reg[cols_exist].copy()

    # Optional: scale continuous predictors for comparability (does NOT change fit, just improves interpretability)
    to_scale = [c for c in ["danger_focus","dangerous_eff","benign_time_share","eff_gap",
                            "mean_z_danger_eff","n_objects"] if c in df_reg.columns]
    for c in to_scale:
        mu, sd = df_reg[c].mean(), df_reg[c].std(ddof=0)
        if np.isfinite(sd) and sd > 0:
            df_reg[c] = (df_reg[c] - mu) / sd

    # Drop rows with missing data in any used column
    df_reg = df_reg.dropna()

    # Build formula: outcome ~ predictors + controls
    # - C(leader_type) adds a categorical dummy (AI vs Human)
    # - C(layout) controls for layout differences (fixed effects)
    predictors = [c for c in ["danger_focus","dangerous_eff","benign_time_share","eff_gap",
                              "mean_z_danger_eff","n_objects"] if c in df_reg.columns]
    formula = "complexity_reduction ~ " + " + ".join(predictors) + " + C(leader_type) + C(layout)"

    # Fit with robust (HC3) SEs
    ols_model = smf.ols(formula, data=df_reg).fit(cov_type="HC3")
    print(ols_model.summary())

    # ---------- Optional: VIF to check multicollinearity (on numeric design matrix) ----------
    # Build design matrices WITHOUT the outcome
    y, X = dmatrices(formula, data=df_reg, return_type='dataframe')
    # dmatrices returns y and X; y includes the intercept already, so X is fine for VIF
    vif_df = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    print("\nVariance Inflation Factors (VIF):")
    print(vif_df)

    # ---------- Optional: effect sizes table (coef, robust SE, 95% CI, p) ----------
    est = ols_model.params
    se  = ols_model.bse
    ci  = ols_model.conf_int()
    ci.columns = ["ci_low","ci_high"]
    tab = pd.concat([est.rename("coef"), se.rename("robust_se"), ci], axis=1)
    tab["pval"] = ols_model.pvalues
    print("\nCoef table (robust):")
    print(tab.sort_values("pval"))
    #Block2------------------
    '''

    '''
    #Block4-------------------
    sns.set(style="whitegrid")

    # ---- Required final_metrics keys (lists of length = #sessions) ----
    # final_metrics["objects_carried_distribution"]            : list[dict[obj -> time_over_distance]]
    # final_metrics["objects_carried_distribution_danger"]     : list[dict[obj -> 1 (benign) / 2 (dangerous)]]
    # final_metrics["objects_carried_distribution_weight"]     : list[dict[obj -> weight]]
    # final_metrics["complexity_reduction"]                    : list[float]  (performance)
    # layout_list                                              : list[str] or list[int]
    # leader type: use ONE of the two options below:
    #   A) final_metrics["leader_type"]                        : list[int] 0/1  (if you already have it)
    #   B) final_metrics["strategy"]                           : list[str] (we infer 1 if "AI" in string else 0)
    #
    # Optional (nice to have):
    # final_metrics["name"]                                    : list[str] (session labels)
    # final_metrics["team_id"]                                 : list[str|int] (if you want to track teams)

    dist_list    = final_metrics["objects_carried_distribution"]
    danger_list  = final_metrics["objects_carried_distribution_danger"]
    weight_list  = final_metrics["objects_carried_distribution_weight"]
    perf_list    = final_metrics["complexity_reduction"]
    layouts      = layout_list

    # Leader type: prefer explicit list if present; otherwise infer from "strategy"
    if "leader_type" in final_metrics:
        leaders = final_metrics["leader_type"]
    else:
        strategies = final_metrics.get("strategy", [""] * len(dist_list))
        leaders = [1 if isinstance(s, str) and ("AI" in s or "ai" in s.lower()) else 0 for s in strategies]

    session_labels = final_metrics.get("name", None)
    team_list      = final_metrics.get("team_id", list(range(len(dist_list))))  # optional

    assert len(dist_list) == len(danger_list) == len(weight_list) == len(perf_list) == len(layouts) == len(leaders), \
        "Input lists must all have the same length."

    # =========================================================
    # 1) Build per-object (long) dataframe: df
    # =========================================================
    records = []
    for sess_id, (dist, dang, wdict, lay, lt) in enumerate(zip(dist_list, danger_list, weight_list, layouts, leaders)):
        for obj, tod in dist.items():
            dflag = dang.get(obj, 1)   # 2 = dangerous, 1 = benign
            w     = wdict.get(obj, np.nan)
            records.append({
                "session": sess_id,
                "session_label": session_labels[sess_id] if session_labels else f"S{sess_id}",
                "team_id": team_list[sess_id] if team_list is not None else None,
                "layout": lay,
                "leader_type": lt,                    # 0 human, 1 AI
                "object": obj,
                "dangerous": (dflag == 2),
                "time_over_distance": float(tod),
                "weight": float(w) if pd.notna(w) else np.nan,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No per-object rows were produced. Check your inputs.")

    # Efficiency (higher is better); avoid non-positive times
    df["time_over_distance"] = pd.to_numeric(df["time_over_distance"], errors="coerce")
    df.loc[df["time_over_distance"] <= 0, "time_over_distance"] = np.nan
    df["efficiency"] = 1.0 / df["time_over_distance"]

    # =========================================================
    # 2) Aggregate to per-session summary: summary2
    # =========================================================
    def agg_session(g: pd.DataFrame) -> pd.Series:
        is_dan = g["dangerous"].values
        eff    = g["efficiency"].values
        tod    = g["time_over_distance"].values
        w      = g["weight"].values

        n_objects = len(g)
        n_danger  = int(is_dan.sum())
        n_benign  = int((~is_dan).sum())

        overall_eff = np.nanmean(eff)
        # Focus: fraction dangerous among carried
        danger_focus = float(np.mean(is_dan)) if n_objects > 0 else np.nan

        total_weight         = np.nansum(w)
        total_danger_weight  = np.nansum(w[is_dan]) if n_danger > 0 else 0.0
        total_benign_weight  = np.nansum(w[~is_dan]) if n_benign > 0 else 0.0

        # Efficiency dispersion (consistency)
        var_eff    = np.nanvar(eff, ddof=0)
        med_eff    = np.nanmedian(eff)
        pct_high   = np.mean(eff > np.nanpercentile(df["efficiency"], 60)) if np.isfinite(np.nanpercentile(df["efficiency"], 60)) else np.nan

        # Time splits (optional)
        total_time     = np.nansum(tod)
        dangerous_time = np.nansum(tod[is_dan]) if n_danger > 0 else 0.0
        benign_time    = np.nansum(tod[~is_dan]) if n_benign > 0 else 0.0
        benign_share   = benign_time / total_time if total_time and total_time > 0 else np.nan

        return pd.Series({
            "n_objects": n_objects,
            "n_danger": n_danger,
            "n_benign": n_benign,
            "danger_focus": danger_focus,
            "overall_eff": overall_eff,
            "median_eff": med_eff,
            "var_eff": var_eff,
            "pct_high_eff": pct_high,
            "total_weight": total_weight,
            "total_danger_weight": total_danger_weight,
            "total_benign_weight": total_benign_weight,
            "total_time": total_time,
            "dangerous_time": dangerous_time,
            "benign_time": benign_time,
            "benign_time_share": benign_share,
        })

    summary2 = (
        df.groupby(["session", "session_label", "team_id", "layout", "leader_type"], dropna=False)
          .apply(agg_session)
          .reset_index()
    )

    # Attach performance
    perf_df = pd.DataFrame({
        "session": np.arange(len(perf_list), dtype=int),
        "complexity_reduction": pd.to_numeric(perf_list, errors="coerce")
    })
    summary2 = summary2.merge(perf_df, on="session", how="left")

    # (Optional) Normalize performance within layout to remove map difficulty
    summary2["perf_z_layout"] = summary2.groupby("layout")["complexity_reduction"] \
        .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0)>0 else 1.0))

    # Palette for leader type
    palette_leader = {0: "#4C78A8", 1: "#F58518"}  # Human, AI

    # =========================================================
    # 3) Visual exploration: split by leader type
    # =========================================================
    factors = [
        "danger_focus","overall_eff",
        "total_danger_weight","total_benign_weight",
        "n_objects","n_danger","n_benign",
        "median_eff","var_eff","pct_high_eff",
    ]
    factors = [f for f in factors if f in summary2.columns]

    for f in factors:
        g = sns.lmplot(
            data=summary2.dropna(subset=[f, "complexity_reduction", "leader_type"]),
            x=f, y="complexity_reduction",
            hue="leader_type", palette=palette_leader,
            scatter_kws={"alpha": 0.8, "s": 60}, ci=None, height=6, aspect=1.2
        )
        g.set_axis_labels(f, "Performance (complexity_reduction)")
        plt.title(f"Performance vs {f} (split by Leader Type)")
        plt.tight_layout()
        plt.show()

    # Also with layout-normalized performance
    for f in factors:
        g = sns.lmplot(
            data=summary2.dropna(subset=[f, "perf_z_layout", "leader_type"]),
            x=f, y="perf_z_layout",
            hue="leader_type", palette=palette_leader,
            scatter_kws={"alpha": 0.8, "s": 60}, ci=None, height=6, aspect=1.2
        )
        g.set_axis_labels(f, "Performance (z within layout)")
        plt.title(f"Layout-normalized Performance vs {f} (split by Leader Type)")
        plt.tight_layout()
        plt.show()

    # =========================================================
    # 4) OLS with interactions: factor * leader_type (+ layout FE)
    # =========================================================
    def fit_interaction_ols(df, factor, outcome="complexity_reduction"):
        cols = [outcome, factor, "leader_type", "layout"]
        dfx = df[cols].dropna()
        formula = f"{outcome} ~ {factor} * leader_type + C(layout)"
        model = smf.ols(formula, data=dfx).fit(cov_type="HC3")
        print(f"\n[Interaction OLS] {outcome} ~ {factor} * leader_type + C(layout)")
        print(model.summary())
        return model

    ols_models_raw   = {f: fit_interaction_ols(summary2, f, "complexity_reduction") for f in factors}
    ols_models_norm  = {f: fit_interaction_ols(summary2, f, "perf_z_layout")        for f in factors}

    # =========================================================
    # 5) Predictive view: Random Forest feature importance per leader type
    # =========================================================
    # Choose features (drop leader_type; we will split by it) and add layout as one-hot
    feature_candidates = [c for c in [
        "danger_focus","overall_eff","median_eff","var_eff","pct_high_eff",
        "total_danger_weight","total_benign_weight",
        "n_objects","n_danger","n_benign",
    ] if c in summary2.columns]

    layout_dummies = pd.get_dummies(summary2["layout"], prefix="layout", drop_first=True)
    ml_df = pd.concat([summary2[["complexity_reduction","leader_type"] + feature_candidates], layout_dummies], axis=1)

    def train_rf_for_group(df, group_label):
        sub = df[df["leader_type"] == group_label].dropna()
        y = sub["complexity_reduction"].values
        X = sub.drop(columns=["complexity_reduction","leader_type"])
        feature_names = X.columns.tolist()

        # Pipeline (scaler kept for permutation stability; trees don't require scaling)
        model = make_pipeline(StandardScaler(with_mean=False), RandomForestRegressor(
            n_estimators=600, random_state=42, max_depth=None, min_samples_leaf=2, n_jobs=-1
        ))
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        # Permutation importance
        result = permutation_importance(model, X, y, n_repeats=200, random_state=42, n_jobs=-1)
        imp = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=False)
        return model, r2, imp

    rf_human, r2_human, imp_human = train_rf_for_group(ml_df, group_label=0)
    rf_ai,    r2_ai,    imp_ai    = train_rf_for_group(ml_df, group_label=1)

    print(f"\n[Random Forest] In-sample R² — Human-led: {r2_human:.3f} | AI-led: {r2_ai:.3f}")

    def plot_importances(imp_series, title, top_k=10):
        s = imp_series.head(top_k)[::-1]
        plt.figure(figsize=(7, 5))
        plt.barh(s.index, s.values)
        plt.title(title)
        plt.xlabel("Permutation Importance (mean ΔR²)")
        plt.tight_layout()
        plt.show()

    plot_importances(imp_human, "Feature importance — Human-led (Random Forest)")
    plot_importances(imp_ai,    "Feature importance — AI-led (Random Forest)")

    print("\nTop features — Human-led:\n", imp_human.head(15))
    print("\nTop features — AI-led:\n", imp_ai.head(15))
    #Block4-----------------
    '''

    '''
    #Block5-----------------
    # Nice defaults
    plt.rcParams["figure.figsize"] = (8, 5)

    # =========================================================
    # 1) Build summary2 (if missing) from final_metrics
    # =========================================================
    def build_summary2_from_metrics(final_metrics, layout_list):
        dist_list    = final_metrics["objects_carried_distribution"]
        danger_list  = final_metrics["objects_carried_distribution_danger"]
        weight_list  = final_metrics["objects_carried_distribution_weight"]
        perf_list    = final_metrics["complexity_reduction"]
        layouts      = layout_list

        # Leader type: prefer explicit list if present; otherwise infer from "strategy"
        if "leader_type" in final_metrics:
            leaders = final_metrics["leader_type"]
        else:
            strategies = final_metrics.get("strategy", [""] * len(dist_list))
            leaders = [1 if isinstance(s, str) and ("ai" in s.lower()) else 0 for s in strategies]

        session_labels = final_metrics.get("name", None)
        team_list      = final_metrics.get("team_id", list(range(len(dist_list))))

        assert len(dist_list) == len(danger_list) == len(weight_list) == len(perf_list) == len(layouts) == len(leaders)

        # Per-object long DF
        rows = []
        for sess_id, (dist, dang, wdict, lay, lt) in enumerate(zip(dist_list, danger_list, weight_list, layouts, leaders)):
            for obj, tod in dist.items():
                dflag = dang.get(obj, 1)   # 2 = dangerous, 1 = benign
                w     = wdict.get(obj, np.nan)
                rows.append({
                    "session": sess_id,
                    "session_label": session_labels[sess_id] if session_labels else f"S{sess_id}",
                    "team_id": team_list[sess_id] if team_list is not None else None,
                    "layout": lay,
                    "leader_type": lt,  # 0 human, 1 AI
                    "object": obj,
                    "dangerous": (dflag == 2),
                    "time_over_distance": float(tod),
                    "weight": float(w) if pd.notna(w) else np.nan,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("No per-object rows were produced. Check your inputs.")

        # Efficiency
        df["time_over_distance"] = pd.to_numeric(df["time_over_distance"], errors="coerce")
        df.loc[df["time_over_distance"] <= 0, "time_over_distance"] = np.nan
        df["efficiency"] = 1.0 / df["time_over_distance"]

        # Aggregate to per-session
        def agg_session(g: pd.DataFrame) -> pd.Series:
            is_dan = g["dangerous"].values
            eff    = g["efficiency"].values
            tod    = g["time_over_distance"].values
            w      = g["weight"].values

            n_objects = len(g)
            n_danger  = int(is_dan.sum())
            n_benign  = int((~g["dangerous"]).sum())

            overall_eff   = np.nanmean(eff)
            median_eff    = np.nanmedian(eff)
            var_eff       = np.nanvar(eff, ddof=0)
            # "High efficiency" threshold: global 60th percentile across all object carries
            global_thr    = np.nanpercentile(df["efficiency"], 60) if np.isfinite(np.nanpercentile(df["efficiency"], 60)) else np.nan
            pct_high_eff  = np.mean(eff > global_thr) if np.isfinite(global_thr) else np.nan

            danger_focus  = float(np.mean(is_dan)) if n_objects > 0 else np.nan

            total_weight         = np.nansum(w)
            total_danger_weight  = np.nansum(w[is_dan]) if n_danger > 0 else 0.0
            total_benign_weight  = np.nansum(w[~is_dan]) if n_benign > 0 else 0.0

            return pd.Series({
                "n_objects": n_objects, "n_danger": n_danger, "n_benign": n_benign,
                "danger_focus": danger_focus,
                "overall_eff": overall_eff, "median_eff": median_eff, "var_eff": var_eff, "pct_high_eff": pct_high_eff,
                "total_weight": total_weight,
                "total_danger_weight": total_danger_weight, "total_benign_weight": total_benign_weight,
            })

        summary2 = (
            df.groupby(["session", "session_label", "team_id", "layout", "leader_type"], dropna=False)
              .apply(agg_session)
              .reset_index()
        )

        # Attach performance
        perf_df = pd.DataFrame({
            "session": np.arange(len(perf_list), dtype=int),
            "complexity_reduction": pd.to_numeric(perf_list, errors="coerce")
        })
        summary2 = summary2.merge(perf_df, on="session", how="left")

        # Normalize performance within layout (optional, handy for some views)
        summary2["perf_z_layout"] = summary2.groupby("layout")["complexity_reduction"] \
            .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0)>0 else 1.0))

        return df, summary2

    # If summary2 not defined in your environment, build it now
    if "summary2" not in globals():
        df, summary2 = build_summary2_from_metrics(final_metrics, layout_list)

    # =========================================================
    # 2) Build ML dataframe (features + one-hot layouts)
    # =========================================================
    feature_candidates = [c for c in [
        "danger_focus","overall_eff","median_eff","var_eff","pct_high_eff",
        "total_danger_weight","total_benign_weight",
        "n_objects","n_danger","n_benign",
    ] if c in summary2.columns]

    layout_dummies = pd.get_dummies(summary2["layout"], prefix="layout", drop_first=True)
    ml_df = pd.concat([summary2[["complexity_reduction","leader_type","team_id","session"] + feature_candidates], layout_dummies], axis=1)

    # =========================================================
    # 3) Train RF(s) and compute SHAP values
    #    Choose: (A) separate models per leader type   OR   (B) a single pooled model
    # =========================================================
    SEPARATE_BY_LEADER = True  # flip to False to use one pooled model

    def train_rf(X, y):
        # Trees don't require scaling; we keep a scaler to stabilize SHAP on sparse dummies
        model = make_pipeline(
            StandardScaler(with_mean=False),
            RandomForestRegressor(n_estimators=600, random_state=42, max_depth=None, min_samples_leaf=2, n_jobs=-1)
        )
        model.fit(X, y)
        return model

    def compute_shap(model, X):
        # Extract the last step estimator (RandomForestRegressor) for TreeExplainer
        rf = model.named_steps["randomforestregressor"] if "randomforestregressor" in model.named_steps else model
        explainer = shap.TreeExplainer(rf)
        # Get the array the RF sees (after scaler)
        Xt = model.named_steps["standardscaler"].transform(X) if "standardscaler" in model.named_steps else X.values
        # SHAP values aligned to features in X
        shap_vals = explainer.shap_values(Xt)
        base_val  = explainer.expected_value
        return shap_vals, base_val

    results = {}

    if SEPARATE_BY_LEADER:
        for group_label, group_name in [(0, "Human-led"), (1, "AI-led")]:
            sub = ml_df[ml_df["leader_type"] == group_label].dropna()
            y = sub["complexity_reduction"].values
            X = sub.drop(columns=["complexity_reduction","leader_type","team_id","session"])
            feature_names = X.columns.tolist()

            model = train_rf(X, y)
            shap_vals, base_val = compute_shap(model, X)

            results[group_name] = {
                "model": model, "X": X, "y": y, "feature_names": feature_names,
                "shap_values": shap_vals, "base_value": base_val,
                "team_ids": sub["team_id"].values, "sessions": sub["session"].values
            }
    else:
        sub = ml_df.dropna()
        y = sub["complexity_reduction"].values
        X = sub.drop(columns=["complexity_reduction","leader_type","team_id","session"])
        feature_names = X.columns.tolist()

        model = train_rf(X, y)
        shap_vals, base_val = compute_shap(model, X)

        results["All"] = {
            "model": model, "X": X, "y": y, "feature_names": feature_names,
            "shap_values": shap_vals, "base_value": base_val,
            "team_ids": sub["team_id"].values, "sessions": sub["session"].values
        }

    # =========================================================
    # 4) GLOBAL SHAP bar summary per group
    # =========================================================
    def shap_bar_summary(group_key, top_k=12):
        r = results[group_key]
        X, shap_vals, feat = r["X"], r["shap_values"], r["feature_names"]

        # Mean absolute SHAP by feature
        mean_abs = np.abs(shap_vals).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:top_k]

        plt.figure(figsize=(9, 6))
        plt.barh(np.array(feat)[order][::-1], mean_abs[order][::-1])
        plt.xlabel("Mean |SHAP value| (impact on prediction)")
        plt.title(f"Global feature impact — {group_key}")
        plt.tight_layout()
        plt.show()

    # Example: draw both (or the single pooled one)
    for g in results.keys():
        shap_bar_summary(g, top_k=12)

    # =========================================================
    # 5) PER-TEAM explanations
    #    A) Aggregate per-team: mean SHAP (signed) and mean |SHAP|
    #    B) Plot top drivers for any chosen team
    # =========================================================
    def per_team_shap(group_key):
        r = results[group_key]
        shap_vals = r["shap_values"]
        feat = np.array(r["feature_names"])
        team_ids = r["team_ids"]

        df_shap = pd.DataFrame(shap_vals, columns=feat)
        df_shap["team_id"] = team_ids

        # Mean signed and absolute contribution per team
        team_mean   = df_shap.groupby("team_id").mean(numeric_only=True)
        team_mean_abs = df_shap.drop(columns=["team_id"]).groupby(df_shap["team_id"]).apply(lambda d: d.abs().mean())

        return team_mean, team_mean_abs  # DataFrames indexed by team_id

    def plot_team_top_drivers(group_key, team_id, top_k=10, use_abs=True):
        team_mean, team_mean_abs = per_team_shap(group_key)
        if team_id not in team_mean.index:
            raise ValueError(f"team_id {team_id} not found in group {group_key}.")

        vals = team_mean_abs.loc[team_id] if use_abs else team_mean.loc[team_id]
        vals = vals.sort_values(ascending=False).head(top_k)[::-1]  # for nice barh order

        plt.figure(figsize=(9, 6))
        plt.barh(vals.index, vals.values)
        plt.xlabel("Mean |SHAP|" if use_abs else "Mean SHAP (signed)")
        plt.title(f"Top drivers for Team {team_id} — {group_key}")
        plt.tight_layout()
        plt.show()

    
    # Example usage:
    # plot_team_top_drivers("Human-led", team_id=<PUT_A_TEAM_ID_HERE>, top_k=10, use_abs=True)
    # plot_team_top_drivers("AI-led",    team_id=<PUT_A_TEAM_ID_HERE>, top_k=10, use_abs=False)

    # =========================================================
    # 6) Optional: SESSION-LEVEL waterfall for a specific session row
    #    (Pick a row index within the chosen group; shows push-up/down factors)
    # =========================================================
    def session_waterfall(group_key, row_index):
        r = results[group_key]
        shap_vals = r["shap_values"]
        base_val  = r["base_value"]
        feat      = r["feature_names"]
        X         = r["X"]

        if not (0 <= row_index < len(X)):
            raise IndexError("row_index out of range for selected group.")

        # SHAP waterfall (requires SHAP’s newer plotting API)
        expl = shap.Explanation(values=shap_vals[row_index],
                                base_values=base_val,
                                data=X.iloc[row_index].values,
                                feature_names=feat)
        shap.plots.waterfall(expl, max_display=14)

    # Example:
    # session_waterfall("Human-led", row_index=0)
    # session_waterfall("AI-led", row_index=0)
    
    pdb.set_trace()
    #Block5----------------
    '''
    mixed_dict = {
        'difficulty': final_metrics["complexity"],
        'performance': final_metrics["complexity_reduction"], #final_metrics["dangerous_objects"], #final_metrics["complexity_reduction"],
        'leader_type': [1 if "AI" in s else 0 for s in final_metrics["strategy"]],
        'suggestions': [0 if "without" in s else 1 for s in final_metrics["strategy"]],
        'session': [r % 4 for r in range(len(final_metrics["strategy"]))],
        'team_id': [r//4 for r in range(len(final_metrics["strategy"]))],
        'distance': final_metrics["total_distance_traveled"],
        'layout': layout_list,
        "name": final_metrics["name"],
        "accidental_drop": final_metrics["accidental_drop"],
        "weight_sum": final_metrics["weight_sum"],
        "idle_time": final_metrics["idle_time"],
        "object_carry_speed": final_metrics["object_carry_speed"],
        "order_completed": final_metrics["order_completed"],
        "dangerous_objects": final_metrics["dangerous_objects"],
        "objects_carried": final_metrics["objects_carried"],
        "object_benign_time": final_metrics["object_benign_time"],
        "object_dangerous_time": final_metrics["object_dangerous_time"],
        "help_carry_dialogue": final_metrics["help_carry_dialogue"],
        "total_danger_weight": [sum(w[obj] for obj in w if danger.get(obj,1)==2) for w, danger in zip(final_metrics["objects_carried_distribution_weight"], final_metrics["objects_carried_distribution_danger"])]

    }


    mixed_df = pd.DataFrame(mixed_dict)

    #pdb.set_trace()
    DF = mixed_df
    transitions_all, model_fd = first_diff_table(DF)
    cols = ["team_id","session","d_perf","d_complexity","perf_change","complexity_change",
         "leader_change","layout_change","pred_task","resid_after_complexity","interpretation"]
    print(transitions_all[cols].to_string(index=False))
    
    '''
    m = smf.ols(
        "performance ~ difficulty + C(session) + C(leader_type) + C(layout) + C(suggestions) + C(team_id)",
        data=DF
    ).fit(cov_type="cluster", cov_kwds={"groups": DF["team_id"]})

    print(m.summary())

    base = smf.ols("performance ~ bs(difficulty, df=4) + C(layout)", data=DF).fit()

    # Step 2: residual = interaction-adjusted performance
    DF["perf_resid"] = base.resid

    # Step 3: check trends in residuals by session/leader_type/team
    smf.ols("perf_resid ~ C(session) + C(leader_type) + C(team_id)", data=DF).fit().summary()

    m_int = smf.ols(
        "performance ~ difficulty * C(leader_type) + C(layout) + C(session) + C(team_id)",
        data=DF
    ).fit(cov_type="cluster", cov_kwds={"groups": DF["team_id"]})


    DF = DF.sort_values(["team_id","session"])
    DF["d_perf"] = DF.groupby("team_id")["performance"].diff()
    DF["d_complexity"] = DF.groupby("team_id")["difficulty"].diff()

    dd = smf.ols("d_perf ~ d_complexity + C(leader_type) + C(layout)", data=DF.dropna()).fit(
        cov_type="cluster", cov_kwds={"groups": DF.dropna()["team_id"]}
    )
    print(dd.summary())

    pdb.set_trace()
    '''
    
    '''
    # 1) Sort once
    df = DF.sort_values(["team_id", "session"]).copy()

    # 2) For each team, create "next" columns by shifting upward
    def add_next(g):
        g = g.copy()
        g["next_session"]      = g["session"].shift(-1)
        g["next_performance"]  = g["performance"].shift(-1)
        g["next_leader_type"]  = g["leader_type"].shift(-1)
        g["next_name"]         = g["name"].shift(-1) if "name" in g else np.nan
        g["next_layout"]       = g["layout"].shift(-1) if "layout" in g else np.nan
        g["next_suggestions"]  = g["suggestions"].shift(-1) if "suggestions" in g else np.nan
        return g

    df_pairs = df.groupby("team_id", group_keys=False).apply(add_next)

    # 3) Keep only valid consecutive pairs (i.e., rows that have a 'next' row)
    pairs = df_pairs.dropna(subset=["next_session", "next_performance", "next_leader_type"]).copy()

    # 4) Performance change and transition label
    pairs["perf_diff"] = pairs["next_performance"] - pairs["performance"]
    pairs["transition"] = pairs["leader_type"].astype(int).astype(str) + "→" + pairs["next_leader_type"].astype(int).astype(str)

    # 5) Build a clean “transitions” table with explicit prev/curr fields
    cols_available = set(pairs.columns)

    transitions = pd.DataFrame({
        "team_id":            pairs["team_id"].values,
        "prev_session":       pairs["session"].values,
        "curr_session":       pairs["next_session"].values,
        "prev_leader_type":   pairs["leader_type"].astype(int).values,
        "curr_leader_type":   pairs["next_leader_type"].astype(int).values,
        "prev_performance":   pairs["performance"].values,
        "curr_performance":   pairs["next_performance"].values,
        "perf_diff":          pairs["perf_diff"].values,
        "transition":         pairs["transition"].values,
    })

    # Optional metadata (only if present)
    if "name" in cols_available:
        transitions["prev_name"] = pairs["name"].astype(str).values
        transitions["curr_name"] = pairs["next_name"].astype(str).values
    if "layout" in cols_available:
        transitions["prev_layout"] = pairs["layout"].values
        transitions["curr_layout"] = pairs["next_layout"].values
    if "suggestions" in cols_available:
        # Choose whether to stratify by suggestions in the NEXT session (most common)
        transitions["sugg_next"] = pairs["next_suggestions"].astype("Int64").values
        transitions["sugg_prev"] = pairs["suggestions"].astype("Int64").values  # also available if you want it

    # 6) Now you can safely slice: no fake “prev without curr” rows remain
    # Biggest increases/decreases by suggestions (based on NEXT session suggestion status)
    mask_has_sugg = "sugg_next" in transitions.columns

    def pick_cols(df_):
        base = ["team_id","prev_leader_type","curr_leader_type",
                "prev_session","curr_session",
                "prev_performance","curr_performance","perf_diff","transition"]
        if "prev_name" in df_.columns and "curr_name" in df_.columns:
            base = base[:5] + ["prev_name","curr_name"] + base[5:]
        if "prev_layout" in df_.columns and "curr_layout" in df_.columns:
            base += ["prev_layout","curr_layout"]
        if "sugg_next" in df_.columns:
            base += ["sugg_next"]
        return base

    # Increases (perf_diff > 0)
    if mask_has_sugg:
        inc_with  = transitions[(transitions["perf_diff"] > 0) & (transitions["sugg_next"] == 1)] \
                        .sort_values("perf_diff", ascending=False)[pick_cols(transitions)]
        inc_without = transitions[(transitions["perf_diff"] > 0) & (transitions["sugg_next"] == 0)] \
                        .sort_values("perf_diff", ascending=False)[pick_cols(transitions)]
        dec_with  = transitions[(transitions["perf_diff"] < 0) & (transitions["sugg_next"] == 1)] \
                        .sort_values("perf_diff", ascending=True)[pick_cols(transitions)]
        dec_without = transitions[(transitions["perf_diff"] < 0) & (transitions["sugg_next"] == 0)] \
                        .sort_values("perf_diff", ascending=True)[pick_cols(transitions)]
    else:
        # If suggestions column is missing, just do global increases/decreases
        inc_with = inc_without = transitions[0:0]
        dec_with = dec_without = transitions[0:0]

    # 7) Print (or save) results
    pd.set_option("display.max_columns", None)

    print("\n=== Biggest INCREASES (suggestions = YES in next session) ===")
    print(inc_with.head(10).to_string(index=False))

    print("\n=== Biggest INCREASES (suggestions = NO in next session) ===")
    print(inc_without.head(10).to_string(index=False))

    print("\n=== Biggest DECREASES (suggestions = YES in next session) ===")
    print(dec_with.head(10).to_string(index=False))

    print("\n=== Biggest DECREASES (suggestions = NO in next session) ===")
    print(dec_without.head(10).to_string(index=False))
    '''
    '''
    plt.figure(figsize=(8,6))
    sns.boxplot(data=DF, x="leader_type", y="performance", hue="suggestions", palette="Set2")

    plt.xticks([0,1], ["Human leader","AI leader"])
    plt.xlabel("Leader type")
    plt.ylabel("Performance")
    plt.title("Performance by leadership type and suggestion use")
    plt.legend(title="Suggestions", labels=["No","Yes"])
    plt.tight_layout()
    plt.show()
    '''
    
    '''
    data = pd.DataFrame({
    "Leader": ["AI"]*12 + ["Human"]*12,
    "Improved": [1]*10 + [0]*2 + [1]*5 + [0]*7
    })

    summary = data.groupby("Leader")["Improved"].agg(["mean","count","sum"])
    summary["ci_low"], summary["ci_high"] = zip(*summary.apply(
        lambda r: sm.stats.proportion_confint(r["sum"], r["count"], alpha=0.05, method="wilson"),
        axis=1
    ))

    plt.figure(figsize=(6,5))
    ax = sns.barplot(x=summary.index, y=summary["mean"], ci=None, palette="Set2")
    ax.errorbar(x=[0,1], y=summary["mean"],
                yerr=[summary["mean"]-summary["ci_low"], summary["ci_high"]-summary["mean"]],
                fmt="none", c="k", capsize=5)

    for i,(n,s) in enumerate(zip(summary["count"], summary["sum"])):
        ax.text(i, summary["mean"].iloc[i]+0.05, f"{s}/{n}", ha="center")

    plt.ylabel("Proportion improved")
    plt.ylim(0,1.1)
    plt.title("Probability of performance improvement per leader type")
    plt.show()
    '''
    
    # ---------- Build transitions (ACROSS-LEADER: prev session regardless of leader) ----------
    def build_transitions_across(df):
        d = (df.sort_values(["team_id","session"])
               .groupby(["team_id","session"], as_index=False)
               .agg(performance=("performance","mean"),
                    leader_type=("leader_type", lambda s: int(s.iloc[-1]))))
        d["perf_change"] = d.groupby("team_id")["performance"].diff()
        d["prev_leader"] = d.groupby("team_id")["leader_type"].shift(1)
        d["transition"] = d["prev_leader"].astype("Int64").astype(str) + "→" + d["leader_type"].astype(str)
        out = d.dropna(subset=["perf_change","prev_leader"]).copy()
        out["sign"] = np.where(out["perf_change"]>0, "Increase",
                        np.where(out["perf_change"]<0, "Decrease", "No change"))
        out = out[out["sign"]!="No change"].copy()
        out["magnitude"] = out["perf_change"].abs()
        out["scope"] = "Across-leader"
        return out[["team_id","transition","perf_change","magnitude","sign","scope"]]

    # ---------- Build transitions (WITHIN-LEADER: previous session with the same leader) ----------
    def build_transitions_within(df):
        d = (df.sort_values(["team_id","leader_type","session"])
               .groupby(["team_id","leader_type","session"], as_index=False)
               .agg(performance=("performance","mean")))
        d["perf_change"] = d.groupby(["team_id","leader_type"])["performance"].diff()
        d["prev_leader"] = d["leader_type"]  # within-leader, prev leader == curr leader
        d["transition"] = d["prev_leader"].astype(int).astype(str) + "→" + d["leader_type"].astype(int).astype(str)
        out = d.dropna(subset=["perf_change"]).copy()
        out["sign"] = np.where(out["perf_change"]>0, "Increase",
                        np.where(out["perf_change"]<0, "Decrease", "No change"))
        out = out[out["sign"]!="No change"].copy()
        out["magnitude"] = out["perf_change"].abs()
        out["scope"] = "Within-leader"
        return out[["team_id","transition","perf_change","magnitude","sign","scope"]]

    # Build both tables
    across = build_transitions_across(DF)
    within = build_transitions_within(DF)

    # Choose which to plot (run twice if you want both)
    to_plot = within  # or: within

    # Order transitions for x-axis
    x_order = ["0→0","0→1","1→0","1→1"]
    to_plot["transition"] = pd.Categorical(to_plot["transition"], categories=x_order, ordered=True)

    # Count labels per bar (n+ and n−)
    counts = (to_plot.groupby(["transition","sign"])["team_id"].count().unstack("sign").fillna(0).astype(int).rename(columns={"Increase":"n+","Decrease":"n−"}))

    '''
    # ---- Split violin + swarm of magnitudes (abs change), by transition ----
    plt.figure(figsize=(9,6))
    sns.violinplot(data=to_plot, x="transition", y="magnitude", hue="sign",
                   order=x_order, split=True, inner="quartile", cut=0)
    sns.stripplot(data=to_plot, x="transition", y="magnitude", hue="sign",
                  order=x_order, dodge=True, alpha=0.6, size=4, linewidth=0)

    # De-duplicate legends (since we added two layers)
    handles, labels = plt.gca().get_legend_handles_labels()
    bylab = dict(zip(labels, handles))
    plt.legend(bylab.values(), bylab.keys(), title="Direction", loc="upper right")

    plt.title(f"Distribution of |Δ performance| by transition ({to_plot['scope'].iloc[0]})")
    plt.xlabel("Leader transition (0=Human, 1=AI)")
    plt.ylabel("|Δ Performance|")

    # Annotate n+ and n− at the top of each x tick
    ylim = plt.gca().get_ylim()
    for i, t in enumerate(x_order):
        if t not in counts.index: continue
        txt = f"n+={counts.loc[t,'n+']}, n−={counts.loc[t,'n−']}"
        plt.text(i, ylim[1]*0.98, txt, ha="center", va="top", fontsize=9)

    plt.tight_layout()
    plt.show()
    '''


    '''
    rows = []
    for metric, values in final_average_survey_results.items():
        for session, val in enumerate(values):
            if "AI" in final_metrics["strategy"][session]:
                leader = "AI"
            else:
                leader = "human"
            rows.append({"metric": metric, "leader_type": leader, "session": session % 4, "value": val})

    subjective_df = pd.DataFrame(rows)
    subjective_df["session"] = subjective_df["session"] + 1
    
    
    sns.set(style="whitegrid", font_scale=1.1)
    
    fields = ['Mental Demand: How mentally demanding was the task?', 'Temporal Demand: How hurried or rushed was the pace of the task?', 'Performance: How successful were you in accomplishing what you were asked to do?', 'Effort: How hard did you have to work to accomplish your level of performance?', 'Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?']
    
    # Extract the short names (before the ":")
    short_labels = [f.split(":")[0] for f in fields]

    # Create subplots horizontally (1 row, many columns)
    fig, axes = plt.subplots(nrows=1, ncols=len(fields), figsize=(5*len(fields), 5), sharey=True)

    for ax, field, short_label in zip(axes, fields, short_labels):
        subset = subjective_df[subjective_df["metric"] == field]
        sns.lineplot(
            data=subset,
            x="session", y="value",
            hue="leader_type",
            marker="o",
            ax=ax,
            legend=True if short_label == short_labels[-1] else False  # keep legend only for last plot
        )
        ax.set_title(short_label)
        ax.set_xlabel("Session")
        ax.set_ylabel(short_label if ax == axes[0] else "")  # only leftmost has y-label
        ax.set_ylim(0, 21)

    # Extract legend handles/labels from the last axis (where we kept it)
    handles, labels = axes[-1].get_legend_handles_labels()
    #fig.legend(handles, labels, title="Leader Type", loc="lower center", ncol=2)

    #plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
    plt.show()
    '''
    
    """
    for metric in subjective_df["metric"].unique():
        
        plt.figure(figsize=(7, 5))
        subset = subjective_df[subjective_df["metric"] == metric]
        sns.lineplot(
            data=subset,
            x="session", y="value",
            hue="leader_type",
            marker="o"
        )
        plt.title(f"{metric.capitalize()} Across Sessions by Leadership Style")
        plt.xlabel("Session")
        plt.ylabel(metric.capitalize())
        plt.legend(title="Leader Type")
        plt.tight_layout()
        plt.show()
        
        '''
        subset = subjective_df[subjective_df["metric"] == metric]
        # --- Boxplot (distribution across sessions by leader type) ---
        plt.figure(figsize=(5, 5))
        sns.boxplot(
            data=subset,
            x="leader_type", y="value",
            palette="Set2"
        )
        sns.stripplot(   # optional: add individual session points
            data=subset,
            x="leader_type", y="value",
            color="black", size=6, jitter=False
        )
        
        plt.title(f"{metric.capitalize()} Distribution by Leadership Style")
        plt.xlabel("Leader Type")
        plt.ylabel(metric.capitalize())
        plt.tight_layout()
        plt.show()
        '''



    """

    #pdb.set_trace()
    '''
    print(mixed_df["difficulty"].unique())


    #mixed_df["leader_type"] = mixed_df["leader_type"].astype("category")
    #pdb.set_trace()

    # Mixed-effects model with random intercept for each team
    model = smf.mixedlm(
        "performance ~ leader_type + difficulty + suggestions + C(session) + C(layout) + C(team_id)", #"performance ~ leader_type * session + leader_type * difficulty + leader_type * distance",
        mixed_df,
        groups=mixed_df["team_id"]
    ).fit()

    print(model.summary())


    # Mixed-effects model with random intercept for each team
    model = smf.mixedlm(
        "performance ~  weight_sum", #"performance ~ leader_type * session + leader_type * difficulty + leader_type * distance",
        mixed_df,
        groups=mixed_df["team_id"]
    ).fit()

    print(model.summary())
    '''


    '''
    # FE model: team, session, and layout as fixed effects (categorical dummies)
    fe_formula = (
        "performance ~ leader_type * C(session) + difficulty + suggestions "
        "+ C(session) + C(layout) + C(team_id)"
    )

    fe_model = smf.ols(fe_formula, data=mixed_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": mixed_df["team_id"]}  # cluster by team
    )

    print(fe_model.summary())

    # Extract key info
    coef = fe_model.params["leader_type"]
    se   = fe_model.bse["leader_type"]
    print(f"\nLeader_type (FE) coef: {coef:.3f}  (SE: {se:.3f})")

    # Adjusted means by leader_type (holding FE and controls as in-sample)
    X0 = mixed_df.copy(); X0["leader_type"] = 0
    X1 = mixed_df.copy(); X1["leader_type"] = 1
    adj_mean_0 = fe_model.predict(X0).mean()
    adj_mean_1 = fe_model.predict(X1).mean()
    print(f"Adjusted mean perf: leader=0 → {adj_mean_0:.3f}, leader=1 → {adj_mean_1:.3f}, diff → {adj_mean_1-adj_mean_0:.3f}")

    fe_int = fe_model
    tests = {
        "s0": fe_int.t_test("leader_type = 0"),
        "s1": fe_int.t_test("leader_type + leader_type:C(session)[T.1] = 0"),
        "s2": fe_int.t_test("leader_type + leader_type:C(session)[T.2] = 0"),
        "s3": fe_int.t_test("leader_type + leader_type:C(session)[T.3] = 0"),
    }

    rows = []
    for k, res in tests.items():
        est = float(res.effect)
        se  = float(res.sd)
        t   = float(res.tvalue)
        p   = float(res.pvalue)
        lo, hi = map(float, res.conf_int().ravel())
        rows.append({"session": k, "effect_leader1_minus_0": est, "se": se, "t": t, "p": p, "ci_low": lo, "ci_high": hi})
    effects_by_session = pd.DataFrame(rows)
    print(effects_by_session)

    joint = fe_int.wald_test(
        "leader_type:C(session)[T.1] = 0, leader_type:C(session)[T.2] = 0, leader_type:C(session)[T.3] = 0"
    )
    print(joint)  # chi2, df, p-value

    early_vs_late = fe_int.t_test(
        "0.5*(leader_type:C(session)[T.2] + leader_type:C(session)[T.3] - leader_type:C(session)[T.1]) = 0"
    )
    print(early_vs_late)


    xx = [0,1,2,3]
    yy = [
        tests["s0"].effect.item(),
        tests["s1"].effect.item(),
        tests["s2"].effect.item(),
        tests["s3"].effect.item(),
    ]
    se = [
        tests["s0"].sd.item(),
        tests["s1"].sd.item(),
        tests["s2"].sd.item(),
        tests["s3"].sd.item(),
    ]
    ci = [1.96*s for s in se]

    plt.figure()
    plt.errorbar(xx, yy, yerr=ci, fmt='o-')
    plt.xticks(xx, ["S0","S1","S2","S3"])
    plt.xlabel("Session")
    plt.ylabel("Leader effect (1 − 0)")
    plt.title("Session-specific effect of leader_type")
    plt.tight_layout()
    plt.show()
    '''
    '''
    m_mod = smf.ols(
        "performance ~ leader_type + object_carry_speed  + C(session) + C(layout) + C(team_id)",
        data=mixed_df
    ).fit(cov_type="cluster", cov_kwds={"groups": mixed_df["team_id"]})

    print(m_mod.summary())
    # Mediator model
    m_med = smf.ols("object_carry_speed ~ leader_type + C(session)+C(layout)+C(team_id)", data=mixed_df).fit(
        cov_type="cluster", cov_kwds={"groups": mixed_df["team_id"]}
    )
    print(m_med.summary())
    '''
    # Outcome with mediator
    y_med = smf.ols("performance ~ C(leader_type) + object_dangerous_time*C(layout) + idle_time +  object_benign_time + C(session)+C(team_id)", data=mixed_df).fit(
        cov_type="cluster", cov_kwds={"groups": mixed_df["team_id"]}
    )
    print(y_med.summary())
    
    '''
    # Indirect ≈ (effect of leader on mediator) * (effect of mediator on performance)
    indirect = m_med.params["leader_type"] * y_med.params["object_carry_speed"]
    print(indirect)
    # (Bootstrap this product for CI)
    '''
    FEATURE = "total_danger_weight" #"object_dangerous_time" #'distance'            # <-- change to your feature column
    TARGET = "performance"

    out = corr_one_feature(mixed_df, FEATURE)
    print(out)
    
    
    
    DF = mixed_df                   # <-- your DataFrame with columns: layout, perf, <FEATURE>, (optional) trial_id


    pivot = (DF.groupby(["layout","session","leader_type"])["performance"]
               .mean().unstack("leader_type"))
    pivot["delta_AI_minus_Human"] = pivot[1] - pivot[0]
    M = pivot["delta_AI_minus_Human"].unstack("session").sort_index()

    '''
    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="Δ performance (AI−Human)")
    plt.xticks(range(M.shape[1]), M.columns)
    plt.yticks(range(M.shape[0]), M.index)
    plt.xlabel("Session"); plt.ylabel("Layout")
    plt.title("Leader advantage by layout × session")
    plt.tight_layout(); plt.show()
    '''

    inc_within, dec_within   = top_changes_within_leader(DF, 10)
    inc_across, dec_across   = top_changes_across_leader(DF, 10)
    team_perf_range_with_context(DF, 10)
    team_first_last_diff(DF,len(final_metrics["strategy"]))
    team_performance_profile(DF,len(final_metrics["strategy"]))
    
    
    
    # 1) Compute mean perf by team × leader × suggestions
    grp = (DF.groupby(["team_id","leader_type","suggestions"])["performance"]
             .mean()
             .unstack("suggestions"))

    # Ensure columns exist
    for c in [0,1]:
        if c not in grp.columns:
            grp[c] = np.nan

    # 2) Delta = with suggestions − without suggestions
    grp["delta"] = grp[1] - grp[0]

    # 3) Wide table of deltas (per team × leader_type)
    delta_wide = grp["delta"].unstack("leader_type")

    # 4) Collect names used for each (team, leader, suggestions)
    names = (DF.groupby(["team_id","leader_type","suggestions"])["name"]
               .apply(lambda s: ", ".join(sorted(set(s.dropna().astype(str)))))
               .unstack("suggestions"))

    # 5) For annotations: concatenate names from with/without suggestions
    names["label"] = names.apply(
        lambda r: f"with: {r.get(1,'')} | without: {r.get(0,'')}", axis=1
    )

    # 6) Order teams by AI−Human delta
    order = (delta_wide.get(1, pd.Series(index=delta_wide.index, dtype=float))
             - delta_wide.get(0, pd.Series(index=delta_wide.index, dtype=float))).sort_values()

    '''
    plt.figure(figsize=(9,6))
    ypos = range(len(order))

    # Human vs AI deltas
    plt.scatter(delta_wide.loc[order.index, 0], ypos, marker='o', label="Human Δ")
    plt.scatter(delta_wide.loc[order.index, 1], ypos, marker='s', label="AI Δ")

    # Connecting lines + annotations
    for i, t in enumerate(order.index):
        x_h = delta_wide.loc[t, 0] if 0 in delta_wide.columns else np.nan
        x_a = delta_wide.loc[t, 1] if 1 in delta_wide.columns else np.nan

        if pd.notna(x_h) and pd.notna(x_a):
            plt.plot([x_h, x_a], [i, i], color='0.6', linewidth=1)

        # Annotate names next to each point
        if pd.notna(x_h):
            label_h = names.loc[(t,0),"label"] if (t,0) in names.index else ""
            plt.annotate(label_h, (x_h, i),
                         xytext=(-6,-2), textcoords="offset points",
                         ha="right", va="center", fontsize=7, color="C0")

        if pd.notna(x_a):
            label_a = names.loc[(t,1),"label"] if (t,1) in names.index else ""
            plt.annotate(label_a, (x_a, i),
                         xytext=(6,-2), textcoords="offset points",
                         ha="left", va="center", fontsize=7, color="C1")

    plt.axvline(0, linestyle='--', color='k', alpha=0.7)
    plt.yticks(ypos, order.index)
    plt.xlabel("Δ performance (with − without suggestions)")
    plt.ylabel("Team #")
    plt.title("Per-team suggestion effect by leadership type\n(names annotated)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    '''

    #pdb.set_trace()


    # --- Build transitions between consecutive sessions per team ---
    df_sorted = DF.sort_values(["team_id","session"]).copy()
    df_sorted["perf_next"]   = df_sorted.groupby("team_id")["performance"].shift(-1)
    df_sorted["leader_next"] = df_sorted.groupby("team_id")["leader_type"].shift(-1)
    df_sorted["perf_change"] = df_sorted["perf_next"] - df_sorted["performance"]
    df_sorted = df_sorted.dropna(subset=["leader_next","perf_change"]).copy()

    # Make a readable transition label: 0→0, 0→1, 1→0, 1→1 (0=Human, 1=AI)
    df_sorted["transition"] = df_sorted["leader_type"].astype(int).astype(str) \
                            + "→" + df_sorted["leader_next"].astype(int).astype(str)

    # --- Summaries per transition ---
    def summarize_changes(g):
        inc = g.loc[g["perf_change"] > 0, "perf_change"]
        dec = g.loc[g["perf_change"] < 0, "perf_change"]
        return pd.Series({
            "mean_increase": inc.mean() if not inc.empty else np.nan,
            "mean_decrease": dec.mean() if not dec.empty else np.nan,
            "mean_change": g["perf_change"].mean()
        })

    summary = df_sorted.groupby("transition").apply(summarize_changes).reset_index()

    # Counts per transition: total / increases / decreases
    counts = df_sorted.groupby("transition").agg(
        n_total   = ("perf_change","size"),
        n_increase= ("perf_change", lambda s: (s > 0).sum()),
        n_decrease= ("perf_change", lambda s: (s < 0).sum())
    ).reset_index()

    # Map from (stat, transition) -> n
    count_map = {}
    for _, r in counts.iterrows():
        t = r["transition"]
        count_map[("mean_increase", t)] = int(r["n_increase"])
        count_map[("mean_decrease", t)] = int(r["n_decrease"])
        count_map[("mean_change",   t)] = int(r["n_total"])

    # Long form for plotting
    summary_melt = summary.melt(
        id_vars="transition",
        value_vars=["mean_increase","mean_decrease","mean_change"],
        var_name="stat", value_name="value"
    )

    # Attach per-bar sample counts
    summary_melt["n"] = [
        count_map.get((stat, trans), 0)
        for stat, trans in zip(summary_melt["stat"], summary_melt["transition"])
    ]

    # Fix plotting order so annotations align predictably
    order      = ["0→0", "0→1", "1→0", "1→1"]  # adjust if you prefer a different order
    hue_order  = ["mean_increase","mean_decrease","mean_change"]


    '''
    plt.figure(figsize=(9,6))
    ax = sns.barplot(
        data=summary_melt, x="transition", y="value", hue="stat",
        order=order, hue_order=hue_order
    )
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.title("Performance changes by leader transition\n(with per-bar sample counts)")
    plt.ylabel("Δ Performance")
    plt.xlabel("Leader transition (0=Human, 1=AI)")

    # Annotate counts on every bar, positioned above for positive bars and below for negative bars
    # ax.containers aligns with hue_order; each container has bars for x order
    for container, stat in zip(ax.containers, hue_order):
        # corresponding transitions in the chosen x order
        for bar, trans in zip(container, order):
            height = bar.get_height()
            if np.isnan(height):
                continue
            n = count_map.get((stat, trans), 0)
            # Offset: a bit above if bar is positive, a bit below if negative
            offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            y_text = height + (offset if height >= 0 else -offset)
            ax.text(
                bar.get_x() + bar.get_width()/2,
                y_text,
                f"n={n}",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=9, color="black"
            )

    plt.legend(title="Statistic", loc="best", frameon=True)
    plt.tight_layout()
    plt.show()
    '''
    
    plot_variable_slot_width_diverging_from_df(df_sorted, gap=0.01, show_total=False)

    
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df_sorted.dropna(subset=["perf_change"]), x="transition", y="perf_change")
    plt.axhline(0, color="k", linestyle="--")
    plt.title("Distribution of performance changes per transition")
    plt.ylabel("Δ Performance")
    plt.xlabel("Leader transition")
    plt.tight_layout()
    plt.show()
    

    # Build: per (team_id, leader_type) -> list of layouts where they led
    layouts_by_team_leader = (
        DF.groupby(["team_id", "leader_type"])["layout"]
          .apply(lambda s: sorted(pd.unique(s.dropna().astype(int))))
          .to_dict()
    )

    names_by_team_leader = (
        DF.groupby(["team_id", "leader_type"])["name"]
          .apply(lambda s: sorted(pd.unique(s.dropna().astype(int))))
          .to_dict()
    )

    def fmt_layouts(lst):
        # e.g., [1,3,4] -> "L1,L3,L4"; handle empty/missing
        if not lst:
            return "L–"
        return "L" + ",L".join(str(x) for x in lst)

    def fmt_names(lst):
        # e.g., [1,3,4] -> "L1,L3,L4"; handle empty/missing
        if not lst:
            return "–"
        return "[" + ", ".join(str(x) for x in lst) + "]"

    # Your existing mean table and order
    g = DF.groupby(["team_id","leader_type"])["performance"].mean().unstack()
    order = (g.get(1, pd.Series(index=g.index, dtype=float)) - g.get(0, pd.Series(index=g.index, dtype=float))).sort_values()

    
    plt.figure()
    
    
    #Block Testing wihin mean performance------
    # Keep teams that have both leader types
    g2 = g.copy()
    if 0 not in g2.columns: g2[0] = np.nan
    if 1 not in g2.columns: g2[1] = np.nan
    g2 = g2[[0, 1]].dropna()  # rows with both Human(0) and AI(1)

    if len(g2) < 2 or g2[0].isna().all() or g2[1].isna().all():
        print("Not enough teams with both Human and AI means to run the asymmetry test.")
    else:
        df_gap = (
            g2.rename(columns={0: "human", 1: "ai"})
              .assign(gap=lambda d: d["ai"] - d["human"],
                      abs_gap=lambda d: (d["ai"] - d["human"]).abs(),
                      ai_wins=lambda d: (d["ai"] > d["human"]).astype(int))
        )

        # Group summaries
        summary = df_gap.groupby("ai_wins")["abs_gap"].agg(["mean", "std", "count"])
        print("\nAbs gap by group (0=Human wins, 1=AI wins):")
        print(summary)

        # Split samples
        pos = df_gap.loc[df_gap["ai_wins"] == 1, "abs_gap"]  # AI > Human
        neg = df_gap.loc[df_gap["ai_wins"] == 0, "abs_gap"]  # Human > AI

        if len(pos) == 0 or len(neg) == 0:
            print("\nOne group is empty—cannot run two-sample tests.")
        else:
            # Welch t-test: H1 = abs_gap(AI wins) > abs_gap(Human wins)
            tval, pval = stats.ttest_ind(pos, neg, equal_var=False, alternative="greater")

            # Mann–Whitney U (robust to non-normality), same alternative
            uval, p_u = stats.mannwhitneyu(pos, neg, alternative="greater")

            # Cohen's d (Hedges' g correction)
            def cohens_d(a, b):
                na, nb = len(a), len(b)
                sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
                s_p = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
                if s_p == 0:
                    return 0.0
                d = (np.mean(a) - np.mean(b)) / s_p
                J = 1 - 3 / (4 * (na + nb) - 9)  # small-sample correction
                return d * J

            d = cohens_d(pos.values, neg.values)

            # Cliff's delta (nonparametric effect size)
            def cliffs_delta(x, y):
                x = np.asarray(x); y = np.asarray(y)
                gt = sum((xi > y).sum() for xi in x)
                lt = sum((xi < y).sum() for xi in x)
                return (gt - lt) / (len(x) * len(y))

            delta = cliffs_delta(pos.values, neg.values)

            print(f"\nWelch t-test (abs_gap | AI_wins > Human_wins): t = {tval:.3f}, p = {pval:.4g}")
            print(f"Mann–Whitney U (greater): U = {uval:.1f}, p = {p_u:.4g}")
            print(f"Effect sizes: Cohen's d = {d:.2f}, Cliff's delta = {delta:.2f}")

            # Optional: quick textual takeaway
            mean_ai = summary.loc[1, "mean"] if 1 in summary.index else np.nan
            mean_h  = summary.loc[0, "mean"] if 0 in summary.index else np.nan
            print(f"\nMean |gap| when AI wins: {mean_ai:.2f}  vs  when Human wins: {mean_h:.2f}")


    pos = df_gap.loc[df_gap["ai_wins"]==1, "abs_gap"].values  # AI wins
    neg = df_gap.loc[df_gap["ai_wins"]==0, "abs_gap"].values  # Human wins
    obs = pos.mean() - neg.mean()

    combined = np.concatenate([pos, neg])
    n_pos = len(pos)

    iters = 100_000
    count = 0
    for _ in range(iters):
        np.random.shuffle(combined)
        new_pos = combined[:n_pos]
        new_neg = combined[n_pos:]
        if new_pos.mean() - new_neg.mean() >= obs:
            count += 1

    p_perm = (count + 1) / (iters + 1)  # one-sided
    print("Permutation p (one-sided):", p_perm)

    rng = np.random.default_rng(0)
    B = 100_000
    boot = []
    for _ in range(B):
        boot.append(rng.choice(pos, size=len(pos), replace=True).mean()
                   - rng.choice(neg, size=len(neg), replace=True).mean())
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print("Bootstrap 95% CI for mean(|gap|_AIwins - |gap|_Humanwins):", (ci_lo, ci_hi))
    #Block Testing within performance----
    
    '''
    #Block2 ---
    
    g2 = g.copy()
    if 0 not in g2.columns: g2[0] = np.nan
    if 1 not in g2.columns: g2[1] = np.nan
    g2 = g2[[0,1]].dropna().rename(columns={0:"human", 1:"ai"})

    # Core derived variables
    g2["gap"] = g2["ai"] - g2["human"]
    g2["abs_gap"] = g2["gap"].abs()
    eps = 1e-9
    g2["gap_pct_mid"] = g2["gap"] / ((g2["ai"] + g2["human"])/2 + eps)  # symmetric %
    g2["gap_pct_h"]   = g2["gap"] / (g2["human"] + eps)                 # vs human baseline

    print("\n--- Simple correlations (exploratory) ---")
    # Expect a negative correlation: lower human → larger |gap|
    r_p, p_p = stats.pearsonr(g2["human"], g2["abs_gap"])
    r_s, p_s = stats.spearmanr(g2["human"], g2["abs_gap"])
    print(f"Pearson corr(human, |gap|): r={r_p:.3f}, p={p_p:.4g}")
    print(f"Spearman corr(human, |gap|): ρ={r_s:.3f}, p={p_s:.4g}")

    print("\n--- Controlled association (preferred) ---")
    # |gap| ~ human + ai (both as predictors)
    X = sm.add_constant(g2[["human","ai"]])
    y = g2["abs_gap"]
    mod = sm.OLS(y, X).fit()
    print(mod.summary().as_text())

    # Optional: standardize predictors for comparability
    g2_std = g2.copy()
    for c in ["human","ai","abs_gap"]:
        g2_std[c] = (g2_std[c] - g2_std[c].mean()) / (g2_std[c].std(ddof=1) + eps)
    Xz = sm.add_constant(g2_std[["human","ai"]])
    mod_z = sm.OLS(g2_std["abs_gap"], Xz).fit()
    print("\n(Standardized) |gap| ~ z(human) + z(ai):")
    print(mod_z.summary().as_text())

    print("\n--- Relative gap checks (scale-free) ---")
    # Correlate human with relative gaps
    r1, p1 = stats.pearsonr(g2["human"], g2["gap_pct_mid"].abs())
    r2, p2 = stats.pearsonr(g2["human"], g2["gap_pct_h"].abs())
    print(f"corr(human, |gap_pct_mid|): r={r1:.3f}, p={p1:.4g}")
    print(f"corr(human, |gap_pct_h|):   r={r2:.3f}, p={p2:.4g}")
    
    #Block2---
    '''
    
    ypos = range(len(order))

    # scatter points
    plt.scatter(g.loc[order.index, 0], ypos, marker='o', label="Human")
    plt.scatter(g.loc[order.index, 1], ypos, marker='s', label="AI")

    # connecting lines + annotations
    for i, t in enumerate(order.index):
        x_h = g.loc[t, 0] if 0 in g.columns and t in g.index else np.nan
        x_a = g.loc[t, 1] if 1 in g.columns and t in g.index else np.nan

        # connect if both exist
        if pd.notna(x_h) and pd.notna(x_a):
            plt.plot([x_h, x_a], [i, i], color='0.6', linewidth=1)

        # annotate layouts next to Human point (left side)
        if pd.notna(x_h):
            labs_h = fmt_layouts(layouts_by_team_leader.get((t, 0), []))
            nabs_h = fmt_names(names_by_team_leader.get((t, 0), []))
            #plt.annotate(nabs_h, (x_h, i), xytext=(-6, -2), textcoords='offset points',
            #             ha='right', va='center', fontsize=8, color='C0')

        # annotate layouts next to AI point (right side)
        if pd.notna(x_a):
            labs_a = fmt_layouts(layouts_by_team_leader.get((t, 1), []))
            nabs_a = fmt_names(names_by_team_leader.get((t, 1), []))
            #plt.annotate(nabs_a, (x_a, i), xytext=(6, -2), textcoords='offset points',
            #             ha='left', va='center', fontsize=8, color='C1')

    plt.yticks(ypos, order.index)
    plt.xlabel("Mean performance")
    plt.ylabel("Team #")
    plt.title("Within-team performance: Human vs AI")
    plt.legend()
    plt.tight_layout()
    plt.show()


    '''
    rows=[]
    for tid, g in DF.groupby("team_id"):
        d = g.groupby("leader_type")["performance"].mean()
        delta = d.get(1, np.nan) - d.get(0, np.nan)
        rows.append((tid, delta))
    fx = pd.DataFrame(rows, columns=["team_id","delta"]).dropna().sort_values("delta")
    y = np.arange(len(fx))
    plt.figure()
    plt.errorbar(fx["delta"], y, xerr=0, fmt='o')  # add bootstrapped xerr if you have it
    plt.axvline(0, ls='--')
    plt.yticks(y, fx["team_id"]); plt.xlabel("AI − Human mean performance"); plt.title("Team-specific leader effect")
    plt.ylabel("Team #")
    plt.tight_layout(); plt.show()

    sns.violinplot(data=DF, x="leader_type", y="performance", cut=0)
    plt.title("Distribution of performance by leader type"); plt.show()
    '''
    '''
    # per-team deltas
    g = DF.groupby(['team_id','leader_type'])['performance'].mean().unstack()
    deltas = (g[1] - g[0]).dropna()  # Δ_i

    pos = deltas[deltas > 0].to_numpy()
    neg = (-deltas[deltas < 0]).to_numpy()  # absolute magnitudes where Human>AI

    rng = np.random.default_rng(0)

    # bootstrap CI for mean difference in magnitudes
    def boot_mean_diff(x, y, B=5000):
        n = len(x); m = len(y)
        diffs = []
        for _ in range(B):
            xb = x[rng.integers(0, n, n)]
            yb = y[rng.integers(0, m, m)]
            diffs.append(xb.mean() - yb.mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return (np.mean(x)-np.mean(y), lo, hi)

    mean_diff, lo, hi = boot_mean_diff(pos, neg)

    # sign-flip permutation on magnitudes (one-sided p)
    magnitudes = np.concatenate([pos, neg])
    labels = np.array([1]*len(pos) + [0]*len(neg))  # 1=AI wins, 0=Human wins
    obs = pos.mean() - neg.mean()
    B = 10000; count = 0
    for _ in range(B):
        perm = rng.permutation(labels)
        pos_m = magnitudes[perm==1]
        neg_m = magnitudes[perm==0]
        if pos_m.size>0 and neg_m.size>0:
            if (pos_m.mean() - neg_m.mean()) >= obs - 1e-12:
                count += 1
    p_one_sided = (count+1)/(B+1)
    print({'mean_pos':pos.mean(), 'mean_neg':neg.mean(),
           'diff':obs, 'boot_CI':[lo,hi], 'perm_p_one_sided':p_one_sided})

    base_h = DF[DF.leader_type==0].groupby('team_id')['performance'].mean()
    base_a = DF[DF.leader_type==1].groupby('team_id')['performance'].mean()
    delta = (base_a - base_h).dropna()
    base = base_h.loc[delta.index]

    import statsmodels.api as sm
    X = sm.add_constant(base.values)
    res = sm.OLS(delta.values, X).fit()
    print(res.summary())  # look for negative slope
    '''
    
    '''
    # --- Plot: rank–rank scatter per layout ---
    for L, g in DF.dropna(subset=['performance', FEATURE]).groupby('layout'):
        if len(g) < 3:
            continue  # too few points to visualize meaningfully

        # ranks (average for ties)
        x = g[FEATURE].rank(method='average').values
        y = g['performance'].rank(method='average').values

        # correlation
        r, p = spearmanr(x, y)

        # simple fitted line on ranks (least-squares on ranks; just a visual guide)
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = np.polyval(coef, x_line)

        # plot
        plt.figure()
        plt.scatter(x, y)
        plt.plot(x_line, y_line)
        title = f'Layout {L}: {FEATURE} vs performance (rank–rank)\nSpearman ρ={r:.3f}, p={p:.4f}, n={len(g)}'
        plt.title(title)
        plt.xlabel(f'{FEATURE} (rank)')
        plt.ylabel('performance (rank)')

        # optional: label points by trial_id if available
        if 'session' in g.columns:
            for xi, yi, tid in zip(x, y, g['session'].astype(str).values):
                plt.annotate(tid, (xi, yi), xytext=(3, 3), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.show()
    '''
    """
    for L, g in DF.dropna(subset=[FEATURE, TARGET]).groupby('layout'):
        if len(g) < 3:
            continue
        x = g[FEATURE].to_numpy()
        y = g[TARGET].to_numpy()
        r, p = spearmanr(x, y)
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = np.polyval(coef, xs)

        plt.figure()
        plt.scatter(x, y)
        plt.plot(xs, ys)
        plt.title(f'Layout {L}: {FEATURE} vs {TARGET}\nSpearman ρ={r:.3f}, p={p:.4f}, n={len(g)}')
        plt.xlabel(f'{FEATURE}')
        plt.ylabel(f'{TARGET}')
        plt.grid(True, alpha=0.3)
        if 'session' in g.columns:
            for xi, yi, tid, nid in zip(x, y, g['session'].astype(str).values, g['name'].astype(str).values):
                plt.annotate(tid + " [" + nid + "]", (xi, yi), xytext=(3, 3), textcoords='offset points', fontsize=8)
        plt.tight_layout()
        plt.show()
    """

    
    if True:
        # Drop NA in the features of interest
        DF[FEATURE] = DF[FEATURE].replace(0, np.nan)
        g = DF.dropna(subset=[FEATURE, TARGET])

        x = g[FEATURE].to_numpy()
        y = g[TARGET].to_numpy()

        plt.figure(figsize=(8,6))

        # scatter plot (color by leader_type, shape optional)
        scatter = plt.scatter(
            x, y, 
            c=g["leader_type"], cmap="coolwarm", 
            marker="o", alpha=0.8
        )

        # regression line across all points
        coef = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = np.polyval(coef, xs)
        plt.plot(xs, ys, color="k", lw=2, label="OLS fit")

        # annotate each point with its 'name'
        for xi, yi, label in zip(x, y, g["name"].astype(str).values):
            plt.annotate(label, (xi, yi),
                         xytext=(3,3), textcoords="offset points",
                         fontsize=8)

        plt.xlabel(FEATURE)
        plt.ylabel(TARGET)
        plt.title(f"All layouts combined: {FEATURE} vs {TARGET}")
        plt.axhline(0, color="k", lw=0.5, linestyle="--")
        plt.axvline(0, color="k", lw=0.5, linestyle="--")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    

    
        # set this True if you want a separate regression line per leader_type
        SEPARATE_FITS_BY_LEADER = True

        for L, g in DF.dropna(subset=[FEATURE, TARGET]).groupby('layout'):
            if len(g) < 3:
                continue

            x = g[FEATURE].to_numpy()
            y = g[TARGET].to_numpy()
            r, p = spearmanr(x, y)

            # overall fit
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ys = np.polyval(coef, xs)

            plt.figure()
            ax = plt.gca()

            # --- color by leader_type ---
            if 'leader_type' in g.columns:
                leader_vals = sorted(g['leader_type'].dropna().unique())
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                colmap = {lt: color_cycle[i % len(color_cycle)] for i, lt in enumerate(leader_vals)}
            else:
                leader_vals = [None]
                colmap = {None: color_cycle[0]}

            # --- marker by session ---
            if 'session' in g.columns:
                session_vals = sorted(g['session'].dropna().unique())
                markers = ['o','s','D','^','v','<','>','P','X','*']  # cycle if many sessions
                markmap = {s: markers[i % len(markers)] for i, s in enumerate(session_vals)}
            else:
                session_vals = [None]
                markmap = {None: 'o'}

            # plot by (leader_type, session) combination
            for lt in leader_vals:
                for s in session_vals:
                    mask = np.ones(len(g), dtype=bool)
                    if 'leader_type' in g.columns:
                        mask &= (g['leader_type'].to_numpy() == lt)
                    if 'session' in g.columns:
                        mask &= (g['session'].to_numpy() == s)
                    if not mask.any():
                        continue
                    plt.scatter(
                        x[mask], y[mask],
                        color=colmap[lt],
                        marker=markmap[s],
                        alpha=0.9, s=42
                    )

                    # OPTIONAL: separate fit lines per leader_type
                    if SEPARATE_FITS_BY_LEADER and (g['leader_type'].to_numpy() == lt).sum() >= 2:
                        m_lt = (g['leader_type'].to_numpy() == lt)
                        coef_lt = np.polyfit(x[m_lt], y[m_lt], 1)
                        xs_lt = np.linspace(x[m_lt].min(), x[m_lt].max(), 50)
                        ys_lt = np.polyval(coef_lt, xs_lt)
                        plt.plot(xs_lt, ys_lt, linestyle='--', linewidth=1.8,
                                 color=colmap[lt], alpha=0.9)

            # overall fit (black)
            plt.plot(xs, ys, linewidth=2, alpha=0.95, color='k', label='overall fit')

            plt.title(f'Layout {L}: {FEATURE} vs {TARGET}\nSpearman ρ={r:.3f}, p={p:.4f}, n={len(g)}')
            plt.xlabel(f'{FEATURE}')
            plt.ylabel(f'{TARGET}')
            plt.grid(True, alpha=0.3)

            # keep your annotations
            if 'session' in g.columns and 'name' in g.columns:
                for xi, yi, tid, nid in zip(x, y, g['session'].astype(str).values, g['name'].astype(str).values):
                    plt.annotate(tid + " [" + nid + "]", (xi, yi),
                                 xytext=(3, 3), textcoords='offset points', fontsize=8)

            # --- build two legends: colors (leader_type) and markers (session) ---
            color_handles = [
                Line2D([0], [0], marker='o', linestyle='None',
                       markerfacecolor=colmap[lt], markeredgecolor=colmap[lt], markersize=8,
                       label=f'leader_type={lt}')
                for lt in leader_vals
            ]
            marker_handles = [
                Line2D([0], [0], marker=markmap[s], linestyle='None',
                       color='k', markersize=8, label=f'session={s}')
                for s in session_vals
            ]

            # place the legends (adjust locs if they overlap)
            leg1 = ax.legend(handles=color_handles + [Line2D([0],[0], color='k', lw=2, label='overall fit')],
                             title='Leader / Fit', loc='best', frameon=True)
            ax.add_artist(leg1)
            ax.legend(handles=marker_handles, title='Session', loc='upper left', frameon=True)

            plt.tight_layout()
            plt.show()

        

        within_team = (
            mixed_df.groupby(["team_id", "leader_type"])["performance"]
            .mean()
            .unstack()
            .assign(diff=lambda x: x[1] - x[0])
        )

        print(within_team)

        print("Mean difference:", within_team["diff"].mean())
        print("Teams preferring AI:", (within_team["diff"] > 0).sum())
        print("Teams preferring Human:", (within_team["diff"] < 0).sum())

        last_sessions = mixed_df[mixed_df["session"].isin([2,3])]
        contrast_last = (
            last_sessions.groupby(["team_id","leader_type"])["performance"]
            .mean()
            .unstack()
            .assign(diff=lambda x: x[1]-x[0])
        )
        print(contrast_last)
        print("Mean diff:", contrast_last["diff"].mean())
        print("Teams preferring AI:", (contrast_last["diff"]>0).sum())
        print("Teams preferring Human:", (contrast_last["diff"]<0).sum())

        
        mixed_df["leader_type"] = mixed_df["leader_type"].replace({0: "human", 1: "AI"})
        mixed_df["session"] = mixed_df["session"] + 1

        sns.lineplot(
            data=mixed_df,
            x="session", y="performance",
            hue="leader_type", ci="sd", estimator="mean"
        )
        plt.title("Performance Trajectories by Leadership Style")
        plt.show()

        best_leader = (
            mixed_df.groupby(["team_id", "leader_type"])["performance"]
            .mean()
            .reset_index()
            .sort_values(["team_id","performance"])
            .groupby("team_id").last()
        )

        sns.countplot(data=best_leader, x="leader_type")
        plt.title("Which Leadership Style Yielded Highest Team Performance")
        plt.show()

if True:
    legacy = True
    if not legacy:
        '''
        for m in range(multiples_final_metrics):
            # Create a single figure with two subplots side by side, each with its own y‑scale
            fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=False)

            legends = final_metrics["strategy"][number_scenarios*(m):number_scenarios*(m+1)]
            # Plot each group on its own axis
            plot_group(axes[0], group_final, 'Final Questions',legends,m)
            plot_group(axes[1], group_other, 'Other Questions',legends,m)

            # Shared legend below the subplots
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=len(legends), title='Options')

            plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
            plt.show()


        '''
        
        # Derive category indices and labels from the first dict
        cat_indices = sorted(final_metrics["distribution_sensed"][0].keys())
        distribution_categories = [f"Sensed {i} times" for i in cat_indices]

        # Grab the default colors
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:number_scenarios]



        """
        arrangement = final_metrics["strategy"][:number_scenarios]

        skip_metrics = ["name","strategy","weight_distribution_carried","suggestions_taken","suggestions_given","sensing_complexity",'experience_llm','total_distance_traveled']

        for m in range(multiples_final_metrics):

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            axes = axes.flatten()
            fig.subplots_adjust(hspace=0.4, wspace=0.3)
            real_idx = 0
            y_limit_complex = 1

            old_legends = final_metrics["strategy"][number_scenarios*(m):number_scenarios*(m+1)]
            
            for idx, (metric, vals) in enumerate(final_metrics.items()):

                vals = vals[number_scenarios*(m):number_scenarios*(m+1)]
                
                '''
                true_vals = []
                for ar in arrangement:
                    true_vals.append(vals[old_legends.index(ar)])

                vals = true_vals
                
                
                legends = arrangement
                '''
                legends = old_legends
                if metric in skip_metrics:
                    continue
                
                if metric == 'experience_videogame':
                    tmp_experience = vals.copy()
                    continue
                
                ax = axes[real_idx]
                real_idx += 1
                  
                
                if metric == 'experience_teamcollab':
                    x = np.arange(2)
                    width = 0.35  # Width of bars
                    bars1 = ax.bar(x - width/2, tmp_experience[0], width, label='Video Games')
                    bars2 = ax.bar(x + width/2, vals[0], width, label='TeamCollab')

                    # Add labels and legend
                    ax.set_xlabel('Participants')
                    ax.set_ylabel('Values')
                    ax.set_title('Grouped Bar Plot')
                    ax.set_xticks(x)
                    ax.set_xticklabels(['A','B'])
                    ax.legend()
                    ax.set_ylim(top=3)
                elif metric != "distribution_sensed":
                
                    if metric == 'complexity_reduction':
                        try:
                            ax.set_ylim(top=y_limit_complex)
                        except:
                            pdb.set_trace()    
                    
                        
                    x = np.arange(len(legends))
                    try:
                        ax.bar(x, vals, width=0.6, color=colors)
                    except:
                        pdb.set_trace()
                    ax.set_xticks(x)
                    ax.set_xticklabels(legends, rotation=45, ha="right")
                    
                    if metric == 'complexity':
                        _, y_limit_complex = ax.get_ylim()
                    
                    if "dangerous_objects" in metric:
                        metric = "Dangerous objects carried"
                    elif "complexity_reduction" in metric:
                        metric = "AWRC Reduction"
                    elif "complexity" in metric:
                        metric = "Agent‑Weighted Retrieval Cost (AWRC)"
                    
                    ax.set_ylabel(metric.replace("_", " ").title())
                    
                    ax.set_title(metric.replace("_", " ").title())
                    
                    
                else:
                    data_dicts = vals  # list of 4 dicts
                    n_cat = len(cat_indices)
                    n_leg = len(legends)
                    x = np.arange(n_cat)
                    bar_w = 0.8 / n_leg

                    # For each legend i, extract its series across categories
                    for i in range(n_leg):
                        try:
                            series = [data_dicts[i][ci] for ci in cat_indices]
                        except:
                            pdb.set_trace()
                        ax.bar(x + i*bar_w, series, width=bar_w, color=colors[i])
                    
                    # Draw separators between category groups
                    for sep in x + bar_w*n_leg:
                        ax.axvline(sep, color='grey', linestyle='--', linewidth=1, label='_nolegend_')
                    
                    ax.set_xticks(x + bar_w*(n_leg-1)/2)
                    ax.set_xticklabels(distribution_categories, rotation=45, ha="right")
                    ax.set_ylabel("Proportion Sensed")
                    ax.set_title("Frequency Distribution of Object Sensing Events")

            # Remove any unused axes
            for idx,j in enumerate(range(len(final_metrics), len(axes))):
                if idx >= 4:
                    break
                fig.delaxes(axes[j])

            # Shared legend in the middle, with solid boxes
            handles = [Patch(facecolor=colors[i]) for i in range(len(legends))]
            fig.legend(
                handles,
                legends,
                loc='center',
                bbox_to_anchor=(0.5, 0.52),
                ncol=len(legends),
                title="Scenario",
                frameon=False
            )

            plt.tight_layout()
            plt.show()
        """
    else:

        # Create DataFrame
        
        '''
        orig_dict = {
            'complexity': final_metrics["complexity"],
            'complexity_reduction': final_metrics["complexity_reduction"],
            'strategy': final_metrics["strategy"],
            "name": final_metrics["name"],
        }
        
        
        df = pd.DataFrame(orig_dict)
        
        

        df = pd.DataFrame(orig_dict).copy()
        df["strategy"] = df["strategy"].astype(str).str.strip()   # ensure clean labels

        # Keep the order of appearance in the data
        order = sorted(df["strategy"].unique())

        sns.set(style="whitegrid", font_scale=1.1)

        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="strategy", y="complexity_reduction", order=order)
        plt.title("Complexity Reduction by Strategy")
        plt.xlabel("Strategy")
        plt.ylabel("Complexity Reduction")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        
        pdb.set_trace()
        # Create figure with two side-by-side boxplots
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6), sharey=False)

        # Boxplot of complexity by strategy
        df.boxplot(column='complexity',
                   by='strategy',
                   ax=ax1,
                   grid=False)
        ax1.set_title('Complexity by Strategy')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Complexity')
        ax1.tick_params(axis='x', rotation=45)

        # Boxplot of complexity reduction by strategy
        df.boxplot(column='complexity_reduction',
                   by='strategy',
                   ax=ax2,
                   grid=False)
        ax2.set_title('Complexity Reduction by Strategy')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Complexity Reduction')
        ax2.tick_params(axis='x', rotation=45)

        # Remove the automatic suptitle added by pandas
        plt.suptitle('')
        plt.tight_layout()
        plt.show()
        '''
        '''
        # Map each strategy to an integer x-position
        strategies = sorted(df['strategy'].unique())
        x_map = {s: i for i, s in enumerate(strategies)}

        # Plot all points for complexity
        plt.figure(figsize=(12, 6))
        for s in strategies:
            y = df.loc[df['strategy'] == s, 'complexity']
            x = np.random.normal(x_map[s], 0.08, size=len(y))  # jitter
            plt.scatter(x, y, alpha=0.6, label=s if s not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.xticks(list(x_map.values()), strategies, rotation=45, ha='right')
        plt.ylabel('Complexity')
        plt.title('All Complexity Values by Strategy')
        plt.tight_layout()
        plt.show()

        # Plot all points for complexity_reduction
        plt.figure(figsize=(12, 6))
        for s in strategies:
            y = df.loc[df['strategy'] == s, 'complexity_reduction']
            x = np.random.normal(x_map[s], 0.08, size=len(y))  # jitter
            plt.scatter(x, y, alpha=0.6, label=s if s not in plt.gca().get_legend_handles_labels()[1] else "")

        plt.xticks(list(x_map.values()), strategies, rotation=45, ha='right')
        plt.ylabel('Complexity Reduction')
        plt.title('All Complexity Reduction Values by Strategy')
        plt.tight_layout()
        plt.show()

        pdb.set_trace()

        '''
        orig_dict = {
            'complexity': final_metrics["complexity"],
            'complexity_reduction': final_metrics["complexity_reduction"], #final_metrics["complexity_reduction"],
            'strategy': ["human leader" if "human" in f else "AI leader" for f in final_metrics["strategy"]], #final_metrics["strategy"],
            "name": final_metrics["name"],
            "layout": layout_list,
            "team_id": [r//4 for r in range(len(final_metrics["strategy"]))],
            "idle_time": final_metrics["idle_time"]



        }
        
        
        df = pd.DataFrame(orig_dict)

        '''
        cr_values = df.loc[:, 'complexity_reduction'].values
        st_values = df.loc[:, 'strategy'].values
        for c_idx in range(0,cr_values.size,4):
            top = cr_values[c_idx:c_idx+4].argmax()
            top_strategy = st_values[c_idx:c_idx+4][top]
            top_layout = layouts[layout_ids[c_idx//4]-1][top]
            print(top, top_strategy, top_layout)




        # Performance according to number of trial
        # Split complexity values and corresponding strategies by index modulo 4
        groups = []
        group_strats = []
        group_name = []
        for k in range(4):
            idxs = np.arange(k, len(df), 4)
            groups.append(df.loc[idxs, 'complexity_reduction'].values)
            group_strats.append(df.loc[idxs, 'strategy'].values)
            group_name.append(df.loc[idxs, 'name'].values)

        labels = ['First', 'Second', 'Third', 'Fourth']
        positions = np.arange(1, 5)

        fig, ax = plt.subplots(figsize=(8, 6))

        # If you also want boxplots, uncomment:
        # bp = ax.boxplot(groups, positions=positions, widths=0.6, patch_artist=True)

        # Get updated y-limits after boxplot to compute offset
        ymin, ymax = ax.get_ylim()
        y_offset = (ymax - ymin) * 0.02  # 2% above the current top

        # Draw one line per experiment across the four positions
        # (i.e., connect item i from each modulo-4 group)
        n_experiments = max(len(g) for g in groups)
        for i in range(n_experiments):
            # Build the y-values for this experiment across First..Fourth
            ys = []
            for g in groups:
                if i < len(g):
                    ys.append(g[i] + y_offset)  # include the same offset used for points
                else:
                    ys.append(np.nan)  # handle ragged tails safely
            # Plot the trajectory line (under points)
            ax.plot(
                positions, ys,
                linewidth=1.0,
                alpha=0.35,
                color='0.4',
                zorder=1
            )
            # If we have a layout_id for this experiment, annotate it at the last non-nan point
            if i < len(layout_ids) and not np.isnan(ys[-1]):
                ax.text(
                    positions[-1] + 0.05, ys[-1],  # small x-offset so it doesn't overlap the point
                    str(layout_ids[i]),
                    va='center', ha='left',
                    fontsize=8, color='black'
                )
                
                
        # Create a color map for strategies
        strategies = df['strategy'].unique()
        cmap = plt.get_cmap('tab10')
        color_map = {s: cmap(i) for i, s in enumerate(strategies)}

        # Overlay points above the boxes, colored by strategy
        for pos, data, strats, names in zip(positions, groups, group_strats, group_name):
            x = np.random.normal(pos, 0.00001, size=len(data)) #x = np.random.normal(pos, 0.08, size=len(data))  # jitter
            y = data + y_offset  # nudge points above the box / baseline
            for xi, yi, strat, name in zip(x, y, strats, names):
                ax.scatter(
                    xi, yi,
                    color=color_map[strat],
                    edgecolor='k', linewidth=0.3,
                    alpha=0.8,
                    zorder=2
                )
                
                if name in together_files_dict.keys():
                    name += " " + together_files_dict[name]
                ax.annotate(name, (xi, yi),
                         xytext=(3, 3), textcoords='offset points', fontsize=8)

        # Build custom legend for strategies
        legend_handles = [
            Line2D(
                [0], [0],
                marker='o', color='w',
                markerfacecolor=color_map[s],
                markeredgecolor='k',
                markersize=8,
                label=s
            )
            for s in strategies
        ]
        ax.legend(handles=legend_handles, title='Strategy',
                  bbox_to_anchor=(1.02, 1), loc='upper left')

        # Formatting
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Complexity Reduction')
        ax.set_title('Complexity Reduction by Trial')

        plt.tight_layout()
        plt.show()
        
        '''
        
        # ===================== CONFIG: add/remove fields here =====================
        # Each key should exist in your top-level `final_metrics` dict (as a list aligned with rows).
        FM_FIELDS = [
            "objects_carried_distribution",
            "objects_carried_distribution_danger",
            "objects_carried_distribution_weight",
            "travel_distance_distribution",
            'orders_parsed',
            'distribution_sensed',
            'carry_order_to_first_carry',
            'num_times_sensed_before_carry'
        ]
        # ========================================================================


        # ------------------------- Formatting helpers ---------------------------
        def fmt_list_of_dicts(lod, max_items=100, max_kv_per_dict=100):
            """
            Compact, readable representation for values that are usually list[dict].
            - Accepts None, a single dict, or list/tuple of dicts or scalars.
            - Truncates long lists and long dicts, appending a (+N more) hint.
            """
            if lod is None:
                return "[]"
            if isinstance(lod, dict):
                lod = [lod]
            if not isinstance(lod, (list, tuple)):
                # Not a list-like; show as-is
                return str(lod)

            if len(lod) == 0:
                return "[]"

            lines = []
            shown_items = 0
            for d in lod[:max_items]:
                if isinstance(d, dict):
                    kvs = list(d.items())[:max_kv_per_dict]
                    inner = ", ".join(f"{str(k)}: {str(v)}" for k, v in kvs)
                    if len(d) > max_kv_per_dict:
                        inner += f", ... (+{len(d) - max_kv_per_dict} more)"
                    lines.append("{ " + inner + " }")
                else:
                    lines.append(str(d))
                shown_items += 1

            if len(lod) > shown_items:
                lines.append(f"... (+{len(lod) - shown_items} more)")
            return "\n".join(lines)


        # --------------------- Optional df builder (your style) -------------------
        def build_df_from_final_metrics(final_metrics, layout_list):
            """
            Mirrors your current df construction so this file is drop-in.
            """
            N = len(final_metrics["strategy"])
            return pd.DataFrame({
                "complexity": final_metrics["complexity"],
                "complexity_reduction": final_metrics["complexity_reduction"],
                "strategy": ["human leader" if "human" in f else "AI leader"
                             for f in final_metrics["strategy"]],
                "name": final_metrics["name"],
                "layout": layout_list,
                "team_id": [r // 4 for r in range(N)],
                "idle_time": final_metrics.get("idle_time", [np.nan] * N),
            })


        # --------------------------- Main plotting fn ----------------------------
        def interactive_plot(df, final_metrics, together_files_dict=None):
            """
            df: pandas DataFrame with columns:
                ['complexity_reduction', 'strategy', 'name', 'layout', 'team_id', ...]
                (built the same way you're already building it)
            final_metrics: dict of lists (top-level), same length N as df rows
            together_files_dict (optional): dict mapping df['name'] -> extra string
            """
            if together_files_dict is None:
                together_files_dict = {}

            N = len(df)
            labels = ['First', 'Second', 'Third', 'Fourth']
            positions = np.arange(1, 5)

            # ---- Bucket rows into 4 session groups (mod 4) ----
            groups, group_strats, group_name, group_layout, group_team_id = [], [], [], [], []
            # Generic store for any number of final_metrics fields:
            # group_fields[field][sess_idx] -> list aligned with that session's points (exp_idx)
            group_fields = {field: [] for field in FM_FIELDS}

            for k in range(4):
                idxs = np.arange(k, N, 4)
                groups.append(df.loc[idxs, 'complexity_reduction'].values)
                group_strats.append(df.loc[idxs, 'strategy'].values)
                group_name.append(df.loc[idxs, 'name'].values)
                group_layout.append(df.loc[idxs, 'layout'].values)
                group_team_id.append(df.loc[idxs, 'team_id'].values)

                # Build per-field lists from top-level final_metrics using the same indices
                for field in FM_FIELDS:
                    arr = final_metrics.get(field, [None] * N)
                    group_fields[field].append([arr[i] if i < len(arr) else None for i in idxs])

            # ---- Colors per strategy ----
            strategies = list(dict.fromkeys(df['strategy'].values))  # preserves first-seen order
            cmap10 = plt.get_cmap('tab10')
            color_map = {s: cmap10(i % 10) for i, s in enumerate(strategies)}
            n_strat = max(1, len(strategies))

            fig, ax = plt.subplots(figsize=(10, 6))

            # ---- y_offset for a tiny nudge above lines ----
            all_vals = np.concatenate([g for g in groups if len(g)]) if any(len(g) for g in groups) else np.array([])
            if all_vals.size:
                ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                y_offset = (ymax - ymin) * 0.01 if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin else 0.01
            else:
                y_offset = 0.01

            # ---- Trajectories across sessions ----
            n_experiments = max((len(g) for g in groups), default=0)
            cmap20 = plt.cm.tab20
            group_colors = [cmap20(j % 20) for j in range(n_experiments)]

            line_artists = []
            point_records = {}  # exp_idx -> list of dicts (per session point metadata)

            for i in range(n_experiments):
                ys = [g[i] + y_offset if i < len(g) else np.nan for g in groups]
                (ln,) = ax.plot(
                    positions, ys,
                    linewidth=1.0, alpha=0.35, color=group_colors[i],
                    zorder=1, picker=5
                )
                ln._exp_idx = i
                line_artists.append(ln)
                point_records[i] = []

            # ---- Points with jitter + metadata ----
            rng = np.random.default_rng(42)
            box_width = 0.7 / n_strat
            jitter_scale = box_width * 0.25

            for sess_idx, pos in enumerate(positions):
                if sess_idx >= len(groups):
                    continue

                data    = groups[sess_idx]
                strats  = group_strats[sess_idx]
                names   = group_name[sess_idx]
                layouts = group_layout[sess_idx]
                teams   = group_team_id[sess_idx]

                # Grab the per-field arrays for this session
                per_field_arrays = {field: group_fields[field][sess_idx] for field in FM_FIELDS}

                for exp_idx, (val, strat, name, ly, ty) in enumerate(zip(data, strats, names, layouts, teams)):
                    xi = pos + rng.uniform(-jitter_scale, jitter_scale)
                    yi = val + y_offset

                    pt = ax.scatter(
                        xi, yi,
                        color=color_map.get(strat, "C0"),
                        edgecolor='k', linewidth=0.3,
                        alpha=0.9, zorder=2,
                        picker=True, pickradius=5
                    )
                    pt._exp_idx  = exp_idx
                    pt._sess_idx = sess_idx

                    # Optional small static label next to each dot
                    name_disp = f"{name} {together_files_dict[name]}" if name in together_files_dict else name
                    ax.annotate(f"{name_disp} {ty} {ly}", (xi, yi),
                                xytext=(3, 3), textcoords='offset points', fontsize=8)

                    # Collect generic field data for this point from per_field_arrays
                    field_data = {field: per_field_arrays[field][exp_idx]
                                  if exp_idx < len(per_field_arrays[field]) else None
                                  for field in FM_FIELDS}

                    point_records[exp_idx].append({
                        "artist": pt,
                        "xy": (xi, yi),
                        "sess_idx": sess_idx,
                        "name": name,
                        "team_id": ty,
                        "layout": ly,
                        "strategy": strat,
                        **field_data,  # merge all `final_metrics` fields
                    })

            # ---- Legend & axes ----
            legend_handles = [
                Line2D([0], [0], marker='s', linestyle='none',
                       markerfacecolor=color_map.get(s, "C0"), markeredgecolor='k',
                       markersize=8, label=s)
                for s in strategies
            ]
            ax.legend(handles=legend_handles, title='Strategy',
                      bbox_to_anchor=(1.02, 1), loc='upper left')

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
            ax.set_ylabel('Complexity Reduction')
            ax.set_title('Complexity Reduction by Trial (Grouped by Strategy)')
            ax.grid(axis='y', alpha=0.25, linestyle='--')

            plt.tight_layout()

            # -------------------- Interaction wiring --------------------
            active_exp_idx = {"val": None}
            active_annots  = []
            just_picked    = {"flag": False}

            def fmt_label(rec):
                """Build the annotation text for one point, using FM_FIELDS."""
                header = f"team_id={rec['team_id']}  layout={rec['layout']}  name={rec['name']}"
                blocks = [header]
                for field in FM_FIELDS:
                    raw_text = fmt_list_of_dicts(rec.get(field))
                    wrapped = "\n".join(textwrap.fill(line, width=50) for line in raw_text.splitlines())
                    blocks.append(field + ":\n" + wrapped)
                return "\n".join(blocks)

            def clear_labels():
                for a in active_annots[:]:
                    a.remove()
                active_annots.clear()
                for ln in line_artists:
                    if ln is not None:
                        ln.set_linewidth(1.0)
                        ln.set_alpha(0.35)
                        ln.set_zorder(1)
                active_exp_idx["val"] = None
                fig.canvas.draw_idle()

            def show_for_exp_idx(exp_idx):
                clear_labels()
                if exp_idx not in point_records:
                    fig.canvas.draw_idle()
                    return
                for rec in point_records[exp_idx]:
                    x, y = rec["xy"]
                    txt = fmt_label(rec)
                    a = ax.annotate(
                        txt, (x, y),
                        xytext=(6, 6),
                        textcoords="offset points",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.95)
                    )
                    active_annots.append(a)
                ln = line_artists[exp_idx]
                if ln is not None:
                    ln.set_linewidth(2.0)
                    ln.set_alpha(0.9)
                    ln.set_zorder(3)
                active_exp_idx["val"] = exp_idx
                fig.canvas.draw_idle()

            def on_pick(event):
                just_picked["flag"] = True
                artist = event.artist
                if hasattr(artist, "_exp_idx"):
                    exp_idx = artist._exp_idx
                    if active_exp_idx["val"] == exp_idx:
                        clear_labels()
                    else:
                        show_for_exp_idx(exp_idx)

            def on_button(event):
                if just_picked["flag"]:
                    just_picked["flag"] = False
                    return
                if event.inaxes == ax:
                    clear_labels()

            fig.canvas.mpl_connect('pick_event', on_pick)
            fig.canvas.mpl_connect('button_press_event', on_button)

            plt.show()


        # ----------------------------- Usage -----------------------------
        # If you already have df built exactly as in your snippet, just do:
        interactive_plot(df, final_metrics, together_files_dict)
                
        '''
        # Compute mean values grouped by strategy
        grouped = df.groupby('strategy').mean()

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        index = list(range(len(grouped)))
        bar_width = 0.35


        ax.bar([i - bar_width/2 for i in index], grouped['complexity'], width=bar_width, label='Avg Complexity')
        ax.bar([i + bar_width/2 for i in index], grouped['complexity_reduction'], width=bar_width, label='Avg Complexity Reduction')


        ax.set_xticks(index)
        ax.set_xticklabels(grouped.index, rotation=45, ha='right')
        ax.set_ylabel('Mean Value')
        ax.set_title('Mean Complexity and Complexity Reduction by Strategy')
        ax.legend()

        plt.tight_layout()
        plt.show()
        '''
        
        
        #Layout performance
        # ---- Flatten (you already have these four lists) ----
        x_vals, y_vals, strat_vals, name_vals, session_vals = [], [], [], [], []
        for x in sorted(group_by_layout[0].keys()):
            ys = group_by_layout[0][x]
            ss = group_by_layout[1][x]
            nn = group_by_layout[2][x]
            x_vals.extend([x] * len(ys))
            y_vals.extend(ys)
            strat_vals.extend(ss)
            name_vals.extend(nn)
            session_vals.extend(group_by_layout[3][x])

        # ---- Strategy palette (consistent order) ----
        unique_strats = list(dict.fromkeys(strat_vals))  # preserves first-seen order
        cmap = plt.get_cmap('tab10')
        color_map = {s: cmap(i % 10) for i, s in enumerate(unique_strats)}
        n_strat = len(unique_strats)

        # ---- Layout positions on x-axis ----
        layouts_sorted = sorted(group_by_layout[0].keys())
        positions = np.array(layouts_sorted, dtype=float)  # typically [1,2,3,4]

        # ---- Grouped box geometry ----
        group_width = 0.7
        box_width   = group_width / max(1, n_strat)
        offsets     = [(i - (n_strat - 1) / 2) * box_width for i in range(n_strat)]

        fig, ax = plt.subplots(figsize=(9, 6))

        # ---- Build per-layout, per-strategy data & draw boxplots ----
        for lx, pos in zip(layouts_sorted, positions):
            # Bucket values by strategy for this layout
            by_strat = {s: [] for s in unique_strats}
            vals     = group_by_layout[0][lx]
            strats   = group_by_layout[1][lx]
            for v, s in zip(vals, strats):
                by_strat[s].append(v)

            data_list, pos_list, colors_list = [], [], []
            for si, s in enumerate(unique_strats):
                if len(by_strat[s]) == 0:
                    continue
                data_list.append(by_strat[s])
                pos_list.append(pos + offsets[si])
                colors_list.append(color_map[s])

            if data_list:
                bp = ax.boxplot(
                    data_list,
                    positions=pos_list,
                    widths=box_width * 0.8,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops=dict(linewidth=1.2, color='k'),
                    whiskerprops=dict(linewidth=1.0, color='0.2'),
                    capprops=dict(linewidth=1.0, color='0.2'),
                    flierprops=dict(marker='o', markersize=3, markeredgecolor='0.3', alpha=0.5)
                )
                for patch, fc in zip(bp['boxes'], colors_list):
                    patch.set_facecolor(fc)
                    patch.set_edgecolor('k')
                    patch.set_alpha(0.35)

        # ---- Small vertical offset so points sit nicely above box centers ----
        ymin, ymax = ax.get_ylim()
        y_offset = (ymax - ymin) * 0.01

        # ---- Overlay points, jittered within each strategy slot, with annotations ----
        rng = np.random.default_rng(42)      # stable jitter; change/remove seed if desired
        jitter_scale = box_width * 0.25

        for lx, pos in zip(layouts_sorted, positions):
            vals   = group_by_layout[0][lx]
            strats = group_by_layout[1][lx]
            names  = group_by_layout[2][lx]
            sessions = group_by_layout[3][lx]

            for v, s, nm, sm in zip(vals, strats, names, sessions):
                si = unique_strats.index(s)
                cx = pos + offsets[si]
                xi = cx + rng.uniform(-jitter_scale, jitter_scale)
                yi = v + y_offset

                ax.scatter(
                    xi, yi,
                    color=color_map[s], edgecolor='k', linewidth=0.3,
                    alpha=0.9, zorder=2
                )

                if nm in together_files_dict:
                    nm = nm + " " + together_files_dict[nm]
                ax.annotate(sm, (xi, yi), xytext=(3, 3),
                            textcoords='offset points', fontsize=8)

        # ---- Legend (strategies) ----
        handles = [
            Line2D([0], [0], marker='s', linestyle='none',
                   markerfacecolor=color_map[s], markeredgecolor='k',
                   markersize=8, label=s)
            for s in unique_strats
        ]
        ax.legend(handles=handles, title='Strategy',
                  bbox_to_anchor=(1.02, 1), loc='upper left')

        # ---- Axes cosmetics ----
        ax.set_xticks(positions)
        ax.set_xticklabels([str(int(p)) for p in positions])
        ax.set_xlabel('Layout (1 to 4)')
        ax.set_ylabel('Result Value')
        ax.set_title('Results per Layout with Strategy-Grouped Boxplots')
        ax.set_xlim(positions[0] - 0.6, positions[-1] + 0.6)
        ax.grid(axis='y', alpha=0.25, linestyle='--')

        plt.tight_layout()
        plt.show()

        
        '''
        orig_dict = {
            'strategy': final_metrics["strategy"]
        }
        orig_dict.update(final_average_survey_results)
        df = pd.DataFrame(orig_dict)
        
        
        for kk in orig_dict.keys():
            if kk == "strategy":
                continue
            #Survey according to number of trial
            # Split complexity values and corresponding strategies by index modulo 4
            groups = []
            group_strats = []
            for k in range(4):
                idxs = np.arange(k, len(df), 4)
                groups.append(df.loc[idxs, kk].values)
                group_strats.append(df.loc[idxs, 'strategy'].values)

            labels = [
                'First',
                'Second',
                'Third',
                'Fourth'
            ]
            positions = np.arange(1, 5)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Draw boxplots
            #bp = ax.boxplot(groups, positions=positions, widths=0.6, patch_artist=True)

            # Get updated y-limits after boxplot to compute offset
            ymin, ymax = ax.get_ylim()
            y_offset = (ymax - ymin) * 0.02  # 2% above the current top

            # Create a color map for strategies
            strategies = df['strategy'].unique()
            cmap = plt.get_cmap('tab10')
            color_map = {s: cmap(i) for i, s in enumerate(strategies)}

            # Overlay points above the boxes, colored by strategy
            for pos, data, strats in zip(positions, groups, group_strats):
                x = np.random.normal(pos, 0.08, size=len(data))  # jitter
                y = data + y_offset  # nudge points above the box
                for xi, yi, strat in zip(x, y, strats):
                    ax.scatter(xi, yi,
                               color=color_map[strat],
                               edgecolor='k', linewidth=0.3,
                               alpha=0.8)

            # Build custom legend for strategies
            legend_handles = [
                Line2D([0], [0],
                       marker='o', color='w',
                       markerfacecolor=color_map[s],
                       markeredgecolor='k',
                       markersize=8,
                       label=s)
                for s in strategies
            ]
            ax.legend(handles=legend_handles, title='Strategy',
                      bbox_to_anchor=(1.02, 1), loc='upper left')

            # Formatting
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(kk)
            ax.set_title(kk)

            plt.tight_layout()
            plt.show()
        '''
    pass

'''
weight_distribution_carried = final_metrics["weight_distribution_carried"]
# Determine all unique weight values across all legends
# 1. Find all unique weight categories
unique_weights = sorted({w for sub in weight_distribution_carried for w in sub})

# 2. Compute frequency of each weight for each legend
freqs_by_legend = []
for w_list in weight_distribution_carried:
    cnt = Counter(w_list)
    freqs_by_legend.append([cnt[w] for w in unique_weights])

# 3. Plot grouped bar chart
n_leg = len(legends)
x = np.arange(len(unique_weights))
bar_w = 0.8 / n_leg

plt.figure(figsize=(10, 6))
for i, freqs in enumerate(freqs_by_legend):
    plt.bar(x + i*bar_w, freqs, width=bar_w, label=legends[i])

plt.xticks(x + bar_w*(n_leg-1)/2, unique_weights, rotation=45, ha='right')
plt.xlabel("Weight")
plt.ylabel("Count")
plt.title("Weight Distribution Carried by Agent")
plt.legend(title="Agent")
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
positions = np.arange(len(legends))
weight_distribution_carried = final_metrics["weight_distribution_carried"]
# 1. Violin plots (distribution shapes)
parts = ax.violinplot(
    weight_distribution_carried,
    positions=positions,
    showextrema=False,
    widths=0.7
)
for body in parts['bodies']:
    body.set_alpha(0.5)


# 3. Jittered scatter of individual data points
for i, data in enumerate(weight_distribution_carried):
    jittered_x = np.random.normal(positions[i], 0.04, size=len(data))
    ax.scatter(jittered_x, data, alpha=0.7, color='black', s=20)

# Formatting
ax.set_xticks(positions)
ax.set_xticklabels(legends, rotation=45, ha='right')
ax.set_xlabel("Scenario")
ax.set_ylabel("Carried Weight")
ax.set_title("Carried Weights per Scenario")

plt.tight_layout()
plt.show()
'''
