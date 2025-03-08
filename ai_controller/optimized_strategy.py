import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pdb
import itertools



class DynamicSensorPlanner:
    def __init__(self, agents, objects, locations, agent_home, PD_PB_params, object_weights, regions_to_explore, occMap, goal_coords):
        self.agents = agents
        self.objects = objects
        self.PD_PB = PD_PB_params  # Dict: {agent: (PD, PB)}
        self.sensed_objects = []
        self.sensed_clusters = []
        self.penalty_weight = 1000
        self.prior_belief = 0.5
        self.pickup_belief_threshold = 0.91
        self.objects_to_carry = []
        self.object_weights = object_weights
        self.regions_to_explore = regions_to_explore
        self.occMap = occMap
        self.past_beliefs = []
        self.goal_coords = goal_coords
        self.already_carried = []
        
        locations = self.recluster(locations)
        
        self.nodes = self.create_nodes()
        
        # Initialize beliefs (uniform prior)
        self.belief = {j: self.prior_belief for j in self.objects}
        
        self.tau_initial = 2.0
        
        self.tau_current = {j: self.tau_initial for j in self.objects}
        
        self.group_belief = {str(label): self.prior_belief for label in self.clusters}  # Initial belief
        self.group_tau = {str(label): self.tau_initial * len(cluster["objects"]) for label, cluster in self.clusters.items()}  # Threshold scaled by group size
        
        # Precompute LLR contributions
        self.LLR = {}
        
        for i in self.agents:
            PD, PB = self.PD_PB[i]
            self.LLR[i] = {
                1: np.log(PD / (1 - PB)),  # LLR for Y=1
                0: np.log((1 - PD) / PB)    # LLR for Y=0
            }
         
        self.update_positions(agent_home,locations)
        
    
    def create_nodes(self):
        #return ['HOME'] + [a + o for a in ['sense_', 'carry_', 'SAFE'] for o in self.objects] + ['REGION' + str(r) for r in self.regions_to_explore]
        #return ['HOME'] + ['sense_' + o for o in self.objects] + [a + o for a in ['carry_', 'SAFE'] for o in self.objects if o in self.objects_to_carry] + ['REGION' + str(r) for r in self.regions_to_explore]
        return ['HOME'] + ['sense_' + 'CLUSTER' + str(c_key) for c_key in self.clusters.keys()] + [a + o for a in ['carry_', 'SAFE'] for o in self.objects if o in self.objects_to_carry] + ['REGION' + str(r) for r in self.regions_to_explore]
    
    def euclidean(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        
    def calculateHValue(self,current,dest):

        dx = abs(current[0] - dest[0])
        dy = abs(current[1] - dest[1])
     
  
        h = dx + dy #For only four movements

        return h   
        
    def tracePath(self,node_details,dest):
        path = []
        
        currentNode = dest

        while node_details[currentNode[0]][currentNode[1]]["parent"][0] != currentNode[0] or node_details[currentNode[0]][currentNode[1]]["parent"][1] != currentNode[1]:
            path.append(currentNode)
            currentNode = node_details[currentNode[0]][currentNode[1]]["parent"]
            
        path.reverse()
        
        return path

    def findPath(self,startNode,endNode,occMap):

        
        openSet = [startNode]
        closedSet = []
        
        highest_cost = float('inf') #2147483647
        
        node_details = {}
        
        for s0 in range(occMap.shape[0]):
            node_details[s0] = {}
            for s1 in range(occMap.shape[1]):
                if s0 == startNode[0] and s1 == startNode[1]:
                    node_details[s0][s1] = {"f":0, "g":0, "h":0, "parent":[startNode[0],startNode[1]]}
                else:
                    node_details[s0][s1] = {"f":highest_cost, "g":highest_cost, "h":highest_cost, "parent":[-1,-1]}
        


        
        next_nodes = np.array([[-1,0],[1,0],[0,1],[0,-1]])

        while openSet:
        
            currentNode = openSet.pop(0)
            closedSet.append(tuple(currentNode))
            
     
                
            for nx in next_nodes:
                neighborNode = currentNode + nx
                
                if neighborNode[0] == endNode[0] and neighborNode[1] == endNode[1]:
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    return self.tracePath(node_details, endNode)
                
                if min(neighborNode) == -1 or any(neighborNode >= occMap.shape) or not (occMap[neighborNode[0],neighborNode[1]] == 0 or occMap[neighborNode[0],neighborNode[1]] == 3 or occMap[neighborNode[0],neighborNode[1]] == -2) or tuple(neighborNode) in closedSet: #modified to allow a robot to step into another robot's place
                    continue

            
                gNew = node_details[currentNode[0]][currentNode[1]]["g"] + 1
                hNew = self.calculateHValue(neighborNode,endNode)
                fNew = gNew + hNew
                
                if node_details[neighborNode[0]][neighborNode[1]]["f"] == highest_cost or node_details[neighborNode[0]][neighborNode[1]]["f"] > fNew:
                    openSet.append(neighborNode)
                    
                    node_details[neighborNode[0]][neighborNode[1]]["f"] = fNew
                    node_details[neighborNode[0]][neighborNode[1]]["g"] = gNew
                    node_details[neighborNode[0]][neighborNode[1]]["h"] = hNew
                    node_details[neighborNode[0]][neighborNode[1]]["parent"] = currentNode
                    

        return [] #No path
        
    def update_positions(self, current_positions,locations):
        # Distance from HOME to objects (for each agent)
        
        self.d_home = {}
        for i in self.agents: 
            for j in self.nodes:
                if j != 'HOME':
                    if '_' in j:
                        n = j.split('_')[1]
                    elif 'SAFE' in j:
                        n = 'SAFE'
                    else:
                        n = j
                    try:
                        #print(i,j,n,current_positions[i],locations[n])
                        
                        if tuple(current_positions[i]) == tuple(locations[n]):
                            self.d_home[(i,j)] = 0
                        else:
                            self.d_home[(i,j)] = len(self.findPath(np.array(current_positions[i]),np.array(locations[n]),self.occMap)) #self.euclidean(current_positions[i], locations[n])    
                    except:
                        pdb.set_trace()
                
        
        self.c_jk = {}
        for j in self.nodes:
        
            if '_' in j:
                n1 = j.split('_')[1]
            elif 'SAFE' in j:
                n1 = 'SAFE'
            else:
                n1 = j
        
            for k in self.nodes:
                if j != k and not (j == 'HOME' or k == 'HOME'):
                
                    if '_' in k:
                        n2 = k.split('_')[1]
                    elif 'SAFE' in k:
                        n2 = 'SAFE'
                    else:
                        n2 = k
                
                    #print(j,k,n1,n2,locations[n1],locations[n2])
                    
                    if tuple(locations[n1]) == tuple(locations[n2]):
                        self.c_jk[(j, k)] = 0
                    elif (k, j) in self.c_jk.keys():
                        self.c_jk[(j, k)] = self.c_jk[(k, j)]
                    else:
                        self.c_jk[(j, k)] = len(self.findPath(np.array(locations[n1]),np.array(locations[n2]),self.occMap)) #self.euclidean(locations[n1], locations[n2]) 
                        
        #print(self.d_home,self.c_jk)
                 
    def cluster_objects(self, object_positions, eps=5.0):
    
        coords = np.array([pos for pos in object_positions.values()])
        clustering = DBSCAN(eps=eps,min_samples=1).fit(coords)
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = {"objects": [], "centroid": None}
            clusters[label]["objects"].append(list(object_positions.keys())[idx])
        # Compute centroids
        for label in clusters:
            cluster_coords = coords[[i for i, l in enumerate(clustering.labels_) if l == label]]
            clusters[label]["centroid"] = np.mean(cluster_coords, axis=0)
            
            
        return clusters
        
    def recluster(self, locations):
        self.clusters = self.cluster_objects({l:locations[l] for l in locations.keys() if 'CLUSTER' not in l and 'SAFE' not in l and 'REGION' not in l})
    
        print(self.clusters)
    
        for c in self.clusters.keys():
            locations['CLUSTER' + str(c)] = self.clusters[c]['centroid']
        
        return locations
                 
    def add_objects(self, new_objects, weights):
    
        for j in new_objects:
            self.belief[j] = self.prior_belief
            self.tau_current[j] = self.tau_initial  # Initial confidence threshold
            self.objects.append(j)
        
        self.nodes = self.create_nodes()
        self.object_weights.update(weights)
        
    def update_belief(self, object_j, agent_i, report_Y):
        PD, PB = self.PD_PB[agent_i]
        prior = self.belief[object_j]
        
        # Bayesian update
        if report_Y == 1:
            likelihood_danger = PD
            likelihood_benign = 1 - PB
        else:
            likelihood_danger = 1 - PD
            likelihood_benign = PB
        
        posterior = (likelihood_danger * prior) / (
            likelihood_danger * prior + likelihood_benign * (1 - prior)
        )
        self.belief[object_j] = posterior
        
        # Update remaining confidence threshold
        self.tau_current[object_j] -= self.LLR[agent_i][report_Y]
        self.tau_current[object_j] = max(0, self.tau_current[object_j])  # Threshold ≥ 0
        self.sensed_objects.append((agent_i,object_j))
        
        if self.belief[object_j] >= self.pickup_belief_threshold and object_j not in self.objects_to_carry:
            self.objects_to_carry.append(object_j)
            
        #CLUSTERS
            
        cluster_label = ""
        for c in self.clusters.keys():
            if object_j in self.clusters[c]["objects"]:
                cluster_label = c
                break
            
        objects = self.clusters[cluster_label]["objects"]
        total_belief = sum(self.belief[o] for o in objects)
        self.group_belief[str(cluster_label)] = total_belief / len(objects)
        
        self.group_tau[str(cluster_label)] = sum(self.tau_current[o] for o in objects)

    def update_state(self, object_beliefs, object_weights, current_positions, object_locations, regions_to_explore, occMap):
        
        for ob in object_beliefs:
            if ob not in self.past_beliefs:
                self.update_belief(ob[0], ob[1], ob[2])
                self.past_beliefs.append(ob)
            
        self.occMap = occMap
        
        ####### Cluster
        self.sensed_clusters = []
        for a in self.agents:
            for c in self.clusters.keys():
                if all((a,ob) in self.sensed_objects for ob in self.clusters[c]["objects"] for ss in self.sensed_objects if ss[0] == a):
                    self.sensed_clusters.append((a,str(c)))
        #######
        
        for ol_key in object_locations.keys():
            if ol_key != 'SAFE' and ol_key not in self.already_carried and tuple(object_locations[ol_key]) in self.goal_coords:
                self.already_carried.append(ol_key)
                self.objects.remove(ol_key)
            
        self.nodes = self.create_nodes()
            
        self.update_positions(current_positions, object_locations)
        
        self.regions_to_explore = regions_to_explore

        
        
        self.object_weights = object_weights

    def replan(self):
    
        model = gp.Model("FixedRoutingAssignment")
        M = len(self.nodes) + 1  # Upper bound for order variables
        
        # ================== Decision Variables ==================
        assign = model.addVars([(i, j) for i in self.agents for j in self.nodes if not (i == 'HOME' or j == 'HOME')], vtype=GRB.BINARY, name="Assign")

        # Travel variables now include HOME as a node
        travel = model.addVars(
            [(i, j, k) for i in self.agents for j in self.nodes for k in self.nodes if j != k],
            vtype=GRB.BINARY, name="Travel"
        )

        # Order variables for objects (HOME is order 0)
        order = model.addVars(
            [(i, j) for i in self.agents for j in self.nodes if j != 'HOME'], 
            vtype=GRB.INTEGER, lb=0, ub=M, name="Order"
        )
        
        sync_order = model.addVars([j for j in self.objects if j in self.objects_to_carry and self.object_weights[j] > 1], vtype=GRB.INTEGER, name="Sync_Carry")
        
        
        max_dist = model.addVars(
            [j for j in self.objects if self.object_weights[j] > 1], 
            lb=0, 
            name="MaxDist"
        )
        

        # ================== Objective Function ==================
        # Total time = Inspection time (HOME to first object) + Travel between objects
        inspection_time = gp.quicksum(
            travel[i, 'HOME', j] * self.d_home[i,j] 
            for i in self.agents for j in self.nodes if j != 'HOME' and not (j.startswith("carry_") and self.object_weights.get(j.split("_")[1], 1) == 1) # if (i,j) not in self.sensed_objects
        )

        travel_time = gp.quicksum(
            travel[i,j,k] * self.c_jk.get((j,k), 0) 
            for i in self.agents for j in self.nodes for k in self.nodes 
            if j != k and (j != 'HOME' or k != 'HOME') and not (j.startswith("carry_") and self.object_weights.get(j.split("_")[1], 1) == 1) #and (i,j) not in self.sensed_objects
        )

        '''
        uncertainty_penalty = gp.quicksum(
            (self.tau_current[j.split('_')[1]] - gp.quicksum(assign[i,j] * self.LLR[i][1] for i in self.agents)) * self.belief[j.split('_')[1]]
            for j in self.nodes if 'sense' in j
        )
        
        slack = model.addVars(self.objects, lb=0, name="Slack")  # Shortfall per object
        
        confidence_penalty = gp.quicksum(slack[j] * self.penalty_weight for j in self.objects)
        '''
        
        uncertainty_penalty = gp.quicksum(
            (self.group_tau[j.split('CLUSTER')[1]] - gp.quicksum(assign[i,j] * self.LLR[i][1] for i in self.agents)) * self.group_belief[j.split('CLUSTER')[1]]
            for j in self.nodes if 'sense' in j
        )
        
        slack = model.addVars([str(c) for c in self.clusters.keys()], lb=0, name="Slack")  # Shortfall per object
        
        confidence_penalty = gp.quicksum(slack[str(j)] * self.penalty_weight for j in self.clusters)
        
        # Collaborative travel time (max distance per object)
        collaborative_travel = gp.quicksum(
            max_dist[j] for j in self.objects if self.object_weights[j] > 1
        )
        
        exploration_reward = gp.quicksum(
            assign[i, 'REGION'+str(r)] * 100  # Weighted by probability
            for i in self.agents 
            for r in self.regions_to_explore 
        )

        

        model.setObjective(inspection_time + travel_time + uncertainty_penalty + confidence_penalty + collaborative_travel - exploration_reward, GRB.MINIMIZE)

        # ================== Constraints ==================
        # Confidence requirement (unchanged)
        '''
        for j in self.objects:
            model.addConstr(
                gp.quicksum(assign[i,j] * self.agent_info[i]['LLR'] for i in self.agents) >= self.tau,
                f"Confidence_{j}"
            )
        '''

  
                    
            
                
        
        for j in self.nodes:
        
            if 'sense' in j:
            
                '''
                model.addConstr(
                    gp.quicksum(assign[i,j] * self.LLR[i][1]  # Assume worst case (Y=1)
                               for i in self.agents) + slack[j.split('_')[1]] >= self.tau_current[j.split('_')[1]],
                    f"Confidence_{j}"
                )
                
                for i in self.agents:
                    if (i,j.split('_')[1]) in self.sensed_objects:
                        model.addConstr(
                            assign[i,j] == 0,
                            f"already_sensed_{i}_{j}"
                        )
                '''
                model.addConstr(
                    gp.quicksum(assign[i,j] * self.LLR[i][1]  # Assume worst case (Y=1)
                               for i in self.agents) + slack[j.split('CLUSTER')[1]] >= self.group_tau[j.split('CLUSTER')[1]],
                    f"Confidence_{j}"
                )
                
                for i in self.agents:
                    if (i,j.split('CLUSTER')[1]) in self.sensed_clusters:
                        model.addConstr(
                            assign[i,j] == 0,
                            f"already_sensed_{i}_{j}"
                        )
                        
            elif 'carry' in j:
                
                object_id = j.split('_')[1]
                
                
                if object_id in self.objects_to_carry:
                    model.addConstr(
                        gp.quicksum(assign[i,j] for i in self.agents) == self.object_weights[object_id], f"Carry_{j}"
                    )
                    
                    
                    for i in self.agents:
                    
                        if self.object_weights[object_id] > 1:
                            model.addConstr(
                                order[i,j] == sync_order[object_id], f"Force_Sync_Carry_{j}_{i}"
                            )
                            
                            # Distance from agent's current position to object j
                            d_to_j = self.d_home.get((i, f"carry_{object_id}"), 0)
                            # Distance from object j to safe zone
                            d_to_safe = self.c_jk.get((f"carry_{object_id}", f"SAFE{object_id}"), 0)
                            total_d = d_to_j + d_to_safe
                            
                            # If agent i carries j, max_dist[j] >= their total distance
                            model.addConstr(
                                (assign[i, f"carry_{object_id}"] == 1) >> (max_dist[object_id] >= total_d),
                                f"MaxDist_{j}_{i}"
                            )
                        
                        model.addConstr(
                            assign[i,j] == assign[i,'SAFE'+object_id], f"Force_SAFE_{j}_{i}"
                        )
                        model.addConstr(
                            order[i,'SAFE'+object_id] == order[i,j]+1, f"Force_SAFE_Order_{j}_{i}"
                        )
                        
                        
                    
                else:
                    model.addConstr(
                        gp.quicksum(assign[i,j] for i in self.agents) == 0, f"NotCarry_{j}"
                    )
                    
            
            elif 'REGION' in j:
                model.addConstr(
                    gp.quicksum(assign[i,j] for i in self.agents) <= 1, f"Explore_{j}_{i}"
                )

        # Linking assignment to travel
        for i in self.agents:
            for j in self.nodes:
                # If assigned to j, there must be incoming or outgoing travel
                if j == 'HOME':
                    continue
                    
                model.addConstr(
                    assign[i,j] == gp.quicksum(travel[i,k,j] for k in self.nodes if k != j and not ("SAFE" in j and "SAFE" in k)),
                    f"AssignTravelIn_{i}_{j}"
                )
                model.addConstr(
                    assign[i,j] == gp.quicksum(travel[i,j,k] for k in self.nodes if k != j and not ("SAFE" in j and "SAFE" in k)),
                    f"AssignTravelOut_{i}_{j}"
                )

        # MTZ Subtour Elimination (revised)
        for i in self.agents:
            # Order starts at 0 for HOME
            for j in self.nodes:
            
                if j == 'HOME':
                    continue
            
                # If traveling from HOME to j, order[j] = 1
                model.addConstr(
                    (travel[i, 'HOME', j] == 1) >> (order[i,j] == 1),
                    f"MTZ_HomeStart_{i}_{j}"
                )
                for k in self.nodes:
                    if j != k and k != 'HOME':
                        # If traveling j->k, order[k] >= order[j] + 1
                        model.addConstr(
                            order[i,k] >= order[i,j] + 1 - M*(1 - travel[i,j,k]),
                            f"MTZ_Order_{i}_{j}_{k}"
                        )
                        
                    

        # Flow conservation (include HOME)
        for i in self.agents:
            # Depart HOME at most once
            model.addConstr(
                gp.quicksum(travel[i, 'HOME', j] for j in self.nodes if j != 'HOME') <= 1,
                f"DepartHome_{i}"
            )
            # Return to HOME not required (optional)
            for j in self.nodes:
                if j != 'HOME':
                    arrivals = gp.quicksum(travel[i,k,j] for k in self.nodes if k != j)
                    departures = gp.quicksum(travel[i,j,k] for k in self.nodes if k != j)
                    model.addConstr(arrivals == departures, f"Flow_{i}_{j}")

        # ================== Solve & Results ==================
        model.Params.TimeLimit = 30
        model.optimize()

        # Print routes with order
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            print(f"Total distance: {model.ObjVal:.2f}")
            print(self.LLR)
            print(max_dist,sync_order)
            
            print(self.objects_to_carry, self.object_weights)

            #for o in self.objects:
            #    print("Object", o, slack[o],self.tau_current[o],self.belief[o]) #, [assign[i,f"sense_{o}"] for i in self.agents],[assign[i,f"carry_{o}"] for i in self.agents])
            
            assignments = []
            
            for i in self.agents:
                assigned = [(j, int(order[i,j].X)) for j in self.nodes if j != 'HOME' and assign[i,j].X > 0.5]
                if assigned:
                    sorted_assigned = sorted(assigned, key=lambda x: x[1])
                    assignments.append((i,sorted_assigned))
                    route = [f"Home -> {sorted_assigned[0][0]} (Order {sorted_assigned[0][1]})"]
                    for idx in range(1, len(sorted_assigned)):
                        prev_obj = sorted_assigned[idx-1][0]
                        curr_obj = sorted_assigned[idx][0]
                        route.append(f"-> {curr_obj} (Order {sorted_assigned[idx][1]})")
                    print(f"Agent {i} route: {' '.join(route)}")
                    
            return assignments
        else:
            print("No solution found", model.status)
        
        return []
# ================== Data Setup ==================

if __name__ == "__main__":
    agents = [1, 2, 3, 4]
    #objects = ['A', 'B', 'C']
    objects = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']
    # Agents' initial positions  
    agent_home = {1: (0, 0), 2: (10, 5), 3: (5, 10), 4:(2,2)}
    #Object coordinates and SAFE area
    #locations = {'A': (2, 3), 'B': (8, 4), 'C': (6, 9), 'SAFE':(10,10)}
    locations = {
        'A': (3, 14), 'B': (18, 7), 'C': (5, 19), 'D': (12, 2), 'E': (9, 15), 'F': (1, 8), 'G': (16, 11), 'H': (7, 4), 'I': (14, 18), 'J': (10, 6), 'K': (19, 3), 'L': (4, 12), 'M': (8, 17), 'N': (15, 9), 'O': (2, 5),
        'P': (11, 20), 'Q': (6, 10), 'R': (13, 1), 'S': (17, 13), 'T': (20, 16), 'SAFE':(10,10), 'REGION1': (3,4), 'REGION2':(6,6), 'REGION3':(9,0)
    }
    #Object weights
    #object_weights = {'A':3,'B':3,'C':1}
    object_weights = {
        'A': 3, 'B': 3, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1, 'I': 1, 'J': 1,
        'K': 1, 'L': 1, 'M': 1, 'N': 1, 'O': 1, 'P': 1, 'Q': 1, 'R': 1, 'S': 1, 'T': 1
    }
    # Sensor parameters for each agent
    PD_PB = {1: (0.9, 0.8), 2: (0.9, 0.8), 3: (0.6, 0.7), 4: (0.6, 0.7)}

    #Regions
    regions_to_explore = [1,2,3]

    planner = DynamicSensorPlanner(agents, objects, locations, agent_home, PD_PB,object_weights, regions_to_explore)
    plan = planner.replan()

    print(planner.tau_current,planner.belief)
    #planner.update_belief('B', 1, 1)
    #planner.update_belief('B', 2, 1)
    #planner.update_belief('C', 3, 0)
    #planner.update_belief('A', 1, 1)
    #planner.update_belief('A', 2, 1)
    #print(planner.tau_current,planner.belief)
    current_positions = {1: locations['A'], 2: locations['B'], 3: locations['C'], 4: locations['C']}
    #planner.update_positions(current_positions,locations)
    planner.update_state([('B', 1, 1), ('B', 2, 1), ('C', 3, 0), ('A', 1, 1), ('A', 2, 1)], current_positions, locations, regions_to_explore)
    #print(planner.tau_current, planner.sensed_objects)
    plan = planner.replan()


