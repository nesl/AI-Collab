import os
import json
import argparse
import statistics
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='scores', help='Directory with results')

args = parser.parse_args()


scores_directory = args.dir + "/"

#scores_files = os.listdir(scores_directory)

scores_files = glob.glob(scores_directory + "*.txt") #2023_09_14_20_17

timeout = 1200

total_team_achievement = []
total_distance_traveled = []
total_sensor_activations = []
total_productivity = []
total_dangerous_collected = []
total_dangerous_fraction = []
total_droppings = []
bad_files = []
max_files = 40

good_files = 0
alerted = 0

for log in scores_files:

	log_file = open(log)
	
	json_dict = json.load(log_file)
	
	team_quality_work = json_dict["results"][0]["team_quality_work"]
	
	#max(0,(number_dangerous_objects_in_goal - (len(all_magnebots[idx].stats.objects_in_goal) - number_dangerous_objects_in_goal) - all_magnebots[idx].stats.dropped_outside_goal)/len(self.dangerous_objects))
	team_end_time = round(json_dict["results"][0]["team_end_time"])
	
	team_speed_of_work = timeout/(max(timeout/10, min(team_end_time,timeout)))
	
	team_achievement = team_quality_work*team_speed_of_work
	
	
	
	avg_sensor_activation = 0
	avg_distance_traveled = 0
	
	dangerous_objects = []
	non_dangerous_objects = []
	dropping = 0
	
	alert = False
	for robot in json_dict["results"]:
		avg_sensor_activation += robot["sensor_activation"]
		avg_distance_traveled += robot["distance_traveled"]
		dropping += robot["dropped_outside_goal"]
		
		for ob in robot["dangerous_objects_in_goal"]:
			if ob not in dangerous_objects:
				dangerous_objects.append(ob)
			else:
				alert = True
				break
		if alert:
			break
			
		for ob in robot["objects_in_goal"]:
			if ob not in non_dangerous_objects:
				non_dangerous_objects.append(ob)
			
	if alert:
		alerted += 1
		bad_files.append(log)
		continue

	
	if len(dangerous_objects)+len(non_dangerous_objects) == 0:
		bad_files.append(log)
		continue
	
	good_files += 1
	total_team_achievement.append(team_achievement)
	total_distance_traveled.append(avg_distance_traveled)
	total_sensor_activations.append(avg_sensor_activation)
	total_dangerous_collected.append(len(dangerous_objects)/json_dict["results"][0]["total_dangerous_objects"])
	total_productivity.append(len(dangerous_objects)/avg_distance_traveled)
	total_dangerous_fraction.append(len(dangerous_objects)/(len(dangerous_objects)+len(non_dangerous_objects)))
	total_droppings.append(dropping)
	avg_sensor_activation /= len(json_dict["results"])
	avg_distance_traveled /= len(json_dict["results"])
	
	
	
	print("File:", log, "Team Quality of Work:", team_quality_work, "Team Speed of Work:", team_speed_of_work, "Team Achievement:", team_achievement, "Average Sensor Activation:", avg_sensor_activation, "Average Distance Traveled:", str(avg_distance_traveled) + " m")
	
	if good_files == max_files:
		break
	

print("Average Team Achievement:", sum(total_team_achievement)/len(total_team_achievement), "SD Team Achievement:", statistics.stdev(total_team_achievement))
print("Average Distance Traveled:", sum(total_distance_traveled)/len(total_distance_traveled), "SD Distance Traveled:", statistics.stdev(total_distance_traveled))
print("Average Sensor Activations:",sum(total_sensor_activations)/len(total_sensor_activations), "SD Sensor Activations:", statistics.stdev(total_sensor_activations))
print("Productivity:", sum(total_productivity)/len(total_productivity))
print("Fraction of Dangerous Objects Collected of Dangerous Total:", sum(total_dangerous_collected)/len(total_dangerous_collected), "SD:", statistics.stdev(total_dangerous_collected))
print("Fraction of Dangerous Objects Collected of Total:", sum(total_dangerous_fraction)/len(total_dangerous_fraction), "SD:", statistics.stdev(total_dangerous_fraction))
print("Objects dropped outside goal:", sum(total_droppings)/len(total_droppings))
print("ALERTS:", alerted)
print("Good files:", good_files)
print("Bad files:", bad_files)
