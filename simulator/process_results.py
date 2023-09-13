import os
import json

scores_directory = "scores" + "/"

scores_files = os.listdir(scores_directory)

timeout = 1200

avg_team_achievement = 0

for log in scores_files:

	log_file = open(scores_directory + log)
	
	json_dict = json.load(log_file)
	
	team_quality_work = json_dict["results"][0]["team_quality_work"]
	team_end_time = round(json_dict["results"][0]["team_end_time"])
	
	team_speed_of_work = timeout/(max(timeout/10, min(team_end_time,timeout)))
	
	team_achievement = team_quality_work*team_speed_of_work
	avg_team_achievement += team_achievement
	
	
	avg_sensor_activation = 0
	avg_distance_traveled = 0
	avg_help_messages = 0
	
	for robot in json_dict["results"]:
		avg_sensor_activation += robot["sensor_activation"]
		avg_distance_traveled += robot["distance_traveled"]
		
	avg_sensor_activation /= len(json_dict["results"])
	avg_distance_traveled /= len(json_dict["results"])
	
	print("File:", log, "Team Quality of Work:", team_quality_work, "Team Speed of Work:", team_speed_of_work, "Team Achievement:", team_achievement, "Average Sensor Activation:", avg_sensor_activation, "Average Distance Traveled:", str(avg_distance_traveled) + " m")
	
print("Average Team Achievement:", avg_team_achievement/len(scores_files))
