using Debugger
using Random
using Scheduling, Scheduling.Algorithms, Scheduling.Objectives
using LinearAlgebra


# Directions are seen as cartesian indexes so that they can be added to a position to get the next position
const UP = CartesianIndex(-1, 0)
const DOWN = CartesianIndex(1, 0)
const LEFT = CartesianIndex(0, -1)
const RIGHT = CartesianIndex(0, 1)
const DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

# manhattan distance between positions in the maze matrix
manhattan(a::CartesianIndex, b::CartesianIndex) = sum(abs.((b - a).I))
# check to be in the maze and filter out moves that go into walls
function mazeneighbours(maze, p)
  res = CartesianIndex[]
  for d in DIRECTIONS
      n = p + d
      if 1 ≤ n[1] ≤ size(maze)[1] && 1 ≤ n[2] ≤ size(maze)[2] && !maze[n]
          push!(res, n)
      end
  end
  return res
end

function solvemaze(maze, start, goal)
  currentmazeneighbours(state) = mazeneighbours(maze, state)
  # Here you can use any of the exported search functions, they all share the same interface, but they won't use the heuristic and the cost
  return astar(currentmazeneighbours, start, goal, heuristic=manhattan)
end

# 0 = free cell, 1 = wall

#start = CartesianIndex(1, 1)
#goal = CartesianIndex(1, 5)

#res = solvemaze(maze, start, goal)

function shortest_path(object_position, agents_positions, maze)
	
	distances = Vector{Any}()
	goal = CartesianIndex(object_position[1], object_position[2])
	
	for row in agents_positions
		
		start = CartesianIndex(row[1], row[2])
		push!(distances,length(solvemaze(maze, start, goal).path)-1)
	end
	
	return distances
end




function gen_jobs(n , m, weights, distances)

	jobs = Array{Job}(undef, n)

	M = 300000000

	for i in 1:n
		p = Array{Float64}(undef, m)
		for j in 1:m
			if j >= weights[i] 
				p[j] = distances[i]
			else
				p[j] = M
			end
		end
		jobs[i] = Job(string(i), ParallelJobParams(p))
	end

	return jobs
end


Random.seed!(5)



function run_function()

	
	object_positions = [[4,2],[5,2],[4,3]]
	num_objects = length(object_positions)
	agents_positions = [[1,1],[1,2],[1,4],[2,1],[3,1]]
	weights = [5,3,1]
	
	maze = [0 0 1 0 0;
        	0 1 0 0 0;
	        0 1 0 0 1;
        	0 0 0 1 1;
	        1 0 1 0 0] .== 1
	
	distances = Vector{Any}()
	for object_idx in 1:length(object_positions)
		push!(distances,convert(Integer,round(shortest_path(object_positions[object_idx], agents_positions, maze)[1])))
	end
	
	m = length(agents_positions)
	machines = Machines(m)
	
	n = length(object_positions)

	jobs = gen_jobs(n , m, weights, distances)
	
	sched = Algorithms.P_any_Cmax_TWY(jobs, machines)

	println("\n-------")
	println("m=$(m)")
	println("n=$(n)")

	mrt_cmax = cmax(sched)

	println(sched)
	#println("p=$(p)")
	println("makespan: $(mrt_cmax)")

	#Scheduling.plot(sched)
end

@enter run_function()
