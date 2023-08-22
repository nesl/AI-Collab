using TaskGraphs
using Debugger
import JSON



function debug_f()
    
  
    ## set up the environment
    #vtx_grid = initialize_dense_vtx_grid(20,20) # 4 x 4 grid world
    #  1   2   3   4
    #  5   6   7   8
    #  9  10  11  12
    # 13  14  15  16

    
    map_data = JSON.parsefile("./map.json")

    pickup_zones = Vector{Int}()
    dropoff_zones = Vector{Int}()
    obstacles = Vector{Tuple{Int,Int}}()
    vtxs = Vector{Tuple{Int,Int}}()
    objects_locations = Vector{Int}()
    robots_locations = Vector{Int}()
    robot_ics = Vector{Any}()
    robot_index = 1
    object_index = 1

    current_drop_location = [round(Int,length(map_data["data"])/2),round(Int,length(map_data["data"][1])/2)]
    N_times = 1
    small_n = 1
    n_index = 1
    
    Ap = zeros(Int, length(map_data["data"]),length(map_data["data"][1]))
    ap_idx = 1

    #movement_drop_location = [[[0,1],[1,0],[0,-1]],[[0,-1],[-1,0],[0,1]]]
    
    ## Define the manufacturing project
    spec = ProjectSpec()
    
    # define the operations that need to take place
    op1 = Operation(Δt=0) # only one operation
    
    for (i, element1) in enumerate(map_data["data"])
        for (j,element2) in enumerate(element1)
            index = (i-1)*length(element1)+j
            if i == 1 || j == 1 || i == length(map_data["data"]) || j == length(map_data["data"][1]) || element2 == 1
                push!(obstacles,(i,j))
                
            elseif element2 == 2 #object
                push!(objects_locations,index)
                set_initial_condition!(spec,OBJECT_AT(object_index,index))
                object_drop_location = (current_drop_location[1]-1)*length(element1)+current_drop_location[2]
                set_precondition!(op1,OBJECT_AT(object_index,object_drop_location))

                
                #assign final position in a spiral order starting at the center of the room
                
                
                
                if n_index == 1
                    current_drop_location[2] += 1
                    n_index += 1
                elseif n_index == 2
                    current_drop_location[1] += 1
                    
                    if small_n == N_times
                        n_index += 1
                        small_n = 1
                    else
                        small_n += 1
                    end
                    
                elseif n_index == 3
                    current_drop_location[2] -= 1
                    if small_n == N_times
                        small_n = 1
                        n_index += 1
                        N_times += 1
                    else
                        small_n += 1
                    end
                elseif n_index == 4
                    current_drop_location[2] -= 1
                    n_index += 1
                elseif n_index == 5
                    current_drop_location[1] -= 1
                    if small_n == N_times
                        small_n = 1
                        n_index += 1
                    else
                        small_n += 1
                    end
                else
                    current_drop_location[2] += 1
                    if small_n == N_times
                        small_n = 1
                        n_index = 1
                        N_times += 1
                    else
                        small_n += 1
                    end
                end


                
                object_index += 1

            elseif element2 == 3 || element2 == 5 #robots
                push!(robots_locations,index)
                push!(robot_ics,ROBOT_AT(robot_index,index))
                robot_index += 1
            end
            
            if ! (i == 1 || j == 1 || i == length(map_data["data"]) || j == length(map_data["data"][1]) || element2 == 1)
                Ap[i,j]  = ap_idx
                
                push!(vtxs, (i, j))
                push!(pickup_zones, ap_idx)
                push!(dropoff_zones, ap_idx)
                ap_idx += 1
            end
        end
    end
    
    free_zones = collect(1:length(vtxs))
    
    vtx_grid = initialize_dense_vtx_grid(20,20)
    env = construct_factory_env_from_vtx_grid(vtx_grid)
    
    #=
    graph = initialize_grid_graph_from_vtx_grid(Ap)

    env = GridFactoryEnvironment(
        graph = graph,
        x_dim = length(map_data["data"]),
        y_dim = length(map_data["data"]),
        cell_width = 1.0,
        transition_time = 1.0,
        vtxs = vtxs,
        pickup_zones = pickup_zones,
        dropoff_zones = dropoff_zones,
        free_zones = free_zones,
        obstacles = obstacles,
    )
    =#
    
    add_operation!(spec,op1)

    
    #=
    vtx_grid = initialize_dense_vtx_grid(20,20) # 4 x 4 grid world
    env = construct_factory_env_from_vtx_grid(vtx_grid)
    ## Define the initial conditions of the robots
    robot_ics = [
        ROBOT_AT(1,2), # robot 1 starts at vertex 2
        ROBOT_AT(2,9), # robot 2 starts at vertex 9
    ]

    ## Define the manufacturing project
    spec = ProjectSpec()
    # set initial conditions of "raw materials"
    set_initial_condition!(spec,OBJECT_AT(1,4)) # object 1 starts at vertex 4
    set_initial_condition!(spec,OBJECT_AT(2,16))  # object 2 starts at vertex 16
    # define the operations that need to take place
    op1 = Operation(Δt=2) # operation 1 has a duration of 2 time steps 
    # inputs
    set_precondition!(op1,OBJECT_AT(1,8)) # object 1 must be at vertex 8 before op1 can begin
    set_precondition!(op1,OBJECT_AT(2,12)) # object 2 must be at vertex 12 before op1 can begin
    # outputs
    set_postcondition!(op1,OBJECT_AT(3,7)) # object 3 appears at vertex 7 when op1 is completed 
    add_operation!(spec,op1)
    # add a terminal operation with no outputs
    op2 = Operation(Δt=0)
    set_precondition!(op2,OBJECT_AT(3,13))
    add_operation!(spec,op2)
    

    =#    

    ## define solver
    solver = NBSSolver()
    solver.logger.runtime_limit = 3600.0
    
    # finalize problem construction (the solver is passed as an argument here 
    # because it determines what cost model is used for the problem)
    prob = pctapf_problem(solver,spec,env,robot_ics)
    # solve the problem
    solution, cost = solve!(solver,prob)
    # check if the problem was solved to optimality
    @show feasible_status(solver)
    @show optimal_status(solver)

    
    print(solution.route_plan)
    
    all_paths = Vector{Vector{Int}}()
    for robot_path in solution.route_plan.paths
        robot_locations = Vector{Int}()

        for path_idx in 1:length(robot_path)
            push!(robot_locations,robot_path[path_idx].sp.vtx)
        end
        push!(all_paths,robot_locations)
    end
    
    json_data = JSON.json(Dict("data" => all_paths))
    
    open("./results.json", "w") do file
        write(file, json_data)
    end

    
end

debug_f()
