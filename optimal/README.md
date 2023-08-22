# Optimal Sequential Task Assignment and Path Finding

## Assumptions

Ideally, a central planner assigns tasks to agents. An agent can move in 8 directions, pick up an object in 8 directions and drop an object. The tasks at hand consist of picking up all objects and dropping them in the middle of the room.

In practice, right now, the current implementation in Julia has some extra limitations: agents can only move in 4 directions (up, down, left, right), agents pick up objects by moving into the cell of the object to be picked up, there are no walls (although they can easily be added as obstacles), and all objects are meant to be carried and dropped in the middle of the room (whereas in reality only a subset will be). Also a great constraint is that we need to specify a particular drop location to each object before the optimization starts, instead of allowing the optimizing process to determine automatically the best location for the object to be dropped in. Therefore, we assign objects to individual locations in the middle of the room following a spiral order, starting at the central cell.
The number of objects, robots and the size of the grid is obtained by parsing an occupancy map stored in **map.json**, for example:

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

Where 2 is an object and 3 is a robot.

Objects and robots are assigned an initial location, and objects are assigned also a final location to where they are to be carried to.

Some example results from this optimization process are shown in **results.json**. In this file there is a list of lists. Each list correspond to the cells at which a robot will be sequentially, for each discrete time step, starting at time 0.

The way cells are numbered reflects the following cell ordering for a given occupancy map size:

 ```
[[1,2,3,4],
[5,6,7,8],
[9,10,11,12],
[13,14,15,16]]
 ```

The results in this case are for a setting with 5 agents and 20 objects, all of them to be picked up and dropped in the middle of the room. The program fails to compute an optimal solution, only a feasible solutions according to their parameters of the solver.
