import numpy as np
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class obstacle:
    # will assume the obstacle has a fixed policy
    def __init__(self, grid_size):
        self.policy = [25, 25, 25, 25]
        self.grid_size = grid_size
        self.x = np.random.randint(grid_size//4, grid_size)
        self.y = np.random.randint(grid_size//4, grid_size)

    def reset(self):
        self.x = np.random.randint(self.grid_size//4, self.grid_size)
        self.y = np.random.randint(self.grid_size//4, self.grid_size)

class greenball:
    def __init__(self, policy, grid_size):
        self.policy = policy

        self.starts = [(0, 0),(3, 0), (0, 2), (1, 2)]
        self.x, self.y  = self.starts[np.random.randint(0, len(self.starts))]

    def reset(self):
        self.x, self.y  = self.starts[np.random.randint(0, len(self.starts))]


class Game:
    def __init__(self, num_obstacles, policy, grid_size, MIN_DIST, LEN_LIMIT):
        self.traj_store = []
        self.grid_size = grid_size
        self.MIN_DIST = MIN_DIST
        self.LEN_LIMIT = LEN_LIMIT
        self.greenball = greenball(policy, grid_size)

        self.obstacles = []
        for i in range(num_obstacles):
            obs = obstacle(grid_size)
            self.obstacles.append(obs)

    def reset(self):
        # reset things
        for ob in self.obstacles:
            ob.reset()
        self.greenball.reset()

    def play_game(self):

        traj = []
        for k in range(self.LEN_LIMIT):
            #self.plotgrid(k)
            g_x, g_y = self.greenball.x, self.greenball.y
            action, new_loc = self.move_greenball(self.MIN_DIST)
            nx, ny = new_loc

            traj.append(((g_x, g_y), action, (nx, ny)))

            if (nx == self.grid_size -1) and (ny == self.grid_size -1):
                print("game over")
                self.reset()
                return traj

            self.greenball.x, self.greenball.y = nx, ny

            # make obstacle move
            moves = self.move_obstacles()
            for i, ob in enumerate(self.obstacles):
                x, y = moves[i]
                ob.x = x
                ob.y = y

        self.reset()
        return traj


    def plotgrid(self):
        # Create a 10x10 table with random values
        table_data = [[(i, j) for j in range(10)] for i in range(10)]

        # Clear the previous plot
        plt.clf()

        # Create a figure and axis for plotting
        fig, ax = plt.subplots()

        # Create a table and add it to the axis
        pd.DataFrame(table_data)

        # Plot obstacles
        for ob in self.obstacles:
            ax.plot(ob.x, ob.y, marker='o', color='black', markersize=10)

        # Plot greenball
        g = self.greenball
        ax.plot(g.x, g.y, marker='o', color='red', markersize=10)  # Use 'o' marker for a circle

        # plot objective
        ax.plot(self.grid_size -1, self.grid_size -1, marker='s', color='green', markersize=10)

        ax.set_xlim(-1, self.grid_size + 1)
        ax.set_ylim(-1, self.grid_size + 1)
        ax.set_xticks(np.arange(grid_size + 1) - 1)
        ax.set_yticks(np.arange(grid_size + 1) - 1)

        plt.gca().invert_yaxis()  # Invert the y-axis

        # Add grid lines for both axes
        ax.grid(True)
        plt.show()


    def move_greenball(self, MIN_DIST):

        # location of obstacles
        obstacle_locations = [(ob.x, ob.y) for ob in self.obstacles]
        x, y = self.greenball.x, self.greenball.y
        grid_size = self.grid_size
        # randomly generate a list of actions to be conducted based on self.cumm_policy
        check_bounds, check_obstacles = {}, {}
        policy = deepcopy(self.greenball.policy)
        action_set = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        selected_action = -1

        while sum(policy) > 0:
            obstacle_near = False
            val = np.random.randint(0, sum(policy))

            for index in range(len(policy)):
                if val <= sum(policy[:index+1]):
                    action = action_set[index]
                    break;

            # now check whether the action obeys bounds
            if action == 0:   # North
                if y - 1 >= 0:
                    x1, y1 = x, y - 1
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 1:  # North-East
                if y - 1 >= 0 and x + 1 <= grid_size - 1:
                    x1, y1 = x + 1, y - 1
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 2:  # East
                if x + 1 <= grid_size - 1:
                    x1, y1 = x + 1, y
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 3:  # South-East
                if y + 1 <= grid_size - 1 and x + 1 <= grid_size - 1:
                    x1, y1 = x + 1, y + 1
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 4:  # South
                if y + 1 <= grid_size - 1:
                    x1, y1 = x, y + 1
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 5:  # South-West
                if y + 1 <= grid_size - 1 and x - 1 >= 0:
                    x1, y1 = x - 1, y + 1
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 6:  # West
                if x - 1 >= 0:
                    x1, y1 = x - 1, y
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 7:  # North-West
                if x - 1 >= 0 and y - 1 >= 0:
                    x1, y1 = x - 1, y - 1
                    obstacle_near = self.obstacle_check(x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)
                else:
                    check_bounds[action] = index

            elif action == 8:
                x1 = np.random.randint(0, grid_size)
                y1 = np.random.randint(0, grid_size)

                # in case random action is already in feasible trans
                feasible_trans = {(x, y - 1):0, (x + 1, y - 1):1, (x + 1, y):2,
                        (x + 1, y + 1):3, (x, y + 1):4, (x - 1, y + 1):5,
                        (x - 1, y):6, (x - 1, y - 1):7}

                if (x1, y1) in feasible_trans:
                    action = feasible_trans[(x1, y1)]
                obstacle_near = self.obstacle_check (x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST)

            # now check bounds
            if action in check_bounds:
                policy.pop(index)
                action_set.pop(index)
                check_bounds = {} # reset checkbounds

            elif (action in check_obstacles) and (action in action_set):
                policy.pop(index)
                action_set.pop(index)

            elif obstacle_near == False:
                selected_action = action
                return selected_action, (x1, y1)

        if selected_action == -1:
            #just select random action out of check-oobstacles since we cant avoid obstacle
            all_actions = [k for k in check_obstacles.keys()]
            selected_action = all_actions[np.random.randint(0, len(all_actions))]
        return selected_action, check_obstacles[selected_action]

    def obstacle_check (self, x1, y1, check_obstacles, obstacle_locations, action, MIN_DIST):
        d = 1000
        obstacle_near = False
        for x0, y0 in obstacle_locations:
            manhattan_dist = abs(x1 - x0) + abs(y1 - y0)
            d = min(d, manhattan_dist)

            if d < MIN_DIST:
                obstacle_near = True
                check_obstacles[action] = (x1, y1) # dont want to take this action
        return obstacle_near


    def move_obstacles(self):
        grid_size = self.grid_size
        actions_arr = []
        for obstacle in self.obstacles:
            x, y = obstacle.x, obstacle.y
            # randomly generate a list of actions to be conducted based on self.cumm_policy
            check_bounds = {}
            policy = deepcopy(obstacle.policy)
            action_set = [0, 1, 2, 3]

            while len(policy) > 0:
                val = np.random.randint(0, sum(policy))
                for index in range(len(policy)):
                    if val <= sum(policy[:index+1]):
                        action = action_set[index]
                        break;

                # now check whether the action obeys bounds
                if action == 0:   #up
                    if y - 1 >= 0:
                        # now check closest location to obstacles
                        x1, y1 = x, y-1
                    else:
                        # cant make action hence we break
                        check_bounds[action] = index

                elif action == 1: #right
                    if x + 1 <= grid_size - 1:
                        x1, y1 = x+1, y
                    else:
                        check_bounds[action] = index

                elif action == 2: # down
                    if y + 1 <= grid_size -1:
                        x1, y1 = x, y+1
                    else:
                        check_bounds[action] = index

                elif action == 3: #left
                    if x - 1 >= 0:
                        x1, y1 = x-1, y
                    else:
                        check_bounds[action] = index

                # now check bounds
                if action in check_bounds:
                    policy.pop(index)
                    action_set.pop(index)
                    check_bounds = {} # reset checkbounds
                else:
                    actions_arr.append((x1, y1))
                    break;

        return actions_arr
