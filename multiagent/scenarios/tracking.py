import numpy as np
from multiagent.core import World, Agent, Landmark ,Obstacle
from multiagent.scenario import BaseScenario
from tracking.target_pf import Target
from utilities.utilities import random_levy


class Scenario(BaseScenario):
    
    def make_world(self, num_agents=3, num_landmarks=3,num_obstacles=5, landmark_depth=15., landmark_movable = False, landmark_vel=0.05, max_vel=0.2, random_vel=False, movement='linear', pf_method = False, rew_err_th=0.0003, rew_dis_th=0.3, max_range = 2., max_current_vel=0.,range_dropping = 0.2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.num_obstacles = num_obstacles
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.04
            agent.max_a_speed = 3.1415
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks*2)]
        for i, landmark in enumerate(world.landmarks):
            if i < num_landmarks:
                landmark.name = 'landmark %d' % i
                landmark.collide = False
                landmark.movable = landmark_movable
            else:
                landmark.name = 'landmark_estimation %d' % (i-num_landmarks)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.002
        # 创建障碍物
        # 翻倍障碍物，为了估计观测
        world.obstacles = [Obstacle() for i in range(num_obstacles*2)]
        for i, obstacle in enumerate(world.obstacles):
            if i < num_obstacles:
                obstacle.name = 'obstacle %d' % i
                obstacle.collide = False
                obstacle.movable = True
                obstacle.size = np.random.uniform(0.05, 0.2)
            else:
                obstacle.name = 'obstacle_estimation %d' % (i-num_obstacles)
                obstacle.collide = False
                obstacle.movable = False
                obstacle.size = world.obstacles[i-num_obstacles].size - 0.02

        # make initial conditions
        world.cov = np.ones(num_landmarks)/30.
        world.error = np.ones(num_landmarks)
        
        #make initial world current 
        self.max_vel_ocean_current = max_current_vel
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
        
        # world.vel_ocean_current = 0.05
        # world.angle_ocean_current = np.pi/2.*3.
        
        self.landmark_vel = landmark_vel
        # print('test',landmark_vel)
        
        #benchmark variables  基准参数   增加与障碍物的碰撞
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        self.obstacle_collision = 0
        #Scenario initial conditions
        self.max_landmark_depth = landmark_depth
        #set random target depth
        self.landmark_depth = float(round(np.random.rand(1).item(0)*self.max_landmark_depth))
        if self.landmark_depth<15.:
            self.landmark_depth = 15.
        self.ra = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
        self.movement = movement
        self.pf_method = pf_method
        self.rew_err_th = rew_err_th
        self.rew_dis_th = rew_dis_th
        self.set_max_range = max_range
        #random variables for target
        self.max_vel = max_vel
        self.random_vel = random_vel
        self.reset_world(world)
        #
        world.damping = landmark_vel/5.
        
        self.range_dropping = range_dropping
        
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.0, 0.0, 1.0]) #蓝
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                landmark.color = np.array([0.0, 0.0, 0.0]) #黑
            else:
                landmark.color = np.array([0.85, 0.0, 0.0]) #红
        # 属性给障碍物
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                obstacle.color = np.array([0.0, 1.0, 0.0]) #绿
            else:
                obstacle.color = np.array([0.0, 1.0, 1.0]) #青色

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.9, 0.9, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_vel_old = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.a_vel = 0.
            agent.state.p_pos_origin = agent.state.p_pos.copy()
        for i, landmark in enumerate(world.landmarks):
            if i < world.num_landmarks:
                dis = np.random.uniform(0.4, 1.8)
                rad = np.random.uniform(0, np.pi*2)
                landmark.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(rad),np.sin(rad)])*dis
                while landmark.state.p_pos[0] < -0.95 or landmark.state.p_pos[0] > 0.95 or landmark.state.p_pos[1] < -0.95 or landmark.state.p_pos[1] > 0.95:
                    dis = np.random.uniform(0.4, 1.8)
                    rad = np.random.uniform(0, np.pi * 2)
                    landmark.state.p_pos = world.agents[0].state.p_pos + np.array([np.cos(rad), np.sin(rad)]) * dis
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.state.p_pos = world.landmarks[i-world.num_landmarks].state.p_pos
                landmark.state.p_vel = np.zeros(world.dim_p)
        #障碍物随机定位，假设只有一个目标点，让障碍物不接触目标点
        target_landmark = world.landmarks[0]
        for i,obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                # 在场地中随机生成位置
                O_pos = np.random.uniform(-0.9, 0.9, world.dim_p)  # 假设场地范围是[-1, 1]
                # 检查是否覆盖目标点,是否覆盖智能体
                while np.linalg.norm(O_pos - target_landmark.state.p_pos) <= (obstacle.size*2 + 0.05) or np.linalg.norm(O_pos - world.agents[0].state.p_pos) <= (obstacle.size*2 + 0.05):
                    O_pos = np.random.uniform(-0.9, 0.9, world.dim_p)  # 假设场地范围是[-1, 1]
                obstacle.state.p_pos = O_pos
                obstacle.state.p_vel = np.zeros(world.dim_p)
            else:
                obstacle.state.p_pos = world.obstacles[i-world.num_obstacles].state.p_pos
                obstacle.state.p_vel = np.zeros(world.dim_p)
        #Initailize the landmark estimated positions
        world.landmarks_estimated = [Target() for i in range(world.num_landmarks)]
        # 定义障碍物估计位置
        world.obstacles_estimated = [Target() for i in range(world.num_obstacles)]
        #initialize the ocean current at random
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
    
        #benchmark variables
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0
        self.obstacle_collision = 0
        #tacke a random velocity
        for landmark in world.landmarks:
            if self.random_vel == True:
                landmark.landmark_vel = np.random.rand(1).item(0)*self.max_vel
            else:
                landmark.landmark_vel = self.landmark_vel
            landmark.max_speed = landmark.landmark_vel
                
        #take a random direction
        for landmark in world.landmarks:
            landmark.ra = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction

            #take a random target depth
            landmark.landmark_depth = float(round(np.random.rand(1).item(0)*self.max_landmark_depth))
            if landmark.landmark_depth<15.:
                landmark.landmark_depth = 15.

        # 给障碍物设置一个随机的速度
        for obstacle in world.obstacles:
            if self.random_vel == True:
                obstacle.obstacle_vel = np.random.rand(1).item(0) * self.max_vel
            else:
                obstacle.obstacle_vel = self.landmark_vel  #后续可以考虑给障碍物加一个配置属性，最大深度，最大速度，等等
            obstacle.max_speed = obstacle.obstacle_vel
        # 给障碍物设置一个随机的方向
        for obstacle in world.obstacles:
            obstacle.ra = (np.random.rand(1) * np.pi * 2.).item(0)  # initial obstacle direction

            # 给障碍物设置一个随机的深度
            obstacle.obstacle_depth = float(round(np.random.rand(1).item(0) * self.max_landmark_depth))
            if obstacle.obstacle_depth < 15.:#深度最少15
                obstacle.obstacle_depth = 15.
    def benchmark_data(self, agent, world):
        landmarks_real_p = []
        for i in range(world.num_landmarks):
            landmarks_real_p.append(world.landmarks[i].state.p_pos)
        # return (rew, collisions, min_dists, occupied_landmarks,landmarks_real_p)
        return(world.error,landmarks_real_p, self.agent_outofworld, self.landmark_collision, self.agent_collision,self.obstacle_collision)  #后续要加入一个障碍物与智能体碰撞次数


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    done_state = False
    def reward(self, agent, world):
        global done_state
        done_state = False
        # Agents are rewarded based on landmarks_estimated covariance_vals, penalized for collisions
        rew = 0.
        
        for i,l in enumerate(world.landmarks_estimated): #计算估计地标位置与真实地标位置关系，给予奖惩
            if self.pf_method == True:
                world.cov[i] = np.sqrt((l.pf.covariance_vals[0])**2+(l.pf.covariance_vals[1])**2)/10.
                # rew -= world.cov[i]/100
            if self.pf_method == True:
                world.error[i] = np.sqrt((l.pfxs[0]-world.landmarks[i].state.p_pos[0])**2+(l.pfxs[2]-world.landmarks[i].state.p_pos[1])**2) #Error from PF
            else:
                world.error[i] = np.sqrt((l.lsxs[-1][0]-world.landmarks[i].state.p_pos[0])**2+(l.lsxs[-1][2]-world.landmarks[i].state.p_pos[1])**2) #Error from LS
            #world.error是估计位置与真实位置的偏差值，如果值小可以奖励
            #REWARD: Based on target estimation error, for each target
            if world.error[i]<self.rew_err_th:
                rew += 1.
            else:
                rew += 0.01*(0.004-world.error[i])
            
            #REWARD: Based on the distance between landmark and agent  计算智能体与所有地标的距离，根据距离给予奖励
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[i].state.p_pos))) for a in world.agents]
            if min(dists) < self.rew_dis_th: #other tests
                rew += 1.
            else:
                rew += 0.01*(0.7-min(dists))
        
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks[:-world.num_landmarks]]
        
                
        if min(dists) > self.set_max_range: #agent out of range 智能体与目标地标要是距离超过了范围，则会惩罚
            dist_from_origin = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.state.p_pos_origin))) #计算智能体目前与源点的距离
            if dist_from_origin > self.set_max_range*2. : #agent outside the world 与源点距离相差两倍的最大范围，则判断智能体逃离世界，惩罚
                rew -= 100
                done_state = True
                self.agent_outofworld += 1
            else:
                rew -= 0.1
        else:
            # the agent is close to the target, and therefore, we save its position as new origin.
            agent.state.p_pos_origin = agent.state.p_pos.copy()
            
        if min(dists) < 0.02: #is collision 撞上了地标
            rew -= 1.
            done_state = True
            self.landmark_collision += 1
        
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 10.
                    self.agent_collision += 1
                    done_state = True

        for i,obstacle in enumerate(world.obstacles): #暂时不加与估计位置的奖惩机制 （等测试完其他功能）
            if i < world.num_obstacles:
                dist_to_obstacle = np.sqrt(np.sum(np.square(agent.state.p_pos - obstacle.state.p_pos)))
                if dist_to_obstacle < (obstacle.size + agent.size):  # 如果与障碍物距离过近
                    self.obstacle_collision += 1
                    done_state = True
                    rew -= 5.0  # 给予惩罚
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        entity_range = []
        entity_depth = []
        obstacle_range = []
        obstacle_depth = []
        for i, entity in enumerate(world.landmarks):
            if i < world.num_landmarks: 
                #Update the landmarks_estiamted position using Particle Fileter
                #1:Compute radius between the agent and each landmark
                slant_range = np.sqrt(((entity.state.p_pos - agent.state.p_pos)[0])**2+((entity.state.p_pos - agent.state.p_pos)[1])**2)
                target_depth = entity.landmark_depth/1000. #normalize the target depth
                slant_range = np.sqrt(slant_range**2+target_depth**2) #add target depth to the range measurement
                # Add some systematic error in the measured range
                slant_range *= 1.01 # where 0.99 = 1% of sound speed difference = 1495 m/s
                # Add some noise in the measured range
                slant_range += np.random.uniform(-0.001, +0.001)
                # Return to a planar range
                slant_range = np.sqrt(abs(slant_range**2-target_depth**2))
                
                #set a maximum range between target and agent where the measurement can not be conducted.
                if slant_range > self.set_max_range * 3 or np.random.rand() < self.range_dropping:
                    slant_range = -1.
                    new_range = False
                else:
                    new_range = True
                
                #2:Update the PF
                add_pos_error = False
                if self.pf_method == True:
                    if add_pos_error == True:
                        world.landmarks_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0]+np.random.randn(1).item(0)*3/1000.,0.,agent.state.p_pos[1]+np.random.randn(1).item(0)*3/1000.,0.], update=new_range)
                    else:
                        world.landmarks_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.], update=new_range)
                else:
                    #2b: Update the LS
                    if add_pos_error == True:
                        #平滑处理
                        smoothed_myobserver = self.smooth_data(
                            [agent.state.p_pos[0] + np.random.randn(1).item(0) * 3 / 1000., 0.,
                             agent.state.p_pos[1] + np.random.randn(1).item(0) * 3 / 1000., 0.])
                        world.landmarks_estimated[i].updateLS(dt=0.04, new_range=new_range, z=slant_range,
                                                              myobserver=smoothed_myobserver)
                        #world.landmarks_estimated[i].updateLS(dt=0.04, new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0]+np.random.randn(1).item(0)*3/1000.,0.,agent.state.p_pos[1]+np.random.randn(1).item(0)*3/1000.,0.])
                    else:
                        #平滑处理
                        smoothed_myobserver = self.smooth_data([agent.state.p_pos[0], 0., agent.state.p_pos[1], 0.])
                        world.landmarks_estimated[i].updateLS(dt=30., new_range=new_range, z=slant_range,
                                                              myobserver=smoothed_myobserver)
                        #world.landmarks_estimated[i].updateLS(dt=30., new_range=new_range, z=slant_range, myobserver=[agent.state.p_pos[0],0.,agent.state.p_pos[1],0.])
                # Traditional plot
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(5,5))
                # plt.plot(world.landmarks_estimated[i].pf._x[0],world.landmarks_estimated[i].pf._x[2], 'r^', ms=20)
                # plt.plot(world.landmarks_estimated[i].pf.x.T[0],world.landmarks_estimated[i].pf.x.T[2], 'ro', ms=5, alpha=0.3)
                # plt.plot(world.landmarks[0].state.p_pos[0],world.landmarks[0].state.p_pos[1], 'ko', ms=6, alpha = 0.5)
                # plt.plot(world.agents[0].state.p_pos[0],world.agents[0].state.p_pos[1], 'bo', ms=6, alpha = 0.5)
                # plt.xlim(-1,1)
                # plt.ylim(-1,1)
                # plt.show()
                #3:Publish the new estimated position
                try:
                    if self.pf_method == True:
                        world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].pfxs[0],world.landmarks_estimated[i].pfxs[2]] #Using PF
                    else:
                        world.landmarks[i+world.num_landmarks].state.p_pos = [world.landmarks_estimated[i].lsxs[-1][0],world.landmarks_estimated[i].lsxs[-1][2]] #Using LS
                except:
                    #An error will be produced if its the initial time and no good range measurement has been conducted yet. In this case, we supose that the target 
                    #is at the same position of the agent.
                    world.landmarks[i+world.num_landmarks].state.p_pos = agent.state.p_pos.copy()
                #Append the position of the landmark to generate the observation state
                #Using the true landmark position
                # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                #Using the estimated landmark position
                entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos - agent.state.p_pos)
                #Using the estimated landmark position but without delating the agent position. so it has a global position.
                # entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos)
                entity_range.append(slant_range)
                entity_depth.append(target_depth)
                
                # Move the landmark if movable
                if entity.movable:
                    if self.movement == 'linear':
                        #linear movement
                        u_force = entity.landmark_vel
                        entity.action.u = np.array([np.cos(entity.ra)*u_force,np.sin(entity.ra)*u_force])
                    
                    elif self.movement == 'random':
                        # random movement
                        entity.action.u = np.random.randn(2)/2.
                    
                    elif self.movement == 'levy':
                        #random walk Levy movement
                        beta = 1.9 #must be between 1 and 2
                        entity.action.u = random_levy(beta)
                        if entity.state.p_pos[0] > 0.8:
                            entity.action.u[0] = -abs(entity.action.u[0])
                        if entity.state.p_pos[0] < -0.8:
                            entity.action.u[0] = abs(entity.action.u[0])
                        if entity.state.p_pos[1] > 0.8:
                            entity.action.u[1] = -abs(entity.action.u[1])
                        if entity.state.p_pos[1] < -0.8:
                            entity.action.u[1] = abs(entity.action.u[1])

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        #增加障碍物观测位置值 （可能会更改网络的输入，大工程！）（需要将障碍物位置进行估计观测，类比地标的位置观测）
        obstacle_pos = []
        #观测障碍物估计位置 （先把障碍物实体翻倍）
        for i, obstacle in enumerate(world.obstacles):
            if i < world.num_obstacles:
                # Update the landmarks_estiamted position using Particle Fileter
                # 1:Compute radius between the agent and each landmark
                slant_range = np.sqrt(((obstacle.state.p_pos - agent.state.p_pos)[0]) ** 2 + (
                (obstacle.state.p_pos - agent.state.p_pos)[1]) ** 2)
                target_depth = obstacle.obstacle_depth / 1000.  # normalize the target depth
                slant_range = np.sqrt(slant_range ** 2 + target_depth ** 2)  # add target depth to the range measurement
                # Add some systematic error in the measured range
                slant_range *= 1.01  # where 0.99 = 1% of sound speed difference = 1495 m/s
                # Add some noise in the measured range
                slant_range += np.random.uniform(-0.001, +0.001)
                # Return to a planar range
                slant_range = np.sqrt(abs(slant_range ** 2 - target_depth ** 2))

                # set a maximum range between target and agent where the measurement can not be conducted.
                if slant_range > self.set_max_range * 3 or np.random.rand() < self.range_dropping:
                    slant_range = -1.
                    new_range = False
                else:
                    new_range = True

                # 2:Update the PF
                add_pos_error = False
                if self.pf_method:
                    if add_pos_error:
                        world.obstacles_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range, myobserver=[
                            agent.state.p_pos[0] + np.random.randn(1).item(0) * 3 / 1000., 0.,
                            agent.state.p_pos[1] + np.random.randn(1).item(0) * 3 / 1000., 0.], update=new_range)
                    else:
                        world.obstacles_estimated[i].updatePF(dt=30., new_range=new_range, z=slant_range,
                                                              myobserver=[agent.state.p_pos[0], 0.,
                                                                          agent.state.p_pos[1], 0.], update=new_range)
                else:
                    # 2b: Update the LS
                    if add_pos_error:
                        world.obstacles_estimated[i].updateLS(dt=0.04, new_range=new_range, z=slant_range, myobserver=[
                            agent.state.p_pos[0] + np.random.randn(1).item(0) * 3 / 1000., 0.,
                            agent.state.p_pos[1] + np.random.randn(1).item(0) * 3 / 1000., 0.])
                    else:
                        world.obstacles_estimated[i].updateLS(dt=30., new_range=new_range, z=slant_range,
                                                              myobserver=[agent.state.p_pos[0], 0.,
                                                                          agent.state.p_pos[1], 0.])
                # Traditional plot
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(5,5))
                # plt.plot(world.landmarks_estimated[i].pf._x[0],world.landmarks_estimated[i].pf._x[2], 'r^', ms=20)
                # plt.plot(world.landmarks_estimated[i].pf.x.T[0],world.landmarks_estimated[i].pf.x.T[2], 'ro', ms=5, alpha=0.3)
                # plt.plot(world.landmarks[0].state.p_pos[0],world.landmarks[0].state.p_pos[1], 'ko', ms=6, alpha = 0.5)
                # plt.plot(world.agents[0].state.p_pos[0],world.agents[0].state.p_pos[1], 'bo', ms=6, alpha = 0.5)
                # plt.xlim(-1,1)
                # plt.ylim(-1,1)
                # plt.show()
                # 3:Publish the new estimated position
                try:
                    if self.pf_method == True:
                        world.obstacles[i + world.num_obstacles].state.p_pos = [world.obstacles_estimated[i].pfxs[0],
                                                                                world.obstacles_estimated[i].pfxs[
                                                                                    2]]  # Using PF
                    else:
                        world.obstacles[i + world.num_obstacles].state.p_pos = [
                            world.obstacles_estimated[i].lsxs[-1][0],
                            world.obstacles_estimated[i].lsxs[-1][2]]  # Using LS
                except:
                    # An error will be produced if its the initial time and no good range measurement has been conducted yet. In this case, we supose that the target
                    # is at the same position of the agent.
                    world.obstacles[i + world.num_obstacles].state.p_pos = agent.state.p_pos.copy()
                # Append the position of the landmark to generate the observation state
                # Using the true landmark position
                # entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                # Using the estimated landmark position
                obstacle_pos.append(world.obstacles[i + world.num_obstacles].state.p_pos - agent.state.p_pos)
                # Using the estimated landmark position but without delating the agent position. so it has a global position.
                # entity_pos.append(world.landmarks[i+world.num_landmarks].state.p_pos)
                obstacle_range.append(slant_range)
                obstacle_depth.append(target_depth)

                # Move the landmark if movable
                if obstacle.movable:
                    if self.movement == 'linear':
                        # linear movement
                        u_force = obstacle.obstacle_vel
                        obstacle.action.u = np.array([np.cos(obstacle.ra) * u_force, np.sin(obstacle.ra) * u_force])

                    elif self.movement == 'random':
                        # random movement
                        obstacle.action.u = np.random.randn(2) / 2.
                    elif self.movement == 'levy':
                        # random walk Levy movement
                        beta = 1.9  # must be between 1 and 2
                        obstacle.action.u = random_levy(beta)
                        if obstacle.state.p_pos[0] > 0.8:
                            obstacle.action.u[0] = -abs(obstacle.action.u[0])
                        if obstacle.state.p_pos[0] < -0.8:
                            obstacle.action.u[0] = abs(obstacle.action.u[0])
                        if obstacle.state.p_pos[1] > 0.8:
                            obstacle.action.u[1] = -abs(obstacle.action.u[1])
                        if obstacle.state.p_pos[1] < -0.8:
                            obstacle.action.u[1] = abs(obstacle.action.u[1])


        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + [entity_range] + [entity_depth] + [agent.state.p_pos_origin] + obstacle_pos + [obstacle_range] + [obstacle_depth] )

    def smooth_data(self, data):
        """
        简单的移动平均平滑处理
        """
        window_size = 3  # 窗口大小
        smoothed_data = []
        for i in range(len(data)):
            if i < window_size:
                smoothed_data.append(data[i])
            else:
                smoothed_data.append(np.mean(data[i - window_size:i], axis=0))
        return smoothed_data
    def done(self, agent, world):
        # episodes are done based on the agents minimum distance from a landmark.
        global done_state
        if done_state:
            done = True
        else:
            done = False
        return done
