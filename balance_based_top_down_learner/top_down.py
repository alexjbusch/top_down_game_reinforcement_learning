import random
from itertools import cycle

import pygame
import math

import torch

import time

FPS = 60
SCREENWIDTH = 512
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
clock = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))

pygame.display.set_caption('Top Down Game')

BASEY = SCREENHEIGHT * 0.79

PLAYER_RADIUS = 3
ENEMY_RADIUS = 3
BULLET_RADIUS = 1



class Player():
    def __init__(self, gamestate):
        self.radius = 20
        self.x = 260
        self.y = 400
        self.color = (15,100,255)

        self.angle = 200

        self.gamestate = gamestate

        self.turn_speed = 3
    def draw(self):
        pygame.draw.circle(SCREEN, self.color, (self.x,self.y), self.radius)
        target = (math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle)))
        pygame.draw.line(SCREEN, (255,0,0), (self.x,self.y), (self.x+target[0]*700,self.y+target[1]*700), width=2)


    def draw_line(self, angle, color):
        target = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        pygame.draw.line(SCREEN, color, (self.x,self.y), (self.x+target[0]*10,self.y+target[1]*10), width=1)

    def fire(self):
        target = math.cos(math.radians(self.angle)), math.sin(math.radians(self.angle))
        bullet = Bullet(self.x, self.y, target, friendly=True)
        self.gamestate.bullets.append(bullet)

    def rotate_CW(self):
        self.angle += self.turn_speed

    def rotate_counter_CW(self):
        self.angle -= self.turn_speed

    def got_hit(self):
        return False

    def laser_on_target(self, target):

        dx = math.cos(math.radians(self.angle))
        dy = math.sin(math.radians(self.angle))
        x0,y0 = target.x, target.y
        x1,y1 = self.x,self.y
        d = (x0-x1) * dx + (y0-y1) * dy
        ox = dx * d + x1
        oy = dy * d + y1
        distance = math.sqrt((ox-x0)**2 + (oy-y0)**2)

        if d > 0 and distance < target.radius:
            collision = True
        else:
            collision = False



        if collision:
            target.color = target.hit_color
        else:
            target.color = target.original_color
        return collision
        """
        x2,y2 = (math.cos(math.radians(self.angle))*10000, math.sin(math.radians(self.angle))*10000)
        x2 += self.x
        y2 += self.y
        x1,y1 = self.x, self.y
        r = target.radius
        x0 = target.x
        y0 = target.y

        if x2 > x1 and y2 > y1:
            collision = False
        else:
            collision = abs((x2-x1)*x0 + (y1-y2)*y0 + (x1-x2)*y1 + x1*(y2-y1))/ math.sqrt((x2-x1)**2 + (y1-y2)**2) <= r

        if collision:
            target.color = target.hit_color
        else:
            target.color = target.original_color

        return collision
        """
        
class Enemy():
    def __init__(self, gamestate):
        self.radius = 20
        self.x = 200
        self.y = 200

        self.original_color = (255,0,0)
        self.hit_color = (255,223,0)
        
        self.color = (255,0,0)

        self.dx = 10
        self.dy = 0

        self.gamestate = gamestate
    def draw(self):
        pygame.draw.circle(SCREEN, self.color, (self.x,self.y), self.radius)

    def got_hit(self):
        for bullet in self.gamestate.bullets:
            if abs(bullet.x - self.x) + abs(bullet.y - self.y) < (bullet.radius + self.radius):
                return True
        return False

    def move(self):
        self.x += self.dx
        self.y += self.dy

        if self.x + self.radius >= SCREENWIDTH:
            self.dx = -10
        elif self.x - self.radius <= 0:
            self.dx = 10

class Bullet():
    def __init__(self, x, y, target, friendly = False):
        self.friendly = friendly
        self.radius = 5
        self.x = x
        self.y = y
        self.target_vector = target
        self.color = (0,0,0)

        self.speed = 4

        self.alive = True
        
    def draw(self):
        if self.alive:
            pygame.draw.circle(SCREEN, self.color, (self.x,self.y), self.radius)


    def move(self):
        if self.alive:
            self.x += self.target_vector[0] * self.speed
            self.y += self.target_vector[1] * self.speed

            if self.x > SCREENWIDTH or self.x < 0 or self.y > SCREENHEIGHT or self.y < 0:
                self.alive = False





class GameState:
    def __init__(self):

        
        self.player = Player(self)
        self.enemy = Enemy(self)

        self.angle = self.get_angle_actual(self.player.angle)

        self.time_on_target = 0


        self.start_time = time.time()
        self.time_elapsed = 0


        self.round_time = 10

        self.bullets = []

        self.action_space = [0,1,2,3]


    def visualize(self,angle_dict, vis_type):
        SCREEN.fill((255,255,255),(512,0,1024,512))
        self.enemy.draw()
        self.player.draw()

        if vis_type == "Q":
            for angle, q_value in angle_dict.items():
                if q_value == 0:
                    color = (255,0,0)
                else:
                    red_value = math.ceil(int(255*q_value))
                    blue_value = math.ceil(int(255/q_value))
                    color = (red_value, 0,blue_value)

                self.player.draw_line(angle, color)

                    

        else:
            for angle, action in angle_dict.items():
                # do nothing  ----  black
                if action == 0:
                    color = (0,0,0)

                # fire        ----  Gold
                elif action == 1:
                    color = (255,215,0)
                # clockwise   ----  green
                elif action == 2:
                    color = (0,255,0)

                # counterclockwise - red
                elif action == 3:
                    color = (255,0,0)      
            self.player.draw_line(angle, color)

    def display(self):
        pygame.display.update()
        

    def get_angle_actual(self,angle):
        return angle % 360
    
    def reset(self):
        self.bullets = []
        self.player.angle = random.randint(0,365)
        self.angle = self.get_angle_actual(self.player.angle)
        return torch.FloatTensor([self.angle, self.player.x, self.player.y, self.enemy.x, self.enemy.y, self.enemy.dx]), {}

    def get_next_states(self):
        angle = torch.FloatTensor([self.get_angle_actual(self.player.angle)])
        counter_CW_angle = torch.FloatTensor([self.get_angle_actual(self.player.angle-1)])
        CW_angle = torch.FloatTensor([self.get_angle_actual(self.player.angle+1)])
        # do nothing, shoot, turn left, turn right
        return {0:angle,1:angle,2:CW_angle,3:counter_CW_angle}

    def step(self, input_action, angle_dict=None):
        pygame.event.pump()

        self.enemy.color = self.enemy.original_color

        if self.round_time < 10:
            self.round_time += 0.01

        reward = -1
            
        terminal = False

        # do nothing
        if input_action == 0:
            reward = -1

        # rotate clockwise
        if input_action == 1:
            self.player.rotate_CW()
            reward = -1
        # rotate counter clockwise
        if input_action == 2:
            self.player.rotate_counter_CW()
            reward = -1
        
        # fire        
        if input_action == 3:
            self.player.fire()
            reward = -5
         


        SCREEN.fill((255,255,255),(0,0,512,512))


        laser_on_target = False

        # laser on target
        if self.player.laser_on_target(self.enemy):
            laser_on_target = True
            reward = 1
        else:
            self.time_on_target = 0


        

        if self.enemy.got_hit():
            self.enemy.color = (0,255,0)
            reward = 50
        


        if self.player.got_hit():
            terminal = True
            self.__init__()
            reward = -1


        self.enemy.draw()
        self.player.draw()

        for bullet in self.bullets:
            bullet.move()
            bullet.draw()

        text = pygame.font.SysFont('Times New Roman', 30).render(str(round(self.time_elapsed,2)), False, (0, 0, 0))
        SCREEN.blit(text, (250,50))

        # clear the right side
        SCREEN.fill((255,255,255),(512,0,1024,512))

        
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        self.time_elapsed = time.time() - self.start_time

        if self.time_elapsed > self.round_time:
            if not laser_on_target:
                terminal = True
                self.start_time = time.time()
                
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        self.enemy.move()

        state = torch.FloatTensor([self.player.angle, self.player.x, self.player.y, self.enemy.x, self.enemy.y, self.enemy.dx])
        
        return state, reward, terminal

