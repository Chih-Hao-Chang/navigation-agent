"""
Author: Dr Zhibin Liao
Organisation: School of Computer Science and Mathematical Sciences, University of Adelaide
Date: 03-Apr-2025
Description: This Python script implements a simple agent which is controllable by human.

The script is a part of Assignment 2 made for the course COMP SCI 3007/7059/7659 Artificial Intelligence for the year
of 2025. Public distribution of this source code is strictly forbidden.
"""
import pygame
import sys

class HumanAgent:
    def __init__(self, show_screen=False):
        self.show_screen = show_screen
        # self.score = -1
        # self.next_pipe = None
        # self.next_pipe2 = None
        # self.next_pipe3 = None
        # self.is_first_jump = True
        self.steps = 0
        
    def choose_action(self, state, action_table):
        # if state['pipes']:
        #     if state['bird_x'] < state['pipes'][0]['x'] + state['pipes'][0]['width']:
        #         self.next_pipe = state['pipes'][0]
        #     else:
        #         self.next_pipe = state['pipes'][1]
            # print((self.next_pipe['x'] + 40 - state['bird_x']) / 108)
            
        # if state['bird_x'] + state['bird_width'] state['pipes'][0]['x']:
        #     sys.exit(0)
        for event in pygame.event.get():
            # if state['pipes']: print("pipes:", len(state['pipes']))
            if event.type == pygame.QUIT:
                return action_table['quit_game']
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                # if state['mileage'] > : self.next_pipe: print(state['bird_y'] + state['bird_height'] - self.next_pipe['bottom'])
                # if self.next_pipe: print((self.next_pipe['x'] + 40 - state['bird_x']) / 108, (state['bird_y'] - self.next_pipe['top'] + 350) / 800)
                # if self.next_pipe: print((state['bird_y'] - self.next_pipe['top'] + 350) / 800)
                # if self.next_pipe: print(state['bird_y'] - self.next_pipe['top'])
                return action_table['jump']

        # no keystroke, so do nothing
        # print(list(action_table.values())[:-1])
        # if len(state['pipes'])>1: print(state['pipes'][1]['x'] - state['pipes'][0]['x'])
        # if state['score'] > self.score:
        #     print("Score! mileage:", state['mileage'])
        #     self.score = state['score']        
        return action_table['do_nothing']
        
    def receive_after_action_observation(self, state, action_table):
        self.steps += 1
        pass
