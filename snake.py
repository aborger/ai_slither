import pygame as pgame

class Snake(pgame.sprite.Sprite):
	def __init__(self, pygame, screen, startX, startY):
		pygame.sprite.Sprite.__init__(self)
		self.pygame = pygame
		self.screen = screen
		self.length = 50
		self.speed = 3
		self.size = 10
		self.color = (255, 0, 0)
		self.movex = 1
		self.movey = 0
		self.circs = []
		for i in range(0, self.length):
			temp = pygame.Rect(startX, startY + i*self.size, self.size, self.size)
			self.rects.append(temp)
			
	def control(self, x, y):
		self.movex = x
		self.movey = y
		
	def draw(self):
		for rect in self.rects:
			self.pygame.draw.rect(self.screen, self.color, rect, 1)
		
	def update(self):
		next = self.rects[0].copy()
		next = next.move(self.movex * self.speed, self.movey * self.speed)
		for i in range(0, len(self.rects)):
			curr = self.rects[i]
			self.rects[i] = next
			next = curr
			