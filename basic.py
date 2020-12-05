# Simple pygame program

# Import and initialize the pygame library
import pygame
from snake import Snake
from time import sleep
pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([500, 500])
snake = Snake(pygame, screen, 240, 240)


# Run until the user asks to quit
running = True
while running:

	# Did the user click the window close button?
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT or event.key == ord('a'):
				snake.control(-1, 0)
			if event.key == pygame.K_RIGHT or event.key == ord('d'):
				snake.control(1, 0)
			if event.key == pygame.K_UP or event.key == ord('w'):
				snake.control(0, -1)
			if event.key == pygame.K_DOWN or event.key == ord('s'):
				snake.control(0, 1)

		'''
		if event.type == pygame.KEYUP:
			if event.key == pygame.K_LEFT or event.key == ord('a'):
				snake.moveX_stop()
			if event.key == pygame.K_RIGHT or event.key == ord('d'):
				snake.moveX_stop()
			if event.key == pygame.K_UP or event.key == ord('w'):
				snake.moveY_stop()
			if event.key == pygame.K_DOWN or event.key == ord('s'):
				snake.moveY_stop()
		'''
	# Fill the background with white
	screen.fill((255, 255, 255))

	#snake.draw()
	snake.update()
	snake.draw()
	# Flip the display
	pygame.display.flip()
	sleep(.01)

# Done! Time to quit.
pygame.quit()