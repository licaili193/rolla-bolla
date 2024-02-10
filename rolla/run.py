import pygame
import pymunk
import pymunk.pygame_util
import math

import model
import level


def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption('Simplified Humanoid Skeleton')
    clock = pygame.time.Clock()

    space = pymunk.Space()
    space.gravity = (0, 981)

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    body_list = model.create_humanoid(space)
    angular_impulse_list = [0] * len(body_list)

    # Add platform at the bottom

    platform = level.create_platform(space, (600 / 2, 600 - 50), 600, 10)
    plank = level.create_plank(space, (600 / 2, 600 - 100), 150, 5, 2)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:  # Press number keys to apply angular impulses
                    angular_impulse_list[1] = 1000
                elif event.key == pygame.K_2:
                    angular_impulse_list[2] = 1000
                elif event.key == pygame.K_3:
                    angular_impulse_list[3] = 1000
                elif event.key == pygame.K_4:
                    angular_impulse_list[4] = 1000
                elif event.key == pygame.K_5:
                    angular_impulse_list[5] = 1000
                elif event.key == pygame.K_6:
                    angular_impulse_list[6] = 1000
                elif event.key == pygame.K_7:
                    angular_impulse_list[7] = 1000
                elif event.key == pygame.K_8:
                    angular_impulse_list[8] = 1000

        screen.fill((255, 255, 255))

        # Begin simulation step

        # Apply angular impulses to bodies
        model.apply_angular_impulse_to_bodies(body_list, angular_impulse_list)

        # Reset angular impulses
        angular_impulse_list = [0] * len(body_list)
        
        # End simulation step

        space.step(1 / 50.0)
        space.debug_draw(draw_options)
        pygame.display.flip()
        clock.tick(50)

    pygame.quit()


if __name__ == '__main__':
    main()
