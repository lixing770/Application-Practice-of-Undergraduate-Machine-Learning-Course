import pygame
import random
import sys

# 游戏设置
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
CELL_SIZE = 20
CELL_WIDTH = WINDOW_WIDTH // CELL_SIZE
CELL_HEIGHT = WINDOW_HEIGHT // CELL_SIZE

# 颜色设置
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DARK_GREEN = (0, 155, 0)
DARK_GRAY = (40, 40, 40)

# 方向
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

# FPS
FPS = 15

def drawGrid(surface):
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(surface, DARK_GRAY, (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(surface, DARK_GRAY, (0, y), (WINDOW_WIDTH, y))

def drawSnake(surface, snakeCoords):
    for coord in snakeCoords:
        x = coord['x'] * CELL_SIZE
        y = coord['y'] * CELL_SIZE
        snakeSegmentRect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, DARK_GREEN, snakeSegmentRect)
        innerSegmentRect = pygame.Rect(x + 4, y + 4, CELL_SIZE - 8, CELL_SIZE - 8)
        pygame.draw.rect(surface, GREEN, innerSegmentRect)

def drawFood(surface, coord):
    x = coord['x'] * CELL_SIZE
    y = coord['y'] * CELL_SIZE
    foodRect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(surface, RED, foodRect)

def getRandomLocation():
    return {'x': random.randint(0, CELL_WIDTH - 1), 'y': random.randint(0, CELL_HEIGHT - 1)}

def terminate():
    pygame.quit()
    sys.exit()

def checkForKeyPress():
    if len(pygame.event.get(pygame.QUIT)) > 0:
        terminate()
    keyUpEvents = pygame.event.get(pygame.KEYUP)
    if len(keyUpEvents) == 0:
        return None
    if keyUpEvents[0].key == pygame.K_ESCAPE:
        terminate()
    return keyUpEvents[0].key

def showGameOverScreen(surface):
    font = pygame.font.Font(None, 48)
    gameOverSurf = font.render('Game Over', True, WHITE)
    gameOverRect = gameOverSurf.get_rect()
    gameOverRect.midtop = (WINDOW_WIDTH / 2, 10)
    surface.blit(gameOverSurf, gameOverRect)

    pygame.display.update()
    pygame.time.wait(500)
    checkForKeyPress()

    while True:
        if checkForKeyPress():
            pygame.event.get()
            return

def runGame():
    # 初始化游戏变量
    startX = random.randint(5, CELL_WIDTH - 6)
    startY = random.randint(5, CELL_HEIGHT - 6)
    snakeCoords = [{'x': startX, 'y': startY},
                   {'x': startX - 1, 'y': startY},
                   {'x': startX - 2, 'y': startY}]
    direction = RIGHT

    food = getRandomLocation()

    while True:  # 游戏主循环
        for event in pygame.event.get():  # 事件处理
            if event.type == pygame.QUIT:
                terminate()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and direction != RIGHT:
                    direction = LEFT
                elif event.key == pygame.K_RIGHT and direction != LEFT:
                    direction = RIGHT
                elif event.key == pygame.K_UP and direction != DOWN:
                    direction = UP
                elif event.key == pygame.K_DOWN and direction != UP:
                    direction = DOWN

        # 更新蛇的位置
        if direction == UP:
            newHead = {'x': snakeCoords[0]['x'], 'y': snakeCoords[0]['y'] - 1}
        elif direction == DOWN:
            newHead = {'x': snakeCoords[0]['x'], 'y': snakeCoords[0]['y'] + 1}
        elif direction == LEFT:
            newHead = {'x': snakeCoords[0]['x'] - 1, 'y': snakeCoords[0]['y']}
        elif direction == RIGHT:
            newHead = {'x': snakeCoords[0]['x'] + 1, 'y': snakeCoords[0]['y']}

        snakeCoords.insert(0, newHead)

        # 检查蛇是否吃到食物
        if snakeCoords[0] == food:
            food = getRandomLocation()  # 生成新的食物
        else:
            snakeCoords.pop()  # 移除蛇尾

        # 检查是否撞到自己或边界
        if (snakeCoords[0]['x'] == -1 or snakeCoords[0]['x'] == CELL_WIDTH or
            snakeCoords[0]['y'] == -1 or snakeCoords[0]['y'] == CELL_HEIGHT):
            return  # 游戏结束
        for snakeBody in snakeCoords[1:]:
            if snakeBody == snakeCoords[0]:
                return  # 游戏结束

        # 绘制屏幕
        surface.fill(BLACK)
        drawGrid(surface)
        drawSnake(surface, snakeCoords)
        drawFood(surface, food)
        pygame.display.update()
        fpsClock.tick(FPS)

def main():
    global fpsClock, surface

    pygame.init()
    fpsClock = pygame.time.Clock()
    surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('贪吃蛇')

    showGameOverScreen(surface)
    while True:
        runGame()
        showGameOverScreen(surface)

if __name__ == '__main__':
    main()
