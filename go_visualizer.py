import pygame
import numpy as np
from go_rl_example import GoBoard, GoAgent

class GoVisualizer:
    def __init__(self, board_size=9, cell_size=60):
        pygame.init()
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 40
        self.window_size = board_size * cell_size + 2 * self.margin
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BOARD_COLOR = (220, 179, 92)
        self.LINE_COLOR = (0, 0, 0)
        self.TEXT_COLOR = (0, 0, 0)
        
        # Initialize window
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Go Game Visualization")
        
        # Initialize font
        self.font = pygame.font.Font(None, 36)
        
    def draw_board(self, board):
        # Fill background
        self.screen.fill(self.BOARD_COLOR)
        
        # Draw grid lines
        for i in range(self.board_size + 1):  # Changed to +1 to draw the last line
            # Vertical lines
            start_pos = (self.margin + i * self.cell_size, self.margin)
            end_pos = (self.margin + i * self.cell_size, self.window_size - self.margin)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_pos, end_pos, 2)
            
            # Horizontal lines
            start_pos = (self.margin, self.margin + i * self.cell_size)
            end_pos = (self.window_size - self.margin, self.margin + i * self.cell_size)
            pygame.draw.line(self.screen, self.LINE_COLOR, start_pos, end_pos, 2)
        
        # Draw stones
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i, j] != 0:
                    center = (self.margin + j * self.cell_size, 
                            self.margin + i * self.cell_size)
                    color = self.BLACK if board[i, j] == 1 else self.WHITE
                    pygame.draw.circle(self.screen, color, center, self.cell_size // 2 - 2)
    
    def draw_score(self, black_score, white_score, move_count=0):
        # Draw score text
        black_text = self.font.render(f"Black: {black_score}", True, self.TEXT_COLOR)
        white_text = self.font.render(f"White: {white_score}", True, self.TEXT_COLOR)
        move_text = self.font.render(f"Moves: {move_count}", True, self.TEXT_COLOR)
        
        self.screen.blit(black_text, (10, 10))
        self.screen.blit(white_text, (self.window_size - 150, 10))
        self.screen.blit(move_text, (self.window_size // 2 - 50, 10))
    
    def update(self, board, black_score, white_score, move_count=0):
        self.draw_board(board)
        self.draw_score(black_score, white_score, move_count)
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

def play_match(agent1, agent2, delay=0.5, epsilon=0.1):
    board = GoBoard()
    visualizer = GoVisualizer()
    state = board.reset()
    done = False
    move_count = 0
    last_move = None
    repeated_moves = 0
    
    try:
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break
            
            # Agent 1's turn (Black)
            action = agent1.get_action(state, epsilon=epsilon)  # Add exploration
            if action == -1:  # Pass move
                x, y = -1, -1
                print(f"Black passes")
            else:
                x, y = action // 9, action % 9
                print(f"Black plays at ({x}, {y})")
                
            # Check for repeated moves
            if (x, y) == last_move:
                repeated_moves += 1
                if repeated_moves >= 3:
                    print("Too many repeated moves, forcing pass")
                    x, y = -1, -1
            else:
                repeated_moves = 0
                
            if board.make_move(x, y):
                state = board.get_state()
                black_score = np.sum(board.board == 1)
                white_score = np.sum(board.board == -1)
                move_count += 1
                visualizer.update(board.board, black_score, white_score, move_count)
                pygame.time.wait(int(delay * 1000))
                last_move = (x, y)
            
            # Agent 2's turn (White)
            action = agent2.get_action(state, epsilon=epsilon)  # Add exploration
            if action == -1:  # Pass move
                x, y = -1, -1
                print(f"White passes")
            else:
                x, y = action // 9, action % 9
                print(f"White plays at ({x}, {y})")
                
            # Check for repeated moves
            if (x, y) == last_move:
                repeated_moves += 1
                if repeated_moves >= 3:
                    print("Too many repeated moves, forcing pass")
                    x, y = -1, -1
            else:
                repeated_moves = 0
                
            if board.make_move(x, y):
                state = board.get_state()
                black_score = np.sum(board.board == 1)
                white_score = np.sum(board.board == -1)
                move_count += 1
                visualizer.update(board.board, black_score, white_score, move_count)
                pygame.time.wait(int(delay * 1000))
                last_move = (x, y)
            
            # Check if game is over
            done = board.is_game_over()
            if done:
                print(f"Game over after {move_count} moves")
                print(f"Final score - Black: {black_score}, White: {white_score}")
        
        # Show final score
        black_score = np.sum(board.board == 1)
        white_score = np.sum(board.board == -1)
        visualizer.update(board.board, black_score, white_score, move_count)
        pygame.time.wait(2000)  # Show final state for 2 seconds
        
    finally:
        visualizer.close()

if __name__ == "__main__":
    # Load trained agents
    agent1 = GoAgent()
    agent2 = GoAgent()
    
    # Try to load pre-trained models
    try:
        agent1.load("trained_agents/agent1_final.pth")
        agent2.load("trained_agents/agent2_final.pth")
        print("Loaded pre-trained models")
    except:
        print("No pre-trained models found, using untrained agents")
    
    # Play a match with some exploration
    play_match(agent1, agent2, delay=0.5) 