import random
import numpy as np
from typing import List, Tuple, Dict, Any

# Các tham số của giải thuật di truyền
POPULATION_SIZE = 80       # Kích thước quần thể
NUM_GENERATIONS = 30       # Số thế hệ
MUTATION_RATE = 0.1        # Tỷ lệ đột biến
CROSSOVER_RATE = 0.8       # Tỷ lệ lai ghép
TOURNAMENT_SIZE = 5        # Kích thước giải đấu cho phương pháp chọn lọc
ACTION_SEQUENCE_LENGTH = 200  # Độ dài chuỗi hành động

# Định nghĩa các hành động có thể của Mario
class MarioAction:
    NOTHING = 0      # Không làm gì
    MOVE_RIGHT = 1   # Di chuyển sang phải
    MOVE_LEFT = 2    # Di chuyển sang trái
    JUMP = 3         # Nhảy
    JUMP_RIGHT = 4   # Nhảy sang phải
    JUMP_LEFT = 5    # Nhảy sang trái
    RUN_RIGHT = 6    # Chạy sang phải
    RUN_LEFT = 7     # Chạy sang trái
    DUCK = 8         # Ngồi xuống
    USE_ITEM = 9     # Sử dụng vật phẩm (nếu có)

    @staticmethod
    def get_all_actions():
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Giả lập môi trường Super Mario
class MarioEnvironment:
    def __init__(self, level_id: int = 1):
        self.level_id = level_id
        self.reset()
    
    def reset(self):
        self.score = 0
        self.time_left = 400
        self.mario_position = (40, 200)  # (x, y)
        self.mario_state = "small"  # small, super, fire, etc.
        self.is_game_over = False
        self.reached_flag = False
        return self.get_state()
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "time_left": self.time_left,
            "position": self.mario_position,
            "state": self.mario_state,
            "game_over": self.is_game_over,
            "reached_flag": self.reached_flag
        }
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool]:  # Thực hiện một hành động và trả về trạng thái mới, phần thưởng và trạng thái kết thúc
        old_position = self.mario_position
        old_score = self.score
        
        # Cập nhật vị trí dựa trên hành động
        x, y = self.mario_position
        if action == MarioAction.MOVE_RIGHT or action == MarioAction.RUN_RIGHT:
            x += 5 if action == MarioAction.MOVE_RIGHT else 8
        elif action == MarioAction.MOVE_LEFT or action == MarioAction.RUN_LEFT:
            x -= 5 if action == MarioAction.MOVE_LEFT else 8
        elif action == MarioAction.JUMP:
            y -= 10
        elif action == MarioAction.JUMP_RIGHT:
            x += 5
            y -= 10
        elif action == MarioAction.JUMP_LEFT:
            x -= 5
            y -= 10
        
        self.mario_position = (x, y)
        
        # Mô phỏng cơ chế gravity
        self.mario_position = (x, min(y + 2, 200))
        
        # Mô phỏng cập nhật điểm số
        self.score += random.randint(0, 100) if random.random() < 0.05 else 0
        
        # Giảm thời gian
        self.time_left = max(0, self.time_left - 1)
        
        # Kiểm tra kết thúc game
        self.is_game_over = (self.time_left <= 0) or (random.random() < 0.001)
        self.reached_flag = (x > 1000) or (random.random() < 0.001)
        
        # Tính toán phần thưởng
        reward = 0
        reward += (self.score - old_score)  # Phần thưởng cho điểm số
        reward += (x - old_position[0])     # Phần thưởng cho việc di chuyển về phía trước
        
        if self.reached_flag:
            reward += 1000 + self.time_left * 10  # Phần thưởng lớn cho việc hoàn thành màn chơi
        
        if self.is_game_over and not self.reached_flag:
            reward -= 500  # Hình phạt cho việc kết thúc game mà không hoàn thành
        
        done = self.is_game_over or self.reached_flag
        
        return self.get_state(), reward, done
    
    def render(self):
        print(f"Score: {self.score}, Time: {self.time_left}, Position: {self.mario_position}")

# Định nghĩa cá thể trong quần thể
class Individual:
    def __init__(self, action_sequence: List[int] = None):  # Khởi tạo một cá thể với chuỗi hành động ngẫu nhiên hoặc xác định
        if action_sequence is None:
            all_actions = MarioAction.get_all_actions()
            self.action_sequence = [random.choice(all_actions) for _ in range(ACTION_SEQUENCE_LENGTH)]
        else:
            self.action_sequence = action_sequence
        
        self.fitness = None  # Sẽ được đánh giá sau
    
    def evaluate(self, env: MarioEnvironment) -> float:
        state = env.reset()
        total_reward = 0
        max_x_position = 40
        
        for action in self.action_sequence:
            next_state, reward, done = env.step(action)
            total_reward += reward
            max_x_position = max(max_x_position, next_state["position"][0])
            
            if done:
                break
        
        # Tính toán fitness dựa trên phần thưởng, điểm số và tiến độ
        self.fitness = total_reward + next_state["score"] + max_x_position
        
        if next_state["reached_flag"]:
            self.fitness += 10000  # Thưởng thêm cho việc hoàn thành màn chơi
        
        return self.fitness

# Lớp chính cho giải thuật di truyền
class GeneticAlgorithm:
    def __init__(self):
        self.population = []
        self.best_individual = None
        self.env = MarioEnvironment()
    
    def initialize_population(self):
        """Khởi tạo quần thể ban đầu."""
        self.population = [Individual() for _ in range(POPULATION_SIZE)]
    
    def evaluate_population(self):
        """Đánh giá toàn bộ quần thể."""
        for individual in self.population:
            individual.evaluate(self.env)
        
        # Cập nhật cá thể tốt nhất
        best_in_generation = max(self.population, key=lambda ind: ind.fitness)
        if self.best_individual is None or best_in_generation.fitness > self.best_individual.fitness:
            self.best_individual = Individual(best_in_generation.action_sequence.copy())
            self.best_individual.fitness = best_in_generation.fitness
    
    def selection(self) -> List[Individual]:
        selected = []
        
        for _ in range(POPULATION_SIZE):
            # Chọn ngẫu nhiên TOURNAMENT_SIZE cá thể từ quần thể
            tournament = random.sample(self.population, TOURNAMENT_SIZE)
            # Chọn cá thể có fitness cao nhất từ giải đấu
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:  # Lai ghép hai cá thể để tạo ra hai cá thể con
        if random.random() > CROSSOVER_RATE:
            # Nếu không lai ghép, trả về bản sao của cha mẹ
            return (
                Individual(parent1.action_sequence.copy()),
                Individual(parent2.action_sequence.copy())
            )
        
        # Lai ghép một điểm
        crossover_point = random.randint(1, ACTION_SEQUENCE_LENGTH - 1)
        
        child1_sequence = parent1.action_sequence[:crossover_point] + parent2.action_sequence[crossover_point:]
        child2_sequence = parent2.action_sequence[:crossover_point] + parent1.action_sequence[crossover_point:]
        
        return Individual(child1_sequence), Individual(child2_sequence)
    
    def mutation(self, individual: Individual) -> Individual:  # Đột biến cá thể
        all_actions = MarioAction.get_all_actions()
        mutated_sequence = individual.action_sequence.copy()
        
        for i in range(len(mutated_sequence)):
            if random.random() < MUTATION_RATE:
                # Thay đổi hành động tại vị trí i
                mutated_sequence[i] = random.choice(all_actions)
        
        return Individual(mutated_sequence)
    
    def evolve(self):
        # Chọn lọc
        selected = self.selection()
        
        # Tạo quần thể mới thông qua lai ghép và đột biến
        new_population = []
        
        while len(new_population) < POPULATION_SIZE:
            # Chọn ngẫu nhiên hai cha mẹ
            parent1, parent2 = random.sample(selected, 2)
            
            # Lai ghép
            child1, child2 = self.crossover(parent1, parent2)
            
            # Đột biến
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            
            # Thêm vào quần thể mới
            new_population.append(child1)
            new_population.append(child2)
        
        # Cắt bớt nếu quần thể quá lớn
        self.population = new_population[:POPULATION_SIZE]
    
    def train(self):
        print("Bắt đầu huấn luyện giải thuật di truyền...")
        
        # Khởi tạo quần thể
        self.initialize_population()
        
        # Lặp qua các thế hệ
        for generation in range(NUM_GENERATIONS):
            # Đánh giá quần thể
            self.evaluate_population()
            
            # In thông tin về thế hệ hiện tại
            best_in_generation = max(self.population, key=lambda ind: ind.fitness)
            avg_fitness = sum(ind.fitness for ind in self.population) / POPULATION_SIZE
            
            print(f"Thế hệ {generation + 1}/{NUM_GENERATIONS}:")
            print(f"  Fitness tốt nhất: {best_in_generation.fitness}")
            print(f"  Fitness trung bình: {avg_fitness}")
            print(f"  Fitness tốt nhất từ trước đến nay: {self.best_individual.fitness}")
            
            # Tiến hóa quần thể
            self.evolve()
        
        # Đánh giá lần cuối
        self.evaluate_population()
        
        print("Kết thúc huấn luyện!")
        print(f"Fitness tốt nhất đạt được: {self.best_individual.fitness}")
    
    def test_best_individual(self, render: bool = True):  # render: Có hiển thị quá trình chơi hay không
        if self.best_individual is None:
            print("Không có cá thể nào để kiểm tra. Hãy huấn luyện trước!")
            return
        
        print("Kiểm tra cá thể tốt nhất...")
        
        state = self.env.reset()
        total_reward = 0
        steps = 0
        
        for action in self.best_individual.action_sequence:
            next_state, reward, done = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                self.env.render()
            
            if done:
                break
        
        print(f"Kết quả kiểm tra:")
        print(f"  Số bước: {steps}")
        print(f"  Tổng phần thưởng: {total_reward}")
        print(f"  Điểm số đạt được: {next_state['score']}")
        print(f"  Vị trí cuối cùng: {next_state['position']}")
        print(f"  Hoàn thành màn chơi: {'Có' if next_state['reached_flag'] else 'Không'}")

# Lớp mở rộng với các chiến lược tiên tiến hơn
class AdvancedMarioAI(GeneticAlgorithm):
    def __init__(self):
        super().__init__()
        # Thêm các tham số mới cho phiên bản nâng cao
        self.elite_size = int(POPULATION_SIZE * 0.1)  # 10% cá thể ưu tú
        
    def elitism(self) -> List[Individual]:  # Chọn những cá thể tốt nhất để giữ lại thế hệ
        # Sắp xếp quần thể theo fitness giảm dần
        sorted_population = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        # Chọn n cá thể tốt nhất
        return [Individual(ind.action_sequence.copy()) for ind in sorted_population[:self.elite_size]]
    
    def adaptive_mutation(self, individual: Individual, avg_fitness: float) -> Individual:
        all_actions = MarioAction.get_all_actions()
        mutated_sequence = individual.action_sequence.copy()
        
        # Điều chỉnh tỷ lệ đột biến dựa trên fitness
        if individual.fitness is None:
            mutation_rate = MUTATION_RATE  # Sử dụng tỷ lệ đột biến mặc định
        # Cá thể có fitness cao có tỷ lệ đột biến thấp hơn
        elif individual.fitness > avg_fitness:
            mutation_rate = MUTATION_RATE * 0.5  # Giảm một nửa tỷ lệ đột biến
        else:
            mutation_rate = MUTATION_RATE * 1.5  # Tăng 1.5 lần tỷ lệ đột biến
        
        for i in range(len(mutated_sequence)):
            if random.random() < mutation_rate:
                # Thay đổi hành động tại vị trí i
                mutated_sequence[i] = random.choice(all_actions)
        
        return Individual(mutated_sequence)
    
    def evolve(self):
        """Tiến hóa quần thể qua một thế hệ với chiến lược nâng cao."""
        # Tính toán fitness trung bình
        avg_fitness = sum(ind.fitness for ind in self.population) / POPULATION_SIZE
        
        # Áp dụng elitism - giữ lại những cá thể tốt nhất
        elites = self.elitism()
        
        # Chọn lọc
        selected = self.selection()
        
        # Tạo quần thể mới thông qua lai ghép và đột biến thích ứng
        new_population = list(elites)  # Bắt đầu với các cá thể ưu tú
        
        while len(new_population) < POPULATION_SIZE:
            # Chọn ngẫu nhiên hai cha mẹ
            parent1, parent2 = random.sample(selected, 2)
            
            # Lai ghép
            child1, child2 = self.crossover(parent1, parent2)
            
            # Đột biến thích ứng
            child1 = self.adaptive_mutation(child1, avg_fitness)
            child2 = self.adaptive_mutation(child2, avg_fitness)
            
            # Thêm vào quần thể mới
            new_population.append(child1)
            new_population.append(child2)
        
        # Cắt bớt nếu quần thể quá lớn
        self.population = new_population[:POPULATION_SIZE]

# Chạy hệ thống AI
if __name__ == "__main__":
    # Sử dụng giải thuật di truyền nâng cao
    mario_ai = AdvancedMarioAI()
    
    # Huấn luyện AI
    mario_ai.train()
    
    # Kiểm tra cá thể tốt nhất
    mario_ai.test_best_individual(render=True)