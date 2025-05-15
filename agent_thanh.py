import numpy as np
import heapq

# Hàm A* để tìm đường đi ngắn nhất
def run_astar(map_grid, start, goal):
    n_rows = len(map_grid)
    n_cols = len(map_grid[0])
    
    # Các hướng di chuyển có thể (lên, xuống, trái, phải) và hành động tương ứng
    # Sẽ trả về hành động đầu tiên cần thực hiện để đi theo đường ngắn nhất
    # movements = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    # action_keys = ['U', 'D', 'L', 'R']
    
    # Định nghĩa các hướng di chuyển (dx, dy) và hành động tương ứng
    # Ưu tiên các hướng nhất định có thể ảnh hưởng đến kết quả nếu có nhiều đường đi ngắn nhất bằng nhau
    # Thứ tự này ưu tiên U, D, L, R. Quan trọng là nó phải nhất quán.
    possible_moves = [
        ('U', (-1, 0)),  # Lên
        ('D', (1, 0)),   # Xuống
        ('L', (0, -1)),  # Trái
        ('R', (0, 1))    # Phải
    ]

    # Hàm heuristic (Manhattan distance)
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Hàng đợi ưu tiên cho A*
    # (f_score, g_score, (row, col), path_taken_actions)
    # g_score là chi phí từ điểm bắt đầu đến nút hiện tại
    # f_score là g_score + heuristic(current_node, goal)
    # path_taken_actions là danh sách các hành động đã thực hiện
    pq = [(heuristic(start, goal), 0, start, [])]  # f_score, g_score, position, path_actions
    
    # Lưu trữ chi phí g_score tốt nhất cho mỗi nút đã khám phá
    g_scores = {start: 0}
    
    # Theo dõi các nút đã được xử lý hoàn toàn (đã lấy ra khỏi hàng đợi ưu tiên và mở rộng)
    # hoặc các nút đang ở trong hàng đợi ưu tiên nhưng có thể tìm thấy đường đi tốt hơn sau này
    came_from = {start: None} # Để truy vết đường đi (không cần thiết nếu chỉ trả về hành động đầu tiên)

    while pq:
        f_score, current_g_score, current_pos, current_path_actions = heapq.heappop(pq)

        # Nếu chi phí hiện tại để đến current_pos lớn hơn một chi phí đã biết trước đó, bỏ qua
        if current_g_score > g_scores.get(current_pos, float('inf')):
            continue

        # Nếu đã đến đích
        if current_pos == goal:
            if not current_path_actions: # Nếu đích là điểm bắt đầu
                return 'S', 0 # Đứng yên, chi phí 0
            return current_path_actions[0], len(current_path_actions) # Trả về hành động đầu tiên và độ dài đường đi

        # Khám phá các ô lân cận
        for action_char, (dr, dc) in possible_moves:
            next_pos = (current_pos[0] + dr, current_pos[1] + dc)

            # Kiểm tra tính hợp lệ của ô tiếp theo
            if 0 <= next_pos[0] < n_rows and 0 <= next_pos[1] < n_cols and map_grid[next_pos[0]][next_pos[1]] == 0:
                new_g_score = current_g_score + 1 # Chi phí mỗi bước là 1
                
                # Nếu tìm thấy đường đi tốt hơn đến next_pos
                if new_g_score < g_scores.get(next_pos, float('inf')):
                    g_scores[next_pos] = new_g_score
                    new_f_score = new_g_score + heuristic(next_pos, goal)
                    new_path_actions = current_path_actions + [action_char]
                    heapq.heappush(pq, (new_f_score, new_g_score, next_pos, new_path_actions))
                    came_from[next_pos] = current_pos


    return 'S', float('inf') # Nếu không tìm thấy đường đi, đứng yên


class Agents:
    def __init__(self):
        # Hàm khởi tạo, thiết lập các thuộc tính ban đầu cho đối tượng Agents
        self.n_robots = 0  # Số lượng robot
        self.map_grid = []  # Bản đồ môi trường (dạng lưới)
        self.robots_info = []  # Danh sách thông tin của các robot
        # Mỗi robot có thông tin: {'pos': (r, c), 'carrying_package_id': None hoặc id gói hàng, 
        # 'target_package_id': None hoặc id gói hàng, 'phase': 'idle'/'to_pickup'/'to_dropoff'}
        self.packages_info = {}  # Thông tin các gói hàng
        # Dạng: {package_id: {'start_pos': (r, c), 'target_pos': (r, c), 'deadline': t, 
        # 'status': 'waiting'/'assigned'/'in_transit'/'delivered', 'assigned_robot': None hoặc id robot, 'start_time': t}}
        self.current_time = 0  # Thời gian hiện tại
        self.is_initialized = False  # Trạng thái khởi tạo

    def init_agents(self, state):
        # Khởi tạo trạng thái ban đầu của các robot và gói hàng từ dữ liệu đầu vào
        self.current_time = state['time_step']  # Cập nhật thời gian hiện tại
        self.map_grid = state['map']  # Cập nhật bản đồ
        self.n_robots = len(state['robots'])  # Số lượng robot

        if not self.is_initialized:
            # Khởi tạo thông tin robot
            self.robots_info = [{'pos': (r - 1, c - 1),  # Chuyển đổi sang chỉ số 0-indexed
                                 'carrying_package_id': None,
                                 'target_package_id': None,
                                 'phase': 'idle'} for r, c, carrying in state['robots']]
            self.is_initialized = True

        # Cập nhật thông tin gói hàng mới xuất hiện tại thời điểm này
        for pkg_data in state['packages']:
            pkg_id, sr, sc, tr, tc, start_time, deadline = pkg_data
            if pkg_id not in self.packages_info:  # Chỉ thêm nếu chưa có
                self.packages_info[pkg_id] = {
                    'id': pkg_id,
                    'start_pos': (sr - 1, sc - 1),
                    'target_pos': (tr - 1, tc - 1),
                    'deadline': deadline,
                    'status': 'waiting',  # Trạng thái ban đầu
                    'assigned_robot': None,
                    'start_time': start_time
                }

    def _update_internal_state(self, state):
        # Cập nhật trạng thái nội bộ của các robot và gói hàng từ trạng thái môi trường
        self.current_time = state['time_step']  # Cập nhật thời gian hiện tại

        # Cập nhật vị trí và trạng thái của robot
        for i in range(self.n_robots):
            # Lấy thông tin robot từ trạng thái môi trường
            r, c, carrying_id = state['robots'][i]
            current_robot_pos = (r - 1, c - 1)

            # Nếu robot vừa nhặt hàng
            if self.robots_info[i]['carrying_package_id'] is None and carrying_id != 0:
                self.robots_info[i]['carrying_package_id'] = carrying_id
                # Chuyển trạng thái robot sang giai đoạn giao hàng
                self.robots_info[i]['phase'] = 'to_dropoff'
                # Chuyển trạng thái gói hàng sang đang vận chuyển
                if carrying_id in self.packages_info:
                    self.packages_info[carrying_id]['status'] = 'in_transit'

            # Nếu robot vừa thả hàng
            elif self.robots_info[i]['carrying_package_id'] is not None and carrying_id == 0:
                dropped_pkg_id = self.robots_info[i]['carrying_package_id']
                # Chuyển trạng thái gói hàng sang đã giao
                # Và bỏ gán robot cho gói hàng
                if dropped_pkg_id in self.packages_info:
                    self.packages_info[dropped_pkg_id]['status'] = 'delivered'
                    self.packages_info[dropped_pkg_id]['assigned_robot'] = None
                # Đặt lại trạng thái robot
                self.robots_info[i]['carrying_package_id'] = None
                self.robots_info[i]['target_package_id'] = None
                self.robots_info[i]['phase'] = 'idle'

            self.robots_info[i]['pos'] = current_robot_pos

        # Cập nhật thông tin gói hàng mới (nếu có)
        for pkg_data in state['packages']:
            pkg_id, sr, sc, tr, tc, start_time, deadline = pkg_data
            # Nếu gói hàng có trong môi trường nhưng chưa có trong danh sách gói hàng
            # Thêm gói hàng mới vào danh sách
            if pkg_id not in self.packages_info or self.packages_info[pkg_id]['status'] == 'None':
                self.packages_info[pkg_id] = {
                    'id': pkg_id,
                    'start_pos': (sr - 1, sc - 1),
                    'target_pos': (tr - 1, tc - 1),
                    'deadline': deadline,
                    'status': 'waiting',  # Sẵn sàng để được gán
                    'assigned_robot': None,
                    'start_time': start_time
                }
            # Đảm bảo các gói hàng đã xuất hiện (start_time <= current_time) và đang chờ thì có status 'waiting'
            elif self.packages_info[pkg_id]['start_time'] <= self.current_time and self.packages_info[pkg_id]['status'] not in ['assigned', 'in_transit', 'delivered']:
                self.packages_info[pkg_id]['status'] = 'waiting'

    def _calculate_score(self, robot_idx, package_id):
        '''
        Tính toán điểm số cho một gói hàng dựa trên khoảng cách, thời gian và phần thưởng
        Điểm số càng cao thì gói hàng càng được ưu tiên hơn
        Điểm số = phần thưởng - (hệ số * thời gian) + chi phí di chuyển
        Điểm số sẽ giảm mạnh nếu không thể giao kịp hoặc rất sát hạn
        '''
        robot = self.robots_info[robot_idx]
        package = self.packages_info[package_id]

        # Thời gian ước tính để robot đến điểm lấy hàng
        _, time_to_pickup = run_astar(self.map_grid, robot['pos'], package['start_pos'])
        if time_to_pickup == float('inf'):  # Không thể đến được
            return -float('inf')

        # Thời gian ước tính từ điểm lấy hàng đến điểm giao hàng
        _, time_to_deliver_from_pickup = run_astar(self.map_grid, package['start_pos'], package['target_pos'])
        if time_to_deliver_from_pickup == float('inf'):  # Không thể giao được
            return -float('inf')

        total_estimated_time = time_to_pickup + time_to_deliver_from_pickup
        estimated_delivery_time = self.current_time + total_estimated_time

        # Phần thưởng cơ bản
        reward = 10.0  # Phần thưởng giao hàng đúng hạn
        if estimated_delivery_time > package['deadline']:
            reward = 1.0  # Phần thưởng giao hàng trễ

        # Chi phí di chuyển
        move_cost_pickup = time_to_pickup * -0.01 if time_to_pickup > 0 else 0
        move_cost_deliver = time_to_deliver_from_pickup * -0.01 if time_to_deliver_from_pickup > 0 else 0
        total_move_cost = move_cost_pickup + move_cost_deliver

        # Điểm số: phần thưởng - (hệ số * thời gian) + chi phí di chuyển
        time_penalty_factor = 0.001
        score = reward - (total_estimated_time * time_penalty_factor) + total_move_cost

        # Giảm mạnh điểm nếu không thể giao kịp hoặc rất sát hạn
        if estimated_delivery_time > package['deadline'] + (len(self.map_grid) * 2):  # Thêm một chút buffer
            return -float('inf')

        # Ưu tiên các gói hàng có deadline sớm hơn nếu điểm số tương đương
        score -= package['deadline'] * 0.0001

        return score

    def get_actions(self, state):
        '''
        Hàm chính để lấy hành động cho các robot
        '''
        if not self.is_initialized:
            self.init_agents(state)  # Khởi tạo lần đầu nếu chưa

        self._update_internal_state(state)  # Cập nhật trạng thái nội bộ từ state của môi trường

        actions = []
        assigned_packages_this_step = set()  # Để tránh nhiều robot cùng nhắm 1 gói

        # Gán nhiệm vụ cho các robot rảnh rỗi
        for i in range(self.n_robots):
            robot = self.robots_info[i]
            
            if robot['phase'] == 'idle':
                best_package_id = None
                best_score = -float('inf')

                # Lọc các gói hàng có thể giao
                available_packages = sorted(
                    [pkg_id for pkg_id, pkg_info in self.packages_info.items()
                     if pkg_info['status'] == 'waiting' and pkg_info['start_time'] <= self.current_time],
                    key=lambda pkg_id: pkg_id
                )

                for pkg_id in available_packages:
                    if pkg_id in assigned_packages_this_step:  # Gói này đã được robot khác chọn
                        continue

                    score = self._calculate_score(i, pkg_id)
                    if score > best_score:
                        best_score = score
                        best_package_id = pkg_id

                # Nếu tìm thấy gói hàng tốt nhất cho robot này
                if best_package_id is not None:
                    # Assign gói hàng cho robot
                    self.robots_info[i]['target_package_id'] = best_package_id
                    # Đặt trạng thái robot là đi đến gói hàng
                    self.robots_info[i]['phase'] = 'to_pickup'
                    # Cập nhật trạng thái gói hàng là đã được gán
                    self.packages_info[best_package_id]['status'] = 'assigned'
                    # Đánh dấu robot đã được gán gói hàng
                    self.packages_info[best_package_id]['assigned_robot'] = i
                    # Đánh dấu gói hàng này đã được gán trong bước này
                    assigned_packages_this_step.add(best_package_id)

        # Xác định hành động cho từng robot
        for i in range(self.n_robots):
            robot = self.robots_info[i]
            move_action = 'S'
            package_action = '0'  # 0: Không có hành động liên quan đến gói hàng

            if robot['phase'] == 'to_pickup':
                target_pkg_id = robot['target_package_id']
                if target_pkg_id is None or target_pkg_id not in self.packages_info or self.packages_info[target_pkg_id]['status'] == 'delivered':
                    # Gói hàng mục tiêu không còn hợp lệ
                    robot['phase'] = 'idle'
                    robot['target_package_id'] = None
                else:
                    package_to_pickup = self.packages_info[target_pkg_id]
                    if robot['pos'] == package_to_pickup['start_pos']:
                        move_action = 'S'  # Đứng yên để nhặt
                        package_action = '1'  # Nhặt hàng
                    else:
                        move_action, _ = run_astar(self.map_grid, robot['pos'], package_to_pickup['start_pos'])

            elif robot['phase'] == 'to_dropoff':
                carrying_pkg_id = robot['carrying_package_id']
                if carrying_pkg_id is None or carrying_pkg_id not in self.packages_info:
                    # Lỗi logic: robot đang ở phase to_dropoff nhưng không mang hàng
                    robot['phase'] = 'idle'
                else:
                    package_to_dropoff = self.packages_info[carrying_pkg_id]
                    if robot['pos'] == package_to_dropoff['target_pos']:
                        move_action = 'S'  # Đứng yên để thả
                        package_action = '2'  # Thả hàng
                    else:
                        move_action, _ = run_astar(self.map_grid, robot['pos'], package_to_dropoff['target_pos'])

            actions.append((move_action, package_action))

        return actions  # Trả về danh sách hành động