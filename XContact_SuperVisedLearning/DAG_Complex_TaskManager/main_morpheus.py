import time
import logging
from collections import defaultdict, deque
from typing import Callable, Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Task:
    def __init__(self, name: str, func: Callable[[], Any], max_retries: int = 3):
        self.name = name
        self.func = func
        self.max_retries = max_retries
        self.retries = 0
        self.execution_time = 0
        self.status = "Pending"

    def run(self):
        while self.retries <= self.max_retries:
            try:
                start_time = time.time()
                logging.info(f"Starting task: {self.name}, Attempt: {self.retries + 1}")
                self.func()  # Run the task
                self.execution_time = time.time() - start_time
                self.status = "Success"
                logging.info(f"Completed task: {self.name}, Execution Time: {self.execution_time:.2f}s")
                break
            except Exception as e:
                self.retries += 1
                self.status = "Failed"
                logging.error(f"Task {self.name} failed on attempt {self.retries}. Error: {str(e)}")
                if self.retries > self.max_retries:
                    logging.error(f"Max retries reached for task {self.name}. Marking as Failed.")
                    break


class TaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.in_degree: Dict[str, int] = defaultdict(int)

    def add_task(self, name: str, func: Callable[[], Any], max_retries: int = 3):
        if name in self.tasks:
            raise ValueError(f"Task '{name}' already exists.")
        task = Task(name, func, max_retries)
        self.tasks[name] = task
        self.in_degree[name] = 0

    def add_dependency(self, task_name: str, dependency_name: str):
        if task_name not in self.tasks or dependency_name not in self.tasks:
            raise ValueError("Both tasks must be added before creating a dependency.")
        if dependency_name == task_name:
            raise ValueError("A task cannot depend on itself.")
        self.graph[dependency_name].append(task_name)
        self.in_degree[task_name] += 1

    def _topological_sort(self) -> List[str]:
        sorted_tasks = []
        queue = deque([task for task in self.tasks if self.in_degree[task] == 0])

        while queue:
            current = queue.popleft()
            sorted_tasks.append(current)

            for neighbor in self.graph[current]:
                self.in_degree[neighbor] -= 1
                if self.in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_tasks) != len(self.tasks):
            raise ValueError("Circular dependency detected. Tasks cannot be scheduled.")

        return sorted_tasks

    def execute(self):
        sorted_tasks = self._topological_sort()
        logging.info("Executing tasks in topological order.")

        for task_name in sorted_tasks:
            task = self.tasks[task_name]
            task.run()

    def get_metrics(self) -> List[Tuple[str, str, int, float]]:
        return [(task.name, task.status, task.retries, task.execution_time) for task in self.tasks.values()]


# Example usage

# Define tasks
def task1():
    time.sleep(1)  # Simulate task work
    logging.info("Task 1 completed.")


def task2():
    time.sleep(2)  # Simulate task work
    logging.info("Task 2 completed.")


def task3():
    time.sleep(1)  # Simulate task work
    logging.info("Task 3 completed.")


# Initialize scheduler
scheduler = TaskScheduler()
scheduler.add_task("task1", task1)
scheduler.add_task("task2", task2)
scheduler.add_task("task3", task3)

# Define dependencies
scheduler.add_dependency("task2", "task1")
scheduler.add_dependency("task3", "task1")

# Execute tasks
scheduler.execute()

# Get and print metrics
metrics = scheduler.get_metrics()
for name, status, retries, exec_time in metrics:
    logging.info(f"Task: {name}, Status: {status}, Retries: {retries}, Execution Time: {exec_time:.2f}s")
