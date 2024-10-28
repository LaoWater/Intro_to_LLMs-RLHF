import time
import logging
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Set

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class Task:
    def __init__(self, name: str, func: Callable, retries: int = 3, priority: int = 0):
        self.name = name
        self.func = func
        self.retries = retries
        self.priority = priority
        self.execution_time = 0
        self.retry_count = 0
        self.success = False

    def execute(self):
        """Execute the task with retry logic and time measurement."""
        start_time = time.time()
        for attempt in range(self.retries):
            try:
                logging.info(f"Executing task {self.name}, attempt {attempt + 1}")
                self.func()
                self.success = True
                break
            except Exception as e:
                self.retry_count += 1
                logging.warning(f"Task {self.name} failed on attempt {attempt + 1}: {e}")
        self.execution_time = time.time() - start_time
        if self.success:
            logging.info(f"Task {self.name} completed successfully in {self.execution_time:.2f} seconds")
        else:
            logging.error(f"Task {self.name} failed after {self.retries} retries")


class TaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.execution_order: List[str] = []

    def add_task(self, task: Task, dependencies: Optional[List[str]] = None):
        """Add a task to the scheduler with optional dependencies."""
        if task.name in self.tasks:
            raise ValueError(f"Task {task.name} already exists.")

        self.tasks[task.name] = task
        dependencies = dependencies or []

        for dep in dependencies:
            if dep not in self.tasks:
                raise ValueError(f"Dependency task {dep} does not exist.")
            self.dependencies[dep].add(task.name)
            self.in_degree[task.name] += 1

        # Ensure task is initialized in in_degree
        if task.name not in self.in_degree:
            self.in_degree[task.name] = 0

        # Check for circular dependencies
        if not self.is_dag():
            # Rollback changes if cycle detected
            for dep in dependencies:
                self.dependencies[dep].remove(task.name)
                self.in_degree[task.name] -= 1
            del self.tasks[task.name]
            raise ValueError("Adding this task introduces a circular dependency.")

    def is_dag(self) -> bool:
        """Check if the current graph structure is a DAG using Kahn's algorithm."""
        temp_in_degree = dict(self.in_degree)
        zero_in_degree = deque([node for node in self.tasks if temp_in_degree[node] == 0])
        count = 0

        while zero_in_degree:
            node = zero_in_degree.popleft()
            count += 1
            for neighbor in self.dependencies[node]:
                temp_in_degree[neighbor] -= 1
                if temp_in_degree[neighbor] == 0:
                    zero_in_degree.append(neighbor)

        return count == len(self.tasks)

    def get_execution_order(self):
        """Generate execution order based on topological sort and task priority."""
        temp_in_degree = dict(self.in_degree)
        priority_queue = sorted((self.tasks[node].priority, node) for node in self.tasks if temp_in_degree[node] == 0)

        while priority_queue:
            priority_queue.sort(reverse=True)  # Higher priority tasks first
            _, node = priority_queue.pop()
            self.execution_order.append(node)

            for neighbor in self.dependencies[node]:
                temp_in_degree[neighbor] -= 1
                if temp_in_degree[neighbor] == 0:
                    priority_queue.append((self.tasks[neighbor].priority, neighbor))

    def execute(self):
        """Execute all tasks in the correct order, handling retries and logging."""
        self.get_execution_order()
        for task_name in self.execution_order:
            task = self.tasks[task_name]
            task.execute()

    def get_metrics(self):
        """Retrieve task metrics including execution time and retry count."""
        metrics = {task.name: {'execution_time': task.execution_time, 'retry_count': task.retry_count} for task in
                   self.tasks.values()}
        return metrics


# Example tasks
def sample_task():
    logging.info("Task executed successfully.")


def failing_task():
    raise ValueError("Simulated task failure.")


# Initializing the scheduler and adding tasks with dependencies
scheduler = TaskScheduler()
task1 = Task("Task1", sample_task, retries=3, priority=2)
task2 = Task("Task2", failing_task, retries=2, priority=1)
task3 = Task("Task3", sample_task, retries=1, priority=3)

# Adding tasks with dependencies
scheduler.add_task(task1)
scheduler.add_task(task2, dependencies=["Task1"])
scheduler.add_task(task3, dependencies=["Task2"])

# Execute tasks and display metrics
scheduler.execute()
print("Task Metrics:", scheduler.get_metrics())
