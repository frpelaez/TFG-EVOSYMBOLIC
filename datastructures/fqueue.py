from typing import Any

class Queue:
    
    def __init__(self) -> None:
        
        self.queue = []
        
    def is_empty(self) -> bool:
        
        return self.queue == []
    
    def first(self) -> Any:
        
        if self.is_empty(): raise Exception("Empty queue has no first element")
        
        return self.queue[0]
    
    def move(self) -> None:
        
        if self.is_empty(): raise Exception("Unnable to move empty queue")
        
        self.queue.pop(0)
        
    def enqueue(self, data: Any) -> None:
    
        self.queue.append(data)
        
    def lenght(self) -> int:
        
        return len(self.queue)
        
    def show(self) -> None:
        
        if self.lenght() == 0: print("| |")
        elif self.lenght() == 1: print(f"| {self.first()} |")
        else:
            tail = ""
            for e in self.queue[1:self.lenght() - 1]: tail += f"{e} > "
            tail += f"{self.queue[-1]}"
            fstr = "".join(f"| {self.first()} > {tail} |")
            print(fstr)