from typing import Any

class Stack:
    
    def __init__(self) -> None:
        
        self.stack= []
    
    def is_empty(self) -> bool:
        
        return self.stack == []
    
    def top(self) -> Any:
        
        if self.is_empty(): raise Exception("Empty stack has no top element")
        
        return self.stack[0]
    
    def pop(self) -> Any:
        
        if self.is_empty(): raise Exception("Unnable to pop top element from empty stack")
        
        return self.stack.pop(0)
    
    def put(self, data: Any) -> None:
        
        self.stack.insert(0, data)
        
    def size(self) -> int:
        
        return len(self.stack)
        
    def show(self) -> None:
        
        if self.size() == 0: print("<>")
        elif self.size() == 1: print(f"<{self.top()}>")
        else:
            tail = ""
            for e in self.stack[1:self.size() - 1]: tail += f"{e} "
            tail += f"{self.stack[-1]}"
            fstr = "".join(f"<{self.top()} | {tail}>")
            print(fstr)