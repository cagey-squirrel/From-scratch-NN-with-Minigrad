from math import exp

class Value(object):

    def __init__(self, data, first_parent=None, second_parent=None, operation=''):
        
        self.data = data
        self.operation = operation
        self.first_parent = first_parent
        self.second_parent = second_parent
        self.grad = 0
        self.label = 'un'
    

    def backprop(self):
        
        visited = set()
        topsorted_list = []
        def topsort(v):
            if v not in visited:
                visited.add(v)
                
                if v.first_parent:
                    topsort(v.first_parent)
                if v.second_parent:
                    topsort(v.second_parent)
                
                topsorted_list.append(v)
        
        topsort(self)

        self.grad = 1
        for v in reversed(topsorted_list):
            v._backprop()
    
    def _backprop(self):
        if self.operation == '+':
            self.first_parent.grad += self.grad
            self.second_parent.grad += self.grad

        elif self.operation == '-':
            self.first_parent.grad += self.grad
            self.second_parent.grad += -1 * self.grad
         
        elif self.operation == '*':
            self.first_parent.grad += self.grad * self.second_parent.data
            self.second_parent.grad += self.grad * self.first_parent.data

        elif self.operation == '/':
            self.first_parent.grad += self.grad / self.second_parent.data
            self.second_parent.grad += self.grad * self.first_parent.data * -1 / (self.second_parent.data)**2 
        
        elif self.operation == 'relu':
            self.first_parent.grad += self.grad * int(self.data > 0)
        
        elif self.operation == 'sigmoid':
            self.first_parent.grad += self.grad * self.data * (1 - self.data)  # data contains value of sigmoid(x)
        
        elif self.operation == 'tanh':
            self.first_parent.grad += self.grad * (1 - (self.data)**2)  # data contains value of tanh(x)

        elif self.operation == 'pow':
            self.first_parent.grad += self.grad * self.second_parent.data * (self.first_parent.data)**(self.second_parent.data - 1)

        elif self.operation == 'neg':
            self.first_parent.grad += self.grad * (-1)

    def __repr__(self):
        label = f"node: data = {self.data}"
        if self.operation != '':
            label += f', operation = {self.operation}'
        return label

    @staticmethod
    def relu(v):
        if not isinstance(v, Value):
            v = Value(v)

        new = Value(max(v.data, 0), first_parent=v, operation='relu')
        return new
    

    @staticmethod
    def sigmoid(v):
        if not isinstance(v, Value):
            v = Value(v)
        
        data = 1 / (1 + exp(-v.data))
        new = Value(data, first_parent=v, operation='sigmoid')
        return new
    

    @staticmethod
    def tanh(v):
        if not isinstance(v, Value):
            v = Value(v)
        
        e_2x = exp(2 * v.data)
        data = (e_2x - 1) / (e_2x + 1)

        new = Value(data, first_parent=v, operation='tanh')
        return new
    
    def __neg__(self):
        new = Value(-self.data, first_parent=self, operation='neg')
        return new

    def __pow__(self, other):
        other = Value(other)
        new = Value(self.data ** other.data, first_parent=self, second_parent=other, operation='pow')
        return new
    
    def __add__(self, other):

        if not isinstance(other, Value):
            other = Value(other)

        new = Value(self.data + other.data, first_parent=self, second_parent=other, operation='+')
        return new
    

    def __sub__(self, other):

        if not isinstance(other, Value):
            other = Value(other)

        new = Value(self.data - other.data, first_parent=self, second_parent=other, operation='-')
        return new
    

    def __rsub__(self, other):

        if not isinstance(other, Value):
            other = Value(other)

        new = Value(other.data - self.data, first_parent=other, second_parent=self, operation='-')
        return new


    def __mul__(self, other):

        if not isinstance(other, Value):
            other = Value(other)

        new = Value(self.data * other.data, first_parent=self, second_parent=other, operation='*')
        return new


    def __truediv__(self, other):

        if not isinstance(other, Value):
            other = Value(other)

        new = Value(self.data / other.data, first_parent=self, second_parent=other, operation='/')
        return new
    

    def __rtruediv__(self, other):

        if not isinstance(other, Value):
            other = Value(other)

        new = Value(other.data / self.data, first_parent=other, second_parent=self, operation='/')
        return new
    
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    
def main():
    a = Value(5)
    b = Value(10)

    


main()