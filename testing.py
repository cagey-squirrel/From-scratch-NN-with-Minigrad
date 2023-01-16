import torch
from value import Value

def basic_add_mul_test():

    x = Value(2); x.label = 'x'

    z = 2 * x; z.label = 'z'
    h = z + x; h.label = 'h'

    h.backprop()

    print(f'x.grad = {x.grad}, z.grad = {z.grad}, h.grad = {h.grad}')


def basic_operations_test():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    h = z / x - 3
    q = 20 - h * x
    y = h + q + q * x
    y.backprop()
    
    print(f'x.grad = {x.grad}, z.grad = {z.grad}, h.grad = {h.grad}, q.grad = {q.grad}, y.grad = {y.grad}')


    with torch.set_grad_enabled(True):
        x = torch.Tensor([-4.0]).double()
        x.requires_grad = True
        z = 2 * x + 2 + x
        z.retain_grad()
        h = z / x - 3
        h.retain_grad()
        q = 20 - h * x
        q.retain_grad()
        y = h + q + q * x
        y.retain_grad()
        y.backward()

    print(f'x.grad = {x.grad.item()}, z.grad = {z.grad.item()}, h.grad = {h.grad.item()}, q.grad = {q.grad.item()}, y.grad = {y.grad.item()}')

def relu_test():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = Value.relu(z) + z * x
    h = Value.relu((z * z))
    y = h + q + q * x
    y.backprop()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    print(f'my grad = {xmg.grad}, torch grad = {xpt.grad.item()}')


def big_relu_test():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + Value.relu((b + a))
    d += 3 * d + Value.relu((b - a))
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()

    print(f'a.grad = {a.grad}, b.grad = {b.grad}, c.grad = {c.grad}, d.grad = {d.grad}, e.grad = {e.grad}, f.grad = {f.grad}, g.grad = {g.grad}')
    

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f

    a.retain_grad(); b.retain_grad(); c.retain_grad(); d.retain_grad(); e.retain_grad(); f.retain_grad(); g.retain_grad(); 
    g.backward()

    print(f'a.grad = {a.grad.item()}, b.grad = {b.grad.item()}, c.grad = {c.grad.item()}, d.grad = {d.grad.item()}, e.grad = {e.grad.item()}, f.grad = {f.grad.item()}, g.grad = {g.grad.item()}')


def tanh_test():
    a = Value(-4.0)
    b = Value(2.0)
    
    c = a * 5 + b - 22
    Value.tanh(c)
    c.backprop()

    print(f'a.grad = {a.grad}, b.grad = {b.grad}, c.grad = {c.grad}')
    

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a * 5 + b - 22
    
    a.retain_grad(); b.retain_grad(); c.retain_grad()
    c.backward()

    print(f'a.grad = {a.grad.item()}, b.grad = {b.grad.item()}, c.grad = {c.grad.item()}')


def tanh_big_test():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + Value.tanh((b + a))
    d += 3 * d + Value.tanh((b - a))
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()

    print(f'a.grad = {a.grad}, b.grad = {b.grad}, c.grad = {c.grad}, d.grad = {d.grad}, e.grad = {e.grad}, f.grad = {f.grad}, g.grad = {g.grad}')
    

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).tanh()
    d = d + 3 * d + (b - a).tanh()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f

    a.retain_grad(); b.retain_grad(); c.retain_grad(); d.retain_grad(); e.retain_grad(); f.retain_grad(); g.retain_grad(); 
    g.backward()

    print(f'a.grad = {a.grad.item()}, b.grad = {b.grad.item()}, c.grad = {c.grad.item()}, d.grad = {d.grad.item()}, e.grad = {e.grad.item()}, f.grad = {f.grad.item()}, g.grad = {g.grad.item()}')

def sigmoid_big_test():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + Value.sigmoid((b + a))
    d += 3 * d + Value.sigmoid((b - a))
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backprop()

    print(f'a.grad = {a.grad}, b.grad = {b.grad}, c.grad = {c.grad}, d.grad = {d.grad}, e.grad = {e.grad}, f.grad = {f.grad}, g.grad = {g.grad}')
    

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).sigmoid()
    d = d + 3 * d + (b - a).sigmoid()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f

    a.retain_grad(); b.retain_grad(); c.retain_grad(); d.retain_grad(); e.retain_grad(); f.retain_grad(); g.retain_grad(); 
    g.backward()

    print(f'a.grad = {a.grad.item()}, b.grad = {b.grad.item()}, c.grad = {c.grad.item()}, d.grad = {d.grad.item()}, e.grad = {e.grad.item()}, f.grad = {f.grad.item()}, g.grad = {g.grad.item()}')


#tanh_test()
tanh_big_test()
#sigmoid_big_test()
#big_relu_test()