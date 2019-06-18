## PyTorch代码调试利器: 自动print每行代码的Tensor信息

[机器之心](javascript:void(0);) *今天*

机器之心发布

**作者：zasdfgbnm**

> 本文介绍一个用于 PyTorch 代码的实用工具 TorchSnooper。作者是TorchSnooper的作者，也是PyTorch开发者之一

GitHub 项目地址： https://github.com/zasdfgbnm/TorchSnooper



大家可能遇到这样子的困扰：比如说运行自己编写的 PyTorch 代码的时候，PyTorch 提示你说数据类型不匹配，需要一个 double 的 tensor 但是你给的却是 float；再或者就是需要一个 CUDA tensor, 你给的却是个 CPU tensor。比如下面这种：



```
RuntimeError: Expected object of scalar type Double but got scalar type Float
```



这种问题调试起来很麻烦，因为你不知道从哪里开始出问题的。比如你可能在代码的第三行用 torch.zeros 新建了一个 CPU tensor, 然后这个 tensor 进行了若干运算，全是在 CPU 上进行的，一直没有报错，直到第十行需要跟你作为输入传进来的 CUDA tensor 进行运算的时候，才报错。要调试这种错误，有时候就不得不一行行地手写 print 语句，非常麻烦。



再或者，你可能脑子里想象着将一个 tensor 进行什么样子的操作，就会得到什么样子的结果，但是 PyTorch 中途报错说 tensor 的形状不匹配，或者压根没报错但是最终出来的形状不是我们想要的。这个时候，我们往往也不知道是什么地方开始跟我们「预期的发生偏离的」。我们有时候也得需要插入一大堆 print 语句才能找到原因。



TorchSnooper 就是一个设计了用来解决这个问题的工具。TorchSnooper 的安装非常简单，只需要执行标准的 Python 包安装指令就好：



```
pip install torchsnooper
```



安装完了以后，只需要用 @torchsnooper.snoop() 装饰一下要调试的函数，这个函数在执行的时候，就会自动 print 出来每一行的执行结果的 tensor 的形状、数据类型、设备、是否需要梯度的信息。



安装完了以后，下面就用两个例子来说明一下怎么使用。



**例子1**



比如说我们写了一个非常简单的函数：



```
def myfunc(mask, x):
    y = torch.zeros(6)
    y.masked_scatter_(mask, x)
    return y
```



我们是这样子使用这个函数的：



```
mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda')
source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = myfunc(mask, source)
```



上面的代码看起来似乎没啥问题，然而实际上跑起来，却报错了：



```
RuntimeError: Expected object of backend CPU but got backend CUDA for argument #2 'mask'
```



问题在哪里呢？让我们 snoop 一下！用 @torchsnooper.snoop() 装饰一下 myfunc 函数：



```
import torch
import torchsnooper

@torchsnooper.snoop()
def myfunc(mask, x):
    y = torch.zeros(6)
    y.masked_scatter_(mask, x)
    return y

mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda')
source = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = myfunc(mask, source)
```



然后运行我们的脚本，我们看到了这样的输出：



```
Starting var:.. mask = tensor<(6,), int64, cuda:0>
Starting var:.. x = tensor<(3,), float32, cuda:0>
21:41:42.941668 call         5 def myfunc(mask, x):
21:41:42.941834 line         6     y = torch.zeros(6)
New var:....... y = tensor<(6,), float32, cpu>
21:41:42.943443 line         7     y.masked_scatter_(mask, x)
21:41:42.944404 exception    7     y.masked_scatter_(mask, x)
```



结合我们的错误，我们主要去看输出的每个变量的设备，找找最早从哪个变量开始是在 CPU 上的。我们注意到这一行：



```
New var:....... y = tensor<(6,), float32, cpu>
```



这一行直接告诉我们，我们创建了一个新变量 y, 并把一个 CPU tensor 赋值给了这个变量。这一行对应代码中的 y = torch.zeros(6)。于是我们意识到，在使用 torch.zeros 的时候，如果不人为指定设备的话，默认创建的 tensor 是在 CPU 上的。我们把这一行改成 y = torch.zeros(6, device='cuda')，这一行的问题就修复了。



这一行的问题虽然修复了，我们的问题并没有解决完整，再跑修改过的代码还是报错，但是这个时候错误变成了：



```
RuntimeError: Expected object of scalar type Byte but got scalar type Long for argument #2 'mask'
```



好吧，这次错误出在了数据类型上。这次错误报告比较有提示性，我们大概能知道是我们的 mask 的数据类型错了。再看一遍 TorchSnooper 的输出，我们注意到：



```
Starting var:.. mask = tensor<(6,), int64, cuda:0>
```



果然，我们的 mask 的类型是 int64, 而不应该是应有的 uint8。我们把 mask 的定义修改好：



```
mask = torch.tensor([0, 1, 0, 1, 1, 0], device='cuda', dtype=torch.uint8)
```



然后就可以运行了。



**例子 2**



这次我们要构建一个简单的线性模型：



```
model = torch.nn.Linear(2, 1)
```



我们想要拟合一个平面 y = x1 + 2 * x2 + 3，于是我们创建了这样一个数据集：



```
x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])
```



我们使用最普通的 SGD 优化器来进行优化，完整的代码如下：



```
import torch

model = torch.nn.Linear(2, 1)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(10):
    optimizer.zero_grad()
    pred = model(x)
    squared_diff = (y - pred) ** 2
    loss = squared_diff.mean()
    print(loss.item())
    loss.backward()
    optimizer.step()
```



然而运行的过程我们发现，loss 降到 1.5 左右就不再降了。这是很不正常的，因为我们构建的数据都是无误差落在要拟合的平面上的，loss 应该降到 0 才算正常。



乍看上去，不知道问题在哪里。抱着试试看的想法，我们来 snoop 一下子。这个例子中，我们没有自定义函数，但是我们可以使用 with 语句来激活 TorchSnooper。把训练的那个循环装进 with 语句中去，代码就变成了：



```
import torch
import torchsnooper

model = torch.nn.Linear(2, 1)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([3.0, 5.0, 4.0, 6.0])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with torchsnooper.snoop():
    for _ in range(10):
        optimizer.zero_grad()
        pred = model(x)
        squared_diff = (y - pred) ** 2
        loss = squared_diff.mean()
        print(loss.item())
        loss.backward()
        optimizer.step()
```



运行程序，我们看到了一长串的输出，一点一点浏览，我们注意到



```
New var:....... model = Linear(in_features=2, out_features=1, bias=True)
New var:....... x = tensor<(4, 2), float32, cpu>
New var:....... y = tensor<(4,), float32, cpu>
New var:....... optimizer = SGD (Parameter Group 0    dampening: 0    lr: 0....omentum: 0    nesterov: False    weight_decay: 0)
02:38:02.016826 line        12     for _ in range(10):
New var:....... _ = 0
02:38:02.017025 line        13         optimizer.zero_grad()
02:38:02.017156 line        14         pred = model(x)
New var:....... pred = tensor<(4, 1), float32, cpu, grad>
02:38:02.018100 line        15         squared_diff = (y - pred) ** 2
New var:....... squared_diff = tensor<(4, 4), float32, cpu, grad>
02:38:02.018397 line        16         loss = squared_diff.mean()
New var:....... loss = tensor<(), float32, cpu, grad>
02:38:02.018674 line        17         print(loss.item())
02:38:02.018852 line        18         loss.backward()
26.979290008544922
02:38:02.057349 line        19         optimizer.step()
```



仔细观察这里面各个 tensor 的形状，我们不难发现，y 的形状是 (4,)，而 pred 的形状却是 (4, 1)，他们俩相减，由于广播的存在，我们得到的 squared_diff 的形状就变成了 (4, 4)。



这自然不是我们想要的结果。这个问题修复起来也很简单，把 pred 的定义改成 pred = model(x).squeeze() 即可。现在再看修改后的代码的 TorchSnooper 的输出：



```
New var:....... model = Linear(in_features=2, out_features=1, bias=True)
New var:....... x = tensor<(4, 2), float32, cpu>
New var:....... y = tensor<(4,), float32, cpu>
New var:....... optimizer = SGD (Parameter Group 0    dampening: 0    lr: 0....omentum: 0    nesterov: False    weight_decay: 0)
02:46:23.545042 line        12     for _ in range(10):
New var:....... _ = 0
02:46:23.545285 line        13         optimizer.zero_grad()
02:46:23.545421 line        14         pred = model(x).squeeze()
New var:....... pred = tensor<(4,), float32, cpu, grad>
02:46:23.546362 line        15         squared_diff = (y - pred) ** 2
New var:....... squared_diff = tensor<(4,), float32, cpu, grad>
02:46:23.546645 line        16         loss = squared_diff.mean()
New var:....... loss = tensor<(), float32, cpu, grad>
02:46:23.546939 line        17         print(loss.item())
02:46:23.547133 line        18         loss.backward()
02:46:23.591090 line        19         optimizer.step()
```



现在这个结果看起来就正常了。并且经过测试，loss 现在已经可以降到很接近 0 了。大功告成。





**本文为机器之心发布，转载请联系本公众号获得授权。**

✄------------------------------------------------

**加入机器之心（全职记者 / 实习生）：hr@jiqizhixin.com**

**投稿或寻求报道：content@jiqizhixin.com**

**广告 & 商务合作：bd@jiqizhixin.com**









微信扫一扫
关注该公众号