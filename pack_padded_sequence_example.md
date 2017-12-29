
## This an example of using pytorch's pack_padded_sequence


```python
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
```


```python
# create input data
# input size = 4
# seq size = [3, 1]
# batch size = 2
input = [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 
            [[13, 14, 15, 16]]]
```


```python
# view input data values
input
```




    [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16]]]




```python
# lengths of sequences of input data
seq_lengths = torch.cuda.LongTensor(list(map(len, input)))
```


```python
# max length of sequences
seq_lengths.max()
```




    3




```python
# create sequence tensor for multi-sequences (4 is input size)
seq_tensor = Variable(torch.zeros(len(input), seq_lengths.max(), 4))
```


```python
# view empty sequence tensor
seq_tensor
```




    Variable containing:
    (0 ,.,.) = 
      0  0  0  0
      0  0  0  0
      0  0  0  0
    
    (1 ,.,.) = 
      0  0  0  0
      0  0  0  0
      0  0  0  0
    [torch.FloatTensor of size 2x3x4]




```python
# fill sequence tensor tensor with the first sequence
seq_tensor[0, :3] = torch.FloatTensor(np.asarray(input[0]))
```


```python
# fill sequence tensor tensor with the second sequence
seq_tensor[1, :1] = torch.FloatTensor(np.asarray(input[1]))
```


```python
# view filled sequence tensor
seq_tensor
```




    Variable containing:
    (0 ,.,.) = 
       1   2   3   4
       5   6   7   8
       9  10  11  12
    
    (1 ,.,.) = 
      13  14  15  16
       0   0   0   0
       0   0   0   0
    [torch.FloatTensor of size 2x3x4]




```python
# view shape of sequence tensor before transposing batch dimension and sequence dimension
seq_tensor.shape
```




    torch.Size([2, 3, 4])




```python
# view sequence tensor before transposing batch dimension and sequence dimension
seq_tensor
```




    Variable containing:
    (0 ,.,.) = 
       1   2   3   4
       5   6   7   8
       9  10  11  12
    
    (1 ,.,.) = 
      13  14  15  16
       0   0   0   0
       0   0   0   0
    [torch.FloatTensor of size 2x3x4]




```python
# transpose batch dimension and sequence dimension before padding data
seq_tensor = seq_tensor.transpose(0,1)
```


```python
# view shape of sequence tensor after transposing batch dimension and sequence dimension
seq_tensor.shape
```




    torch.Size([3, 2, 4])




```python
# view sequence tensor after transposing batch dimension and sequence dimension
seq_tensor
```




    Variable containing:
    (0 ,.,.) = 
       1   2   3   4
      13  14  15  16
    
    (1 ,.,.) = 
       5   6   7   8
       0   0   0   0
    
    (2 ,.,.) = 
       9  10  11  12
       0   0   0   0
    [torch.FloatTensor of size 3x2x4]




```python
# pad sequence tensor for rnn/lstm/gru network (batch_first=True if no transposing)
padded_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy(), batch_first=False)
```


```python
# view the padded result
padded_input
```




    PackedSequence(data=Variable containing:
      1   2   3   4
     13  14  15  16
      5   6   7   8
      9  10  11  12
    [torch.FloatTensor of size 4x4]
    , batch_sizes=[2, 1, 1])




```python
# unpad sequence tensor after training rnn/lstm/gru (batch_first=True if no transposing)
unpadded, unpadded_shape = pad_packed_sequence(packed_input, batch_first=False)
```


```python
# view unpadded tensor
unpadded
```




    Variable containing:
    (0 ,.,.) = 
       1   2   3   4
      13  14  15  16
    
    (1 ,.,.) = 
       5   6   7   8
       0   0   0   0
    
    (2 ,.,.) = 
       9  10  11  12
       0   0   0   0
    [torch.FloatTensor of size 3x2x4]




```python
# view shape of unpadded tensor
unpadded_shape
```




    [3, 1]


