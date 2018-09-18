CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

## Stream Compaction and Scan Algorithms with CUDA on the GPU
### Connie Chang
  * [LinkedIn](linkedin.com/in/conniechang44), [Demo Reel](vimeo.com/ConChang/DemoReel)
* Tested on: Windows 10, Intel Xeon CPU E5-1630 v4 @ 3.70 GHz, GTX 1070 8GB (SIG Lab)

## Introduction
This project explores different algorithms for stream compaction and exclusive scan, implemented in parallel with CUDA. We compare their performances to see which one is the best.  

__Exclusive Scan__  
Exclusive scan takes an array of integers as input and outputs another array of integers of the same length. Each element of the output is the sum of all previous elements of the input, excluding the current element. (For example, output[i] = sum(input[0] + ... +  input[i - 1]).  

__Stream Compaction__  
Stream compaction takes an array of integers as input and removes all the zeroes in it. Therefore, it outputs another array of integers that is of the same length or shorter. (For example, an input of [1, 5, 0, 3, 6, 0, 9] would result in [1, 5, 3, 6, 9])

## CPU Implementation
The CPU implementation is the most basic of the implementations in this project. The results from here become the baseline for comparison with later methods. 

__Exclusive Scan__  
On the CPU, exclusive scan is a simple FOR loop summing up all the elements in the input array. The pseudocode is as follows:
```
sum = 0
for (i = 0 ... (n - 1))  // n is size of input
    output[i] = sum
    sum += input[i]
```
 
 __Stream Compaction__
Likewise, stream compaction loops through all elements of the input array. If the element is not a zero, add it to the end of the output.  
```
count = 0
for (i = 0 ... (n - 1))  // n is size of input
    if input[i] != 0
        output[count] = input[i]
        count++
```

__Stream Compaction using Scan__  
This implementation is more interesting because it uses two other algorithms (scan and scatter) to complete stream compaction. First, the input is mapped to an array of 0s and 1s. If the input is a 0, then the map contains a 0 at the element. Otherwise, it is a 1. For example,  
```
input = [1, 5, 0, 3, 6, 0, 9]  
```
maps to
```
01 map = [1, 1, 0, 1, 1, 0, 1] 
```
Next, the 01 map is scanned using an exclusive scan. The result of the scan gives the index of where each non-zero element goes in the final output. For our example,  
```
scan of 01 map = [0, 1, 2, 2, 3, 4, 4] 
```
Finally, to achieve the final result, we loop through the 01 map. If the current position contains a 1, we know the element in the input array at that position should be in the output. The find what index it should be in for the output, we look at the scan of 01 map. The number at the current position is the final index. Then, we just copy the input element to the corresponding index in the output.  

Here's the pseudocode of the entire algorithm:  
```
// Map input to 01 map
for (i = 0 ... (n - 1))  // n is length of input
    if (input[i] == 0)
        01map[i] = 0
    else
        01map[i] = 1
        
// Run exclusive scan on 01map
scan = exclusive scan(01map)

for (i = 0 ... (n - 1))
    if 01map[i] == 1
        output[scan[i - 1]] = input[i]
```

## GPU Naive Algorithms
The naive algorithm on the GPU has to take advantage of the GPU's parallelism. The algorithm starts by replacing the current element with the sum of the current element and the one before it. Then, the current element is replaced by the sum of it and the integer 2 positions before it. Then, it's the sum of itself and the integer 2^2=4 positions before. Then, itself and 2^3=8 positions before. And so on, until 2^(k) exceeds the size of the array. The image below, from *GPU Gems,* illustrates this method:  

![](img/figure-39-2.jpg)  

Note that this creates an inclusive scan. We need to convert it to an exclusive scan by shifting every element to the right and adding a zero to the beginning.  

The pseudocode for the CPU side is:  
```
for (k = 1 ... ilog2(n))  // n is size of input
    Launch kernel, passing in k
```

The pseudocode for the device kernel is:  
```
if (index >= 2^k)
    output[index] = input[index - 2^k] + input[index]
else
    output[index] = input[index]
```

## GPU Work-Efficient Algorithms
The work efficient algorithm is somewhat similar to naive but with some improvements (maybe? See Performance Analysis below). The algorithm has two steps: Up-Sweep and Down-Sweep. Up-Sweep starts by going to every other element and summing itself with the previous element. It replaces the current element with the sum. Then, it does the same for the new integers. And so on, until there is only one new sum. In a sense, you are building up a tree. The image below, taken from CIS 565 slides, demonstrates this:  
![](img/work_efficient_up_sweep.PNG)  

The Down-Sweep is more complicated. First, the last element is set to 0. Then, each element passes its value to its left child in the tree. The right child is the sum of the current value and the left child's previous value. This process is repeated as we work down the tree. The image, also taken from CIS 565 slides, illustrates this:  
![](img/work_efficient_down_sweep.PNG)  

The pseudocode is as follows:  
```
// Up Sweep
for k = 0 ... ilogceil(n)  // n is size of input
    Launch Up Sweep kernel
    
Set last element of Up Sweep result to 0

// Down Sweep
for k = (ilogceil(n) - 1) ... 0
    Launch Down Sweep kernel
```

Pseudocode for Up Sweep kernel:  
```
if (index % 2^(k+1) == 0)
    data[index + 2^(k+1) - 1] += data[index + 2^k - 1]
```

Pseudocode for Down Sweep kernel:
```
if (index % 2^(k+1) == 0)
    temp = data[index + 2^k - 1]
    data[index + 2^k - 1] = data[index + 2^(k+1) - 1]  // Set left child to current element
    data[index + 2^(k+1) - 1] += temp  // Set current element to sum of itself and left child
```

## GPU Thrust Function 

## Performance Analysis
