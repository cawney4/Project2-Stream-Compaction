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

## GPU Work-Efficient Algorithms

## GPU Thrust Function 

## Performance Analysis
