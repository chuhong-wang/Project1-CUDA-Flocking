**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

![Alt text](<images/Peek 2024-02-02 18-45.gif>)

### Boid Simulation 

#### In case you're have the same question in mind ...
Q:  From brute force to pre-processing boids with uniform grid, what is the advantage of **sorting by grid cell index** strategy? It didn't come nature to me. 

A:  In order to improve computing efficiency, dividing the space into **uniform grid** and pre-assigning a grid cell index to each boid is a common approach to reduce the number of neighbor checking. 

(For demonstration purpose, suppose there're `N` boids. The space is divided into `M` grid cells.)

To implement the pre-processing, naturally, on **CPU** I'd allocate a  `gridCellIndexArrary` of size `N` to store the grid cell index of each boid, and a dynamic array `particleIndexArray` for each grid cell, then iterate through each boid and add its index to the corresponding grid cell's `particleIndexArray`. 

During flocking, for each boid we can obtain which cell it belongs to by `gridCellIndexArrary` and then the boids enclosed can be accessed by `particleIndexArray` in constant time.

Now we move on to **GPU**. The seemingly general strategy above is no longer applicable. For the reasons 1) There's no dynamic array on GPU. 2) potential race condition. Point 1 is straightward. Let me explain a little more on point 2. Suppose we parallelize the strategy above on GPU, each thread is responsible for one boid to add its index to the corresponding grid cell's `particleIndexArray`. When two of the boids in the same cell grid are being added to the same `particleIndexArray` at the same time, the two corresponding threads might overwrite each other's results, leading to erroneous record. 

So we need to revise the pre-processing method to avoid 1) using dynamics array 2) letting multiple threads writing to shared memory. One of the viable approaches is by sorting. Same as on CPU, we first allocate a  `gridCellIndexArrary` of size `N` to store the grid cell index of each boid. Meanwhile, we also allocate a `particleIndexArray` of size `N` to store the boid cell index of each boid. Then **sorting by key** with `gridCellIndexArrary` as key and `particleIndexArray` as value, we can obtain the start and end boid indices of each grid cell at the junction of blocks of homogeneous values in sorted `gridCellIndexArrary`. Mapping the boid enclosed in those grid cells during flocking is similar to CPU version. 





