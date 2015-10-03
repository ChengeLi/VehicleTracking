## Automatic Vehicle tracking system 

## Input data
Given a traffic video or a sequence of frame figures as input. 

## Usage
#### Steps:

###### 1
###### KLT tracker 
	Using KLT tracking algorithm to extract feature points and get
	their corresponding trajectory infomation.
	Save the trajectory info as .Mat file.

	The KLT tracking parameters are tunable. 

###### 2
###### Trajectory filtering and preprocessing
	Filter out bad trajectories based on: 
	1) the duration length, (trjs too short will be eliminated),
	2) the maximum speed, (low speed trjs will be eliminated). 
	Low speed trajectories can come from stationary feature points from 
	the background such as buildings or streets, or from the walking pedestrians. 

	As normally there exists multiple feature point trajectories on one 
	vehicle, we group those who are very close and merge them.

	Construct an adjacent matrix:
	Trajectories who share similar horizontal speed and vertical speed 
	will be treated as nearby, and be grouped together.
	Correspondingly these locations in the adjacent matrix will be assigned as 1. 

###### 3
###### Conduct Sparse Subspace clustering based on the built adjacency matrix. 

###### 4
###### Unify the labels across different frame chunks
	In order to speed up the process and also save the memory, we 
	process the video in chunks, 
		truncation length = 600 frames. 
	Hence long trajectories who span across differnt chunks may be assigned 
	to differnt labels and need to be unified and kept consistent.

###### 5
###### Visualize the labeled result


## Credits
An-ti Chiang, New York University,	dawnandyknight@gmail.com 
Chenge Li, New York University,	   cl2840@nyu.edu
Greg Dobler, New York University,  greg.dobler@nyu.edu 

## Information
If you have any questions or need more detailed explanations, 
please email Chenge Li for further information. :D


## License
MIT License







