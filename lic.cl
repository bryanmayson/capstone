//check that a number is within given bounds
int checkBounds(int value,int lower,int upper)
{
	if(value>=lower &&value<upper)
	{
		return 1;
	}
	return 0;
}


kernel void lic(   global float *map, constant double *weights, constant double *weightsSum, int weightLength)
{
	
	//add these numbers to get corresponding ajacent values
	const int UP=-get_global_size(0);
	const int DOWN=-UP;
	const int LEFT=-1;
	const int RIGHT=1;
	
	//starting point in the buffer for this 2D point
	const int startingIdx = get_global_id(0)+get_global_id(1)*get_global_size(0);
	//map[startingIdx]=map[startingIdx]+1;
	//return;
	//float thing=5/0;
	//map[startingIdx]=map[startingIdx]*2;
	//return;
	
	//size of the map
	int2 mapSize={get_global_size(0)-1,get_global_size(1)-1};
	
	//initialise weights and height
	float totalWeight=weights[0];
	float totalHeight=map[startingIdx]*totalWeight;
	
	
	//algorithm starts at a point and looks for it's highest (or lowest for downhill) neighbour, adds it's height with the appropriate weight and repeats the proccess on it
	int currentIdx=startingIdx;
	int currentVal=map[startingIdx];
	int2 currentPos={get_global_id(0),get_global_id(1)};
	int i;
	//uphill climb
	for(i=1;i<=weightLength;i++)
	{
		//get the nextuphill point
		int currentHighestIdx=currentIdx;
		float currentHighestVal=currentVal;
		int2 currentHighestPos=currentPos;
		
		int2 leftPos=currentPos;
		leftPos.x-=1;
		int2 rightPos=currentPos;
		rightPos.x+=1;
		int2 upPos=currentPos;
		upPos.y-=1;
		int2 downPos=currentPos;
		downPos.y+=1;
		
		float2 grad={0,0};
		if(leftPos.x>=0)
		{
			if(rightPos.x<get_global_size(0))
			{
				grad.x=(map[currentIdx+RIGHT]-map[currentIdx+LEFT])/2.0;
			}
			else
			{
				grad.x=currentVal-map[currentIdx+LEFT];
			}
		}
		else
		{
			grad.x=map[currentIdx+RIGHT]-currentVal;
		}
		
		if(upPos.x>=0)
		{
			if(downPos.y<get_global_size(1))
			{
				grad.y=(map[currentIdx+UP]-map[currentIdx+DOWN])/2.0;
			}
			else
			{
				grad.y=map[currentIdx+UP]-currentVal;
			}
		}
		else
		{
			grad.y=currentVal-map[currentIdx+DOWN];
		}
		//grad.y=-grad.y;
		grad.x=-grad.x;
	
		float2 normalised=normalize(grad);
		int2 nextPos={clamp((int)round(currentPos.x+normalised.x),0,(int)get_global_size(0)-1),clamp((int)round(currentPos.y+normalised.y),0,(int)get_global_size(1)-1)};
		int nextIdx=nextPos.x+nextPos.y*get_global_size(0);
		float nextVal=map[nextIdx];
		totalHeight+=nextVal*weights[i];
		totalWeight+=weights[i];
		currentPos=nextPos;
		currentVal=nextVal;
		currentIdx=nextIdx;
	}
	
	//downhill climb
	for(i=1;i<=weightLength;i++)
	{
		//get the nextuphill point
		int currentHighestIdx=currentIdx;
		float currentHighestVal=currentVal;
		int2 currentHighestPos=currentPos;
		
		int2 leftPos=currentPos;
		leftPos.x-=1;
		int2 rightPos=currentPos;
		rightPos.x+=1;
		int2 upPos=currentPos;
		upPos.y-=1;
		int2 downPos=currentPos;
		downPos.y+=1;
		
		float2 grad={0,0};
		if(leftPos.x>=0)
		{
			if(rightPos.x<get_global_size(0))
			{
				grad.x=(map[currentIdx+RIGHT]-map[currentIdx+LEFT])/2.0;
			}
			else
			{
				grad.x=currentVal-map[currentIdx+LEFT];
			}
		}
		else
		{
			grad.x=map[currentIdx+RIGHT]-currentVal;
		}
		
		if(upPos.x>=0)
		{
			if(downPos.y<get_global_size(1))
			{
				grad.y=(map[currentIdx+UP]-map[currentIdx+DOWN])/2.0;
			}
			else
			{
				grad.y=map[currentIdx+UP]-currentVal;
			}
		}
		else
		{
			grad.y=currentVal-map[currentIdx+DOWN];
		}
		//grad.x=-grad.x;
		grad.y=-grad.y;
		float2 normalised=normalize(grad);
		int2 nextPos={clamp((int)round(currentPos.x+normalised.x),0,(int)get_global_size(0)-1),clamp((int)round(currentPos.y+normalised.y),0,(int)get_global_size(1)-1)};
		int nextIdx=nextPos.x+nextPos.y*get_global_size(0);
		float nextVal=map[nextIdx];
		totalHeight+=nextVal*weights[i];
		totalWeight+=weights[i];
		currentPos=nextPos;
		currentVal=nextVal;
		currentIdx=nextIdx;
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	map[startingIdx]=totalHeight/totalWeight;
}

//the lic function that will be run on each point
kernel void licOriginal(   global float *map, constant double *weights, constant double *weightsSum, int weightLength)
{
	//add these numbers to get corresponding ajacent values
	const int UP=-get_global_size(0);
	const int DOWN=-UP;
	const int LEFT=-1;
	const int RIGHT=1;
	
	//starting point in the buffer for this 2D point
	const int startingIdx = get_global_id(0)+get_global_id(1)*get_global_size(0);
	//float thing=5/0;
	//map[startingIdx]=map[startingIdx]*2;
	//return;
	
	//size of the map
	int2 mapSize={get_global_size(0)-1,get_global_size(1)-1};
	
	//initialise weights and height
	float totalWeight=weights[0];
	float totalHeight=map[startingIdx]*totalWeight;
	
	
	//algorithm starts at a point and looks for it's highest (or lowest for downhill) neighbour, adds it's height with the appropriate weight and repeats the proccess on it
	int currentIdx=startingIdx;
	int currentVal=map[startingIdx];
	int2 currentPos={get_global_id(0),get_global_id(1)};
	int i;
	//uphill climb
	for(i=1;i<=weightLength;i++)
	{
		//get the nextuphill point
		int currentHighestIdx=currentIdx;
		float currentHighestVal=currentVal;
		int2 currentHighestPos=currentPos;
		
		int2 leftPos=currentPos;
		leftPos.x-=1;
		int2 rightPos=currentPos;
		rightPos.x+=1;
		int2 upPos=currentPos;
		upPos.y-=1;
		int2 downPos=currentPos;
		downPos.y+=1;
		
		
		//checkBounds(currentIdx+DOWN,0,mapSize.y)
		
		//Up
		if(upPos.y>=0)
		{
			float v = map[currentIdx+UP];
			if(v>currentHighestVal)
			{
				currentHighestIdx=currentIdx+UP;
				currentHighestVal=v;
				currentHighestPos=upPos;
			}
		}
		
		//Down
		if(downPos.y<get_global_size(1))
		{
			float v = map[currentIdx+DOWN];
			if(v>currentHighestVal)
			{
				currentHighestIdx=currentIdx+DOWN;
				currentHighestVal=v;
				currentHighestPos=downPos;
			}
		}
		
		//Left
		if(leftPos.x>=0)
		{
			float v = map[currentIdx+LEFT];
			if(v>currentHighestVal)
			{
				currentHighestIdx=currentIdx+LEFT;
				currentHighestVal=v;
				currentHighestPos=leftPos;
			}
		}
		
		//Right
		if(rightPos.x<get_global_size(0))
		{
			float v = map[currentIdx+RIGHT];
			if(v>currentHighestVal)
			{
				currentHighestIdx=currentIdx+RIGHT;
				currentHighestVal=v;
				currentHighestPos=rightPos;
			}
		}
		//add weights and heights and set the new index
		totalHeight+=(float)currentHighestVal*weights[i];
		totalWeight+=(float)weights[i];
		currentIdx=currentHighestIdx;
		currentPos=currentHighestPos;
	}
	
	//downhill climb
	currentIdx=startingIdx;
	currentVal=map[startingIdx];
	for(i=1;i<=weightLength;i++)
	{
		//get the next downhill point
		int currentHighestIdx=currentIdx;
		float currentHighestVal=currentVal;
		
		//Up
		if(checkBounds(currentIdx+UP,0,mapSize.y))
		{
			float v = map[currentIdx+UP];
			if(v<currentHighestVal)
			{
				currentHighestIdx=currentIdx+UP;
				currentHighestVal=v;
			}
		}
		
		//Down
		if(checkBounds(currentIdx+DOWN,0,mapSize.y))
		{
			float v = map[currentIdx+DOWN];
			if(v<currentHighestVal)
			{
				currentHighestIdx=currentIdx+DOWN;
				currentHighestVal=v;
			}
		}
		
		//Left
		if(checkBounds(currentIdx+LEFT,0,mapSize.x))
		{
			float v = map[currentIdx+LEFT];
			if(v<currentHighestVal)
			{
				currentHighestIdx=currentIdx+LEFT;
				currentHighestVal=v;
			}
		}
		
		//Right
		if(checkBounds(currentIdx+RIGHT,0,mapSize.x))
		{
			float v = map[currentIdx+RIGHT];
			if(v<currentHighestVal)
			{
				currentHighestIdx=currentIdx+RIGHT;
				currentHighestVal=v;
			}
		}
		totalHeight+=(float)currentHighestVal*weights[i];
		totalWeight+=(float)weights[i];
		currentIdx=currentHighestIdx;
	}
	//wait for everything else to finish before writing to the buffer
	/*for(i=1;i<weightLength;i++)
	{
		int clampedIdx=clamp(startingIdx+RIGHT,0,(int)(get_global_size(0)*get_global_size(1)-1));
		totalHeight+=map[clampedIdx];
		totalWeight+=1;
	}*/
	barrier(CLK_GLOBAL_MEM_FENCE);
	map[startingIdx]=totalHeight/totalWeight;
	
	
	
}





/*kernel void licOther(   global float *map, constant double *weights, constant double *weightsSum, int weightLength)
{
	//add these numbers to get corresponding ajacent values
	const int UP=-get_global_size(1);
	const int DOWN=-UP;
	const int LEFT=-1;
	const int RIGHT=1;
	
	//starting point in the buffer for this 2D point
	const int startingIdx = get_global_id(1)+(get_global_id(0)*get_global_size(1));
	map[startingIdx]=map[startingIdx]*2;
	return;
	
	//size of the map
	int2 mapSize={get_global_size(1)-1,get_global_size(0)-1};
	
	//initialise weights and height
	float totalWeight=weights[0];
	float totalHeight=map[startingIdx]*totalWeight;
	
	
	//algorithm starts at a point and looks for it's highest (or lowest for downhill) neighbour, adds it's height with the appropriate weight and repeats the proccess on it
	int currentIdx=startingIdx;
	int i;
	//uphill climb
	for(i=1;i<=weightLength;i++)
	{
		int upIdx=currentIdx+UP;
		upIdx=(upIdx>=0) ? upIdx:currentIdx;
		int downIdx=currentIdx+DOWN;
		downIdx=(downIdx<(get_global_size(0)*get_global_size(1))) ? downIdx:currentIdx;
		int leftIdx=currentIdx+LEFT;
		leftIdx=(leftIdx>=get_global_id(0)*get_global_size(1)) ? leftIdx:currentIdx;
		int rightIdx=currentIdx+RIGHT;
		upIdx=(rightIdx<(get_global_id(0)+1)*get_global_size(1)) ? rightIdx:currentIdx;
		
		float currentHieght=map[currentIdx];
		float upDelta=map[currentIdx+UP]-currentHieght;
		float downDelta=map[currentIdx+DOWN]-currentHieght;
		float leftDelta=map[currentIdx+LEFT]-currentHieght;
		float rightDelta=map[currentIdx+RIGHT]-currentHieght;
		float maxDelta=max(max(upDelta,downDelta),max(leftDelta,rightDelta));
		
		//check for local maximum
		int highestIdx=currentIdx;
		float highestVal;
		if(maxDelta>0)
		{
			if(maxDelta==upDelta)
			{
				highestIdx=currentIdx+UP;
			}
			else if(maxDelta==downDelta)
			{
				highestIdx=currentIdx+DOWN;
			}
			else if(maxDelta==leftDelta)
			{
				highestIdx=currentIdx+LEFT;
			}
			else if(maxDelta==rightDelta)
			{
				highestIdx=currentIdx+RIGHT;
			}
		}
		highestVal=map[highestIdx];
		totalHeight+=highestVal*weights[i];
		totalWeight+=weights[i];
		currentIdx=highestIdx;
	}
	
	//downhill climb
	currentIdx=startingIdx;
	for(i=1;i<=weightLength;i++)
	{
		int upIdx=currentIdx+UP;
		upIdx=(upIdx>=0) ? upIdx:currentIdx;
		int downIdx=currentIdx+DOWN;
		downIdx=(downIdx<(get_global_size(0)*get_global_size(1))) ? downIdx:currentIdx;
		int leftIdx=currentIdx+LEFT;
		leftIdx=(leftIdx>=get_global_id(0)*get_global_size(1)) ? leftIdx:currentIdx;
		int rightIdx=currentIdx+RIGHT;
		upIdx=(rightIdx<(get_global_id(0)+1)*get_global_size(1)) ? rightIdx:currentIdx;
		
		float currentHieght=map[currentIdx];
		float upDelta=map[currentIdx+UP]-currentHieght;
		float downDelta=map[currentIdx+DOWN]-currentHieght;
		float leftDelta=map[currentIdx+LEFT]-currentHieght;
		float rightDelta=map[currentIdx+RIGHT]-currentHieght;
		float maxDelta=min(min(upDelta,downDelta),min(leftDelta,rightDelta));
		
		//check for local maximum
		int lowestIdx=currentIdx;
		float lowestVal;
		if(maxDelta<0)
		{
			if(maxDelta==upDelta)
			{
				lowestIdx=currentIdx+UP;
			}
			else if(maxDelta==downDelta)
			{
				lowestIdx=currentIdx+DOWN;
			}
			else if(maxDelta==leftDelta)
			{
				lowestIdx=currentIdx+LEFT;
			}
			else if(maxDelta==rightDelta)
			{
				lowestIdx=currentIdx+RIGHT;
			}
		}
		lowestVal=map[lowestIdx];
		totalHeight+=lowestVal*weights[i];
		totalWeight+=weights[i];
		currentIdx=lowestIdx;
	}
	
	
	//wait for everything else to finish before writing to the buffer
	barrier(CLK_GLOBAL_MEM_FENCE);
	map[startingIdx]=totalHeight/totalWeight;
	
}*/