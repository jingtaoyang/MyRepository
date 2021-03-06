/*
 *  Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].
Note:
    You must do this in-place without making a copy of the array.
    Minimize the total number of operations.
 */

public class MyMoveZones {
	
	public void moveZeroes(int[] nums) {   
        if (nums==null) return;
        if (nums.length<2) return;
        int i=0;
        int j=1;
        while(j<=nums.length-1){
            if (nums[i]!=0){
                i++;
                j=i+1;
            }else if(nums[j]==0){
                j++;
            }else{
                nums[i] = nums[j];
                nums[j] = 0; 
            }
        }
    }
}
