import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class MyNumUtils {

	//Given an integer, write a function to determine if it is a power of two. 
	public boolean isPowerOfTwo(int n) {
		if (n<=0) return false;
		if ((n==1) || (n==2)) return true;
		int div = n;
		int rem = 0;
		while(div>2){
			rem = div%2;
			div = div/2;
			if (rem!=0) return false;
		}
		return true;
	}


	// Implement pow(x, n). 
	public double myPow(double x, int n) {
		if (n==0) return 1;
		if (n==1) return x;
		boolean isNeg = false;
		if (n<0) isNeg = true;
		double d = x;
		for (int i=1;i<n;i++){
			d = d*x;
		}
		if (isNeg)
			d = 1/d;
		return d;
	}


	/* Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.
	 * 
	For example:
	Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].
	Note:
	    The order of the result is not important. So in the above example, [5, 3] is also correct.
	    Your algorithm should run in linear runtime complexity. Could you implement it using only constant space complexity?
	 */
	public int[] singleNumber(int[] nums) {
		if (nums==null) return null;
		if (nums.length==1) return nums;

		HashMap<Integer,Integer> hm = new HashMap<Integer,Integer>();
		for(int i=0;i<nums.length;i++){
			if (hm.get(nums[i]) ==null){
				hm.put(nums[i], 1);
			}else{
				hm.put(nums[i],hm.get(nums[i])+1);
			}
		}
		ArrayList<Integer> ai = new ArrayList<Integer>();
		for(int j=0;j<nums.length;j++){
			if (hm.get(nums[j])==1)
				ai.add(nums[j]);
		}
		int[] retInt = new int[ai.size()];
		for(int k=0;k<ai.size();k++){
			retInt[k]=ai.get(k).intValue();
		}
		return retInt;
	}


	/*Reverse digits of an integer.

	Example1: x = 123, return 321
	Example2: x = -123, return -321
	 */
	public int reverse(int x) {
		long y=0;
		if (x<0)
			y=0-x;
		else
			y=x;
		long z=0;    
		while(y!=0){    
			z=z*10+y%10;
			y=y/10;
		}
		if (x<0){
			z=0-z;
			if (z<Integer.MIN_VALUE)
				z=0;
		}
		if (z>Integer.MAX_VALUE) 
			z=0; 
		return (int) z;
	}


	public int findMinDifference(List<String> timePoints) {

		if ((timePoints==null)||(timePoints.size()==0)) return 0;

		int[] timeInMin = new int[timePoints.size()];

		for (int i=0; i<timePoints.size(); i++){
			char[] cs = timePoints.get(i).toCharArray();
			timeInMin[i] = (cs[0]-'0')*10*60 + (cs[1]-'0')*60 +(cs[3]-'0')*10 + (cs[4]-'0');

		}

		Arrays.sort(timeInMin);
		int min = Integer.MAX_VALUE;
		for (int j=1; j< timeInMin.length; j++){
			int diff = ((timeInMin[j] - timeInMin[j-1]) < (1440-timeInMin[j] + timeInMin[j-1]))? 
					(timeInMin[j] - timeInMin[j-1]) : (1440-timeInMin[j] + timeInMin[j-1]);
					min = (diff<min)? diff:min;
		}

		int diff0 = ((timeInMin[timeInMin.length-1]-timeInMin[0]) < (1440-timeInMin[timeInMin.length-1]+timeInMin[0]))? 
				(timeInMin[timeInMin.length-1]-timeInMin[0]) : (1440-timeInMin[timeInMin.length-1]+timeInMin[0]);
				min = (diff0<min)? diff0:min;

				return min;
	}

	/*    Given a non-negative integer n, count all numbers with unique digits, x, where 0 ≤ x < 10n.
    Example:
    Given n = 2, return 91. (The answer should be the total numbers in the range of 0 ≤ x < 100, excluding [11,22,33,44,55,66,77,88,99])
	 */
	public int countNumbersWithUniqueDigits(int n) {

		if(n==0) return 1;
		if(n==1) return 10;

		int sum = 10; 
		int base = 9;
		for (int i=2;i<=n;i++){
			base = base*(9-i+2);
			sum = sum+base;
		}
		return sum;
	}


	/*Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].
			Solve it without division and in O(n).
			For example, given [1,2,3,4], return [24,12,8,6].
			*/
	public int[] productExceptSelf(int[] nums) {
        
        if (nums==null) return null;      
        int[] ret = new int[nums.length];
        ret[0] = 1;
        for (int i=1;i<nums.length; i++){
            ret[i]=ret[i-1]*nums[i-1];
        }
        int tmp = 1;
        for(int j=nums.length-1;j>=0;j--){
            ret[j]=tmp*ret[j];
            tmp = tmp*nums[j];
        }
        return ret;
    }


	public static void main(String[] args){

		MyNumUtils mnu = new MyNumUtils();
		List<String> timePoints = new ArrayList<String>();
		timePoints.add("12:01");
		timePoints.add("00:01");
		//timePoints.add("00:35");
		mnu.findMinDifference(timePoints);

	}




}
