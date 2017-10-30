import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

public class MyArrayTraversal {
	
	
	public static void quickSort(int[] arr, int low, int high) {  
		if (arr == null || arr.length == 0) return;    
		if (low >= high) return;   // pick the pivot  
		int middle = low + (high - low) / 2;  
		int pivot = arr[middle];   // make left < pivot and right > pivot  
		int i = low, j = high;  
		while (i <= j) {  
			while (arr[i] < pivot) { i++; }    
			while (arr[j] > pivot) { j--; }    
			if (i <= j) {  
				int temp = arr[i];  
				arr[i] = arr[j];  
				arr[j] = temp;  
				i++;  
				j--;  
			}  
		}   
		// recursively sort two sub parts  
		if (low < j) quickSort(arr, low, j);	 
		if (high > i) quickSort(arr, i, high);  
	} 
		 
	
    private void mergeSort(int[] array, int lowerIndex, int higherIndex) {       
        if (lowerIndex < higherIndex) {
            int middle = lowerIndex + (higherIndex - lowerIndex) / 2;
            // Below step sorts the left side of the array
            mergeSort(array, lowerIndex, middle);
            // Below step sorts the right side of the array
            mergeSort(array, middle + 1, higherIndex);
            // Now merge both sides
            merge(array, lowerIndex, middle, higherIndex);
        }
    }
 
    private void merge(int[] array, int lowerIndex, int middle, int higherIndex) {
 
    	int[] tempMergArr = new int[array.length];   	
        for (int i = lowerIndex; i <= higherIndex; i++) {
            tempMergArr[i] = array[i];
        }
        int i = lowerIndex;
        int j = middle + 1;
        int k = lowerIndex;
        while (i <= middle && j <= higherIndex) {
            if (tempMergArr[i] <= tempMergArr[j]) {
                array[k] = tempMergArr[i];
                i++;
            } else {
                array[k] = tempMergArr[j];
                j++;
            }
            k++;
        }
        while (i <= middle) {
            array[k] = tempMergArr[i];
            k++;
            i++;
        }
    }
    
    
    
    public void heapSort(int arr[])
    {
        int n = arr.length;
 
      	// Build max heap
        for (int i = n / 2 - 1; i >= 0; i--) {
          heapify(arr, n, i);
        }
            
		// Heap sort
        for (int i=n-1; i>=0; i--)
        {
            int temp = arr[0];
            arr[0] = arr[i];
            arr[i] = temp;
 
          	// Heapify root element
            heapify(arr, i, 0);
        }
    }
 
    void heapify(int arr[], int n, int i)
    {
      	// Find largest among root, left child and right child
        int largest = i; 
        int l = 2*i + 1; 
        int r = 2*i + 2;  
 
        if (l < n && arr[l] > arr[largest])
            largest = l;
 
        if (r < n && arr[r] > arr[largest])
            largest = r;
 
      	// Swap and continue heapifying if root is not largest
        if (largest != i)
        {
            int swap = arr[i];
            arr[i] = arr[largest];
            arr[largest] = swap;
            heapify(arr, n, largest);
        }
    }
 

	public boolean containsDuplicate(int[] nums) {
		if (nums==null) return false;
		if (nums.length<1) return false;
		Arrays.sort(nums);
		for (int i=0; i< nums.length-1; i++){
			if (nums[i]==nums[i+1]) return true;
		}
		return false;
	}

	// Given an array of integers and an integer k, find out whether there are two distinct 
	// indices i and j in the 
	// array such that nums[i] = nums[j] and the difference between i and j is at most k. 
	public boolean containsNearbyDuplicate(int[] nums, int k) {
		if (nums==null) return false;
		if (nums.length<=1) return false;

		//value, last index
		HashMap<Integer,Integer> hm = new HashMap<Integer, Integer>();
		int ind = -1;
		for(int i=0; i<nums.length; i++){
			if (hm.containsKey(new Integer(nums[i]))){
				ind = hm.get(nums[i]);
				if ((i-ind)<=k) return true;
				else hm.put(nums[i],i);
			}else{
				hm.put(nums[i],i);
			}
		}
		return false;
	}

	public List<Integer> findDuplicates(int[] nums) {
		HashMap<Integer, Integer> hm = new HashMap<Integer, Integer>();
		for (int i=0; i<nums.length;i++){
			if (hm.get(nums[i])==null)
				hm.put(nums[i],1);
			else
				hm.put(nums[i], hm.get(nums[i])+1);
		}
		List<Integer> l = new ArrayList<Integer>();
		for (Integer key:hm.keySet()){
			if (hm.get(key)==2)
				l.add(key);
		}
		return l;
	}

	/*
	 *  Given a sorted array, remove the duplicates in place, such that each element appear 
	 *  only once and return the new length.Do not allocate extra space for another array, 
	 *  you must do this in place with constant memory.
	 *  Given input array nums = [1,1,2],
	 *  Your function should return length = 2, with the first two elements of nums being 1 
	 *  and 2 respectively.
	 *  It doesn't matter what you leave beyond the new length. 
	 */
	public int removeDuplicates(int[] nums) {
		if (nums.length == 0) return 0;
		int i = 0;
		for (int j = 1; j < nums.length; j++) {
			if (nums[j] != nums[i]) {
				i++;
				nums[i] = nums[j];
			}
		}
		return i + 1;
	}


	public int removeDuplicates1(int[] nums) {
		if ((nums==null) || (nums.length==0))
			return 0;
		int end = nums.length-1;
		int i = 0;
		while(i<end){
			while ((nums[i]==nums[i+1])&&(i<end)){
				int j=i+1;
				while(j<=end-1){
					nums[j]=nums[j+1];
					j++;
				}
				end--;
			}
			i++;
		}
		return end+1;
	}


	/* 
	 * Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted 
	 * array.Note: You may assume that nums1 has enough space (size that is greater or equal 
	 * to m + n) to hold additional elements from nums2. 
	 * The number of elements initialized in nums1 and nums2 are m and n respectively.
	 */
	public void merge(int[] nums1, int m, int[] nums2, int n) {
		if (nums1==null) return;
		if ((nums2==null) ||(n==0)) return;
		int index1 = m-1;
		int index2 = n-1;
		for(int i=m+n-1;i>=0;i--){
			if ((index1)<0) {
				nums1[i]= nums2[index2];
				index2--;
			}else if ((index2)<0) {
				nums1[i] = nums1[index1];
				index1--;
			}else if (nums1[index1]<nums2[index2]){
				nums1[i] = nums2[index2];
				index2--;
			}else if (nums1[index1]>=nums2[index2]){
				nums1[i] = nums1[index1];
				index1--;
			}
		}      
	}


	//Rotate an array of n elements to the right by k steps.
	//For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].

	public void rotate(int[] nums, int k) {
		k %= nums.length;
		reverse(nums, 0, nums.length - 1);
		reverse(nums, 0, k - 1);
		reverse(nums, k, nums.length - 1);
	}

	public void reverse(int[] nums, int start, int end) {
		while (start < end) {
			int temp = nums[start];
			nums[start] = nums[end];
			nums[end] = temp;
			start++;
			end--;
		}
	}


	/**
	 * Definition for a binary tree node.
	 * public class TreeNode {
	 *     int val;
	 *     TreeNode left;
	 *     TreeNode right;
	 *     TreeNode(int x) { val = x; }
	 * }
	 */
	public MyTreeNode sortedArrayToBST(int[] nums) {
		if ((nums==null)||(nums.length==0)) return null;
		return sortedArrayToBST(nums,0,nums.length-1);
	}

	public MyTreeNode sortedArrayToBST(int[] nums, int b, int e){
		if ((b>e)||(b>nums.length-1)||(e>nums.length-1)) return null;
		if (b==e){
			MyTreeNode tn = new MyTreeNode(nums[b]);
			return tn;
		}
		int mid = b+ (e-b)/2;
		MyTreeNode tn = new MyTreeNode(nums[mid]);
		if (b<=mid-1)
			tn.left =  sortedArrayToBST(nums, b, mid-1);
		if (mid+1<=e)
			tn.right = sortedArrayToBST(nums, mid+1, e);
		return tn;
	}


	/*Given a non-empty array of integers, return the k most frequent elements.
	For example,
	Given [1,1,1,2,2,3] and k = 2, return [1,2]. */

	public List<Integer> topKFrequent(int[] nums, int k) {

		List<Integer> li = new ArrayList<Integer>();

		if ((nums==null)||(nums.length==0)) return li;
		HashMap<Integer,Integer> hm = new HashMap<Integer,Integer>();

		for(int i=0;i<nums.length; i++){
			if(hm.get(nums[i])==null)
				hm.put(nums[i],1);
			else
				hm.put(nums[i], hm.get(nums[i])+1);
		}
		Integer[] intA = new Integer[hm.size()];
		int index=0;
		for(Integer key:hm.keySet()){
			intA[index++]=key;
		}

		Arrays.sort(intA, new Comparator<Integer>(){   // (a,b) -> hm.get(b) - hm.get(a)
			public int compare(Integer a, Integer b) {
				return hm.get(b)-hm.get(a);
			}
		});	

		for(int j=0;j<k;j++){
			li.add(intA[j]);
		}
		return li;
	}

	/*
	 * You are given two arrays (without duplicates) nums1 and nums2 where nums1’s elements are subset of nums2. Find all the next greater numbers
	 *  for nums1's elements in the corresponding places of nums2.
The Next Greater Number of a number x in nums1 is the first greater number to its right in nums2. If it does not exist, output -1 for this number.
Example 1:
Input: nums1 = [4,1,2], nums2 = [1,3,4,2].
Output: [-1,3,-1]
Explanation:
    For number 4 in the first array, you cannot find the next greater number for it in the second array, so output -1.
    For number 1 in the first array, the next greater number for it in the second array is 3.
    For number 2 in the first array, there is no next greater number for it in the second array, so output -1.
	 */
	public int[] nextGreaterElement(int[] findNums, int[] nums) {

		if (findNums==null) return null;
		int[] retInt = new int[findNums.length];

		HashMap<Integer,Integer> hm = new HashMap<Integer,Integer>();
		for(int i=0;i<nums.length-1;i++){
			hm.put(nums[i],-1);
			for (int j=i+1; j<nums.length;j++){
				if (nums[j]>nums[i]){
					hm.put(nums[i],nums[j]);
					break;
				}
			}
		}
		hm.put(nums[nums.length-1],-1);
		for (int k=0; k<findNums.length; k++){
			if ((hm.get(findNums[k])==null))
				retInt[k] = -1;
			else 
				retInt[k] = (int) hm.get(findNums[k]);
		}
		return retInt;
	}

	public static void main(String[] args){

		MyArrayTraversal mat= new MyArrayTraversal();
		
		mat.nextGreaterElement(new int[]{4,1,2}, new int[]{1,3,4,2});
		
		int[] nums = new int[]{1,1,1,2,2,3};
		mat.topKFrequent(nums, 2);
	}
}
