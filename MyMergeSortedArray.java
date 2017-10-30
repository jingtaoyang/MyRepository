/* 
 * Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
Note:
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. 
The number of elements initialized in nums1 and nums2 are m and n respectively.
 */
public class MyMergeSortedArray {
	
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
}
