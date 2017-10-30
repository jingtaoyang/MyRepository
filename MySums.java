import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

public class MySums {
	
	
	/*Given an array of integers, find two numbers such that they add up to a specific target number.
	The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.
	You may assume that each input would have exactly one solution.
	Input: numbers={2, 7, 11, 15}, target=9
	Output: index1=1, index2=2
	*/

	public int[] twoSum(int[] nums, int target) {
	        if ((nums==null) || (nums.length==0))
	            return null;
	        
	        for (int i=0;i<nums.length;i++){
	            int j = i+1;
	            while(j<nums.length){
	                if (nums[i]+nums[j]==target){
	                    return new int[]{i+1,j+1};
	                }
	                j++;
	            }
	        }
	        return null;   
	}
	


	/*You are given two linked lists representing two non-negative numbers. 
	 * The digits are stored in reverse order and each of their nodes contain a single digit. 
	 * Add the two numbers and return it as a linked list.
	Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
	Output: 7 -> 0 -> 8

	public class ListNode {
	      int val;
	      ListNode next;
	      ListNode(int x) { val = x; }
	}
	 */
	public MyListNode addTwoNumbers(MyListNode l1, MyListNode l2) {
		if (l1==null) return l2;
		if (l2==null) return l1;
		int a=l1.val;
		int b=l2.val;
		int inc=(a+b)/10;
		int rem=(a+b)%10;
		MyListNode l3 = new MyListNode(rem);
		MyListNode lc = l3;
		l1=l1.next;
		l2=l2.next;
		while((l1!=null)||l2!=null||(inc!=0)){
			a = (l1==null)? 0:l1.val;
			b = (l2==null)? 0:l2.val;
			rem = (a+b+inc)%10;
			inc = (a+b+inc)/10;
			lc.next = new MyListNode(rem);
			lc = lc.next;
			l1 = (l1!=null)? l1.next:null;
			l2 = (l2!=null)? l2.next:null;
		}
		return l3;
	}

	/*
	 * You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
	    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
	    Follow up:
	    What if you cannot modify the input lists? In other words, reversing the lists is not allowed.
	    Example:
	    Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
	    Output: 7 -> 8 -> 0 -> 7
	 */   

	public MyListNode addTwoNumbersII(MyListNode l1, MyListNode l2) {
		if (l1==null) return l2;
		if (l2==null) return l1;

		Stack s1 = new Stack();
		Stack s2 = new Stack();

		while((l1!=null)||(l2!=null)){
			if (l1!=null){
				s1.push(l1.val);
				l1 = l1.next;
			}
			if (l2!=null){
				s2.push(l2.val);
				l2 = l2.next;
			}
		}

		int inc = 0;
		int rem = 0;
		int v1 = 0;
		int v2 = 0;
		MyListNode pre = null;
		MyListNode cur = null;

		while((!s1.isEmpty())||(!s2.isEmpty())){
			if (s1.isEmpty()==false){
				v1 = (int) s1.pop();
			}else
				v1 = 0;
			if (s2.isEmpty()==false){
				v2 = (int) s2.pop();
			}else
				v2 = 0;
			rem = (v1+v2+inc)%10;
			inc = (v1+v2+inc)/10;
			cur = new MyListNode(rem);
			cur.next = pre;
			pre = cur;
		}

		if (inc>0) {
			MyListNode first = new MyListNode(inc);
			first.next = cur;
			cur = first;
		}
		return cur;    
	}


	/*
	 * Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
	 Note:
	    Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
	    The solution set must not contain duplicate triplets.
	    For example, given array S = {-1 0 1 2 -1 -4},
	    A solution set is:
	    (-1, 0, 1)
	    (-1, -1, 2)
	 */
	public List<List<Integer>> threeSum(int[] nums) {    
		if (nums==null) return null;
		if (nums.length<3) return null;
		ArrayList<List<Integer>> ret = new ArrayList<List<Integer>>();
		ArrayList<Integer> tmp = new ArrayList<Integer>();
		Arrays.sort(nums);
		for(int i=0; i< nums.length; i++){
			int cur3 = nums[i];
			int sum2 = 0 - cur3;
			for(int j=i+1;j<nums.length; j++){
				for(int k=j+1;k<nums.length;k++){
					if (nums[k]+nums[j] == sum2){
						tmp = new ArrayList<Integer>();
						tmp.add(i);
						tmp.add(j);
						tmp.add(k);
						ret.add(tmp);
					}
				}
			}
		}
		return ret;   
	}

	public List<List<Integer>> threeSum1(int[] nums) {

		if (nums == null || nums.length == 0) {
			return new ArrayList<List<Integer>>();
		}
		// algo - sort the array, then fix one element, and check if rest pairs
		// sum to (target-num[i])
		Arrays.sort(nums);
		List<List<Integer>> result = new ArrayList<List<Integer>>();

		// compare target-num[i] against num[j]+num[k] , j starts from i+1, k
		// starts from end of array
		for (int i = 0; i < nums.length - 2; i++) {
			for (int j = i + 1, k = nums.length - 1; j < k;) {
				if (nums[j] + nums[k] == 0 - nums[i]) {
					List<Integer> list = new ArrayList<Integer>();
					list.add(nums[i]);
					list.add(nums[j]);
					list.add(nums[k]);
					j++;
					k--;
					// avoid duplicate triplet
					//if (!containsTriplet(result, list)) {
					result.add(list);
					//}
				} else if (nums[j] + nums[k] < 0 - nums[i]) {
					j++;
				} else {
					k--;
				}
			}
		}
		return result;
	}


	/*Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero.
	To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.
	Example:
	Input:
	A = [ 1, 2]
	B = [-2,-1]
	C = [-1, 2]
	D = [ 0, 2]
	Output:
	2
	Explanation:
	The two tuples are:
	1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
	2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
	 */
	public int fourSumCount(int[] A, int[] B, int[] C, int[] D) {
		int ret = 0;
		HashMap<Integer,Integer> hm = new HashMap<Integer,Integer>();
		for (int i=0; i<A.length; i++){
			for(int j=0;j<B.length; j++){
				int sum = A[i]+B[j];
				if (hm.get(sum)==null){
					hm.put(sum,1);
				}else{
					hm.put(sum, hm.get(sum)+1);
				}
			}
		}

		for (int i=0; i<C.length; i++){
			for(int j=0;j<D.length; j++){
				int sum = 0 - (C[i]+D[j]);
				if (hm.containsKey(sum)){
					ret = ret + hm.get(sum);
				}
			}
		}
		return ret;
	}


	/* Given two binary strings, return their sum (also a binary string).
	For example,
	a = "11"
	b = "1"
	Return "100". 
	 */		
	public String addBinary(String a, String b) {

		if (a==null) return b;
		if (b==null) return a;
		String c ="";
		int aL = a.length()-1;
		int bL = b.length()-1;
		char carry = '0';

		while((aL>=0)||(bL>=0)){
			char ac = '0';
			char bc = '0';
			ac = aL>=0?a.charAt(aL):'0';
			bc = bL>=0?b.charAt(bL):'0';
			if ((ac=='0')&&(bc=='0')){
				c = carry+c;
				carry = '0';
			}else if ((ac=='1')&&(bc=='1')){
				c = carry + c;
				carry = '1';
			}else{
				if (carry=='1'){
					c = '0'+c;
				}else{
					c = '1'+c;
				}
			}
			aL--;
			bL--;
		}
		if (carry=='1') c = '1' + c;
		return c;
	}


	public String addBinary1(String a, String b) {

		if (a==null) return b;
		if (b==null) return a;
		String c ="";
		int aL = a.length()-1;
		int bL = b.length()-1;
		int carry = 0;
		int rem = 0;
		int ac = 0;
		int bc = 0;

		while((aL>=0)||(bL>=0)){
			ac = aL>=0?a.charAt(aL)-'0':0;
			bc = bL>=0?b.charAt(bL)-'0':0;
			rem = (ac+bc+carry)%2;
			carry = (ac+bc+carry)/2;
			c = rem + c;
			aL--;
			bL--;
		}
		if (carry==1) c = '1' + c;
		return c;
	}

	 
	/*
	You are given a list of non-negative integers, a1, a2, ..., an, and a target, S. Now you have 2 symbols + and -. For each integer, you should choose one from + and - as its new symbol.
	Find out how many ways to assign symbols to make sum of integers equal to target S.
	Example 1:
	Input: nums is [1, 1, 1, 1, 1], S is 3. 
	Output: 5
	Explanation: 
	-1+1+1+1+1 = 3
	+1-1+1+1+1 = 3
	+1+1-1+1+1 = 3
	+1+1+1-1+1 = 3
	+1+1+1+1-1 = 3
	There are 5 ways to assign symbols to make the sum of nums be target 3.
	 */
	public int findTargetSumWays(int[] nums, int S) {

		if (nums==null) return 0;
		if ((nums.length==0) && (S==0)) return 0;
		Stack<Integer> s1 = new Stack<Integer>();
		Stack<Integer> s2 = new Stack<Integer>();
		s1.push(nums[0]);
		s1.push(0-nums[0]);
		int index = 1;
		while(index<nums.length){
			int a = nums[index];
			while(!s1.empty()){
				int b = s1.pop();
				s2.push(b+a);
				s2.push(b-a);
			}
			if (s1.empty()){
				s1 = s2;
				s2 = new Stack<Integer>();
			}
			index++;
		}
		int cnt=0;
		while(!s1.empty()){
			int c = s1.pop();
			if (c==S) cnt++;
		}
		return cnt;
	}


	public static void main(String[] args){

		MySums ms = new MySums();
		int[] input = {1, 1, 1, 1, 1};
		ms.findTargetSumWays(input, 3);
		
		String s1 ="1";
		String s2 ="1";
		
		ms.addBinary1(s1, s2);


	}
}
