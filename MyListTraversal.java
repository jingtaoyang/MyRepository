import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class MyListTraversal {

	/*Write a function to delete a node (except the tail) in a singly linked list, 
	 * given only access to that node. Supposed the linked list is 1 -> 2 -> 3 -> 4 and 
	 * you are given the third node with value 3, the linked list should 
         become 1 -> 2 -> 4 after calling your function.
	 */
	public void deleteNode(MyListNode node) {
		if (node==null) return;
		if (node.next==null) return;

		node.val = node.next.val;
		node.next = node.next.next;
	}

	// Remove Duplicates from Sorted List
	public MyListNode deleteDuplicates(MyListNode head) {
		if (head==null) return null;
		if (head.next==null) return head;
		MyListNode fNode = head.next;
		MyListNode sNode = head;
		while(fNode!=null){
			if (fNode.val == sNode.val){
				fNode = fNode.next;
				sNode.next = fNode;
			}else{
				fNode=fNode.next;
				sNode=sNode.next;
			}
		}
		return head;
	}


	// Remove Duplicate from linkedlist
	public MyListNode deleteDuplicatesI(MyListNode head) {
		if (head==null) return head;

		HashMap<Integer, Integer> hm = new HashMap<Integer,Integer>();
		MyListNode ret = head;
		hm.put(head.val,1);
		while(head.next!=null){
			int value = head.next.val;
			if (hm.get(value)==null) {
				hm.put(value,1);
				head = head.next;
			}else{
				head.next = head.next.next;
			}
		}
		return ret;
	}


	// Reverse LinkedList
	public MyListNode reverseList(MyListNode head) {

		if (head==null) return head;
		if (head.next==null) return head;

		MyListNode newHead = head;
		MyListNode next = head.next;
		MyListNode tmp = next;
		newHead.next = null;
		while(next!=null){
			tmp = next.next;
			next.next = newHead;
			newHead = next;
			next = tmp;
		}
		return newHead;
	}


	// Given a singly linked list, determine if it is a palindrome.
	public boolean isPalindrome(MyListNode head) {

		if ((head==null)||(head.next==null))
			return true;

		MyListNode fast = head;
		MyListNode slow = head;
		MyListNode tmp = head;
		MyListNode pre = null;

		while((fast!=null)&&(fast.next!=null)){
			slow = slow.next;
			fast = fast.next.next;
			tmp.next = pre;
			pre = tmp;
			tmp = slow;
		}

		if ((fast!=null)&&(fast.next==null)){ // odd
			slow = slow.next;
		}

		while((pre!=null)&&(slow!=null)){ // compare pre and slow
			if (pre.val != slow.val)
				return false;
			pre = pre.next;
			slow = slow.next;
		}

		if ((pre==null)&&(slow==null))
			return true;

		return false;
	}


	// LinkedList has Cycle
	public boolean hasCycle(MyListNode head) {
		if (head==null) return false;
		if (head.next==null) return false;

		MyListNode fNode = head;
		MyListNode sNode = head;

		while(fNode!=null){
			fNode = fNode.next;
			if (fNode!=null)
				fNode = fNode.next;
			sNode = sNode.next;
			if (fNode == sNode)
				return true;
		}

		return false;
	}

	/*
	Merge two sorted linked lists and return it as a new list. The new list 
	should be made by splicing together the nodes of the first two lists.		
	 */
	public MyListNode mergeTwoLists(MyListNode l1, MyListNode l2) {

		if (l1==null) return l2;
		if (l2==null) return l1;

		MyListNode tmp = new MyListNode(-1);
		MyListNode l3 = tmp;

		while((l1!=null)&&(l2!=null)){
			if (l1.val <= l2.val){
				tmp.val = l1.val;
				l1= l1.next;
			}else {
				tmp.val = l2.val;
				l2= l2.next;
			}       
			if ((l1==null)&&(l2==null))
				tmp.next = null;
			else
				tmp.next = new MyListNode(-1);
			tmp = tmp.next;
		}
		if (l1!=null){
			tmp.val = l1.val;
			tmp.next = l1.next;
		}if (l2!=null){
			tmp.val = l2.val;
			tmp.next = l2.next;
		}

		return l3;
	}

	public MyListNode mergeTwoLists1(MyListNode l1, MyListNode l2) {

		if (l1==null) return l2;
		if (l2==null) return l1;

		MyListNode tmp = new MyListNode(-1);
		MyListNode l3 = tmp;

		while((l1!=null)&&(l2!=null)){
			if (l1.val <= l2.val){
				l3.next = l1;
				l1= l1.next;
			}else {
				l3.next = l2;
				l2= l2.next;
			}       
			l3=l3.next;
		}
		if (l1!=null){
			l3.next = l1;
		}if (l2!=null){
			l3.next = l2;
		}

		return tmp.next;
	}

	// Insertion sort of a linkedlist
	public MyListNode insertionSortList(MyListNode head) {

		if (head==null) return head;

		MyListNode tmp = new MyListNode(0);
		tmp.next = head;
		MyListNode cur = head;
		MyListNode next = head.next;

		while(next!=null){

			if (cur.val<=next.val){
				cur=next;
				next=next.next;
			}else{
				MyListNode newN = new MyListNode(next.val);
				head = tmp.next;
				if (head.val>=newN.val){
					tmp.next = newN;
					newN.next = head;
					//head = tmp.next;
					next=next.next;
					cur.next=next;
					continue;
				}

				while(head!=cur){
					if ((head.val<newN.val)&&(head.next.val>=newN.val)){

						newN.next = head.next;
						head.next = newN;
						next = next.next;
						cur.next = next;
						//head = tmp.next;
						break;
					}else{
						head = head.next;
					}
				} 
			}            
		}
		return tmp.next;   
	}


	// Convert Sorted LinkedList to BST
	public MyTreeNode sortedListToBST(MyListNode head) {
		if(head==null) return null;
		return toBST(head,null);
	}
	public MyTreeNode toBST(MyListNode head, MyListNode tail){
		MyListNode slow = head;
		MyListNode fast = head;
		if(head==tail) return null;

		while(fast!=tail&&fast.next!=tail){
			fast = fast.next.next;
			slow = slow.next;
		}
		MyTreeNode thead = new MyTreeNode(slow.val);
		thead.left = toBST(head,slow);
		thead.right = toBST(slow.next,tail);
		return thead;
	}


	/*Given a binary tree, flatten it to a linked list in-place.
         1
        / \
       2   5
      / \   \
     3   4   6
	The flattened tree should look like:
   1
    \
     2
      \
       3
        \
         4
          \
           5
            \
             6
	 */
	public void flatten(MyTreeNode root) {
		if (root == null) return;
		MyTreeNode left = root.left;
		MyTreeNode right = root.right;
		root.left = null;

		flatten(left);
		flatten(right);
		root.right = left;
		MyTreeNode cur = root;
		while (cur.right != null) cur = cur.right;
		cur.right = right;
	}


	/*Given a list, rotate the list to the right by k places, 
	 *  where k is non-negative.
		For example:
		Given 1->2->3->4->5->NULL and k = 2,
		return 4->5->1->2->3->NULL.
	 */
	public MyListNode rotateRight(MyListNode head, int k) {
		if ((head==null)|| (k==0)) return head;
		int len = 1;
		MyListNode tail = head;
		while(tail.next!=null){
			len++;
			tail=tail.next;
		}
		k=k%len;
		if (k==0) return head;
		int cnt = 0;
		MyListNode tmp = head;
		tail.next=head;
		while(tmp.next!=null){
			if (cnt==len-k-1){
				head = tmp.next;
				tmp.next = null;
				break;
			}
			tmp = tmp.next;
			cnt++;
		}
		return head;
	}

	/*
	 * Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
	If n = 4 and k = 2, a solution is:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
	 */
	public List<List<Integer>> combine(int n, int k) {

		List<Integer> li = new ArrayList<Integer>();  
		List<List<Integer>> lli = new ArrayList<List<Integer>>();
		List<List<Integer>> lli1 = new ArrayList<List<Integer>>();

		if ((k==0)||(n==0)) return lli;
		if (k>n) return lli;
		if (k==n) {
			for (int i=1;i<n+1;i++)
				li.add(i);
			lli.add(li);
			return lli;
		}
		if (k==1){
			for (int j=1;j<n+1;j++){
				li = new ArrayList<Integer>(); 
				li.add(j);
				lli.add(li);
			}
			return lli;
		}

		lli = combine(n-1,k);
		lli1 = combine(n-1,k-1);
		for (List<Integer> li1: lli1){
			li1.add(n);
			lli.add(li1);
		}
		return lli;   
	}

	/*
	 * Pascal's Triangle II
	 * Given an index k, return the kth row of the Pascal's triangle.
	 * For example, given k = 3,
	 * Return [1,3,3,1].
	 */
	public List<Integer> getRow(int rowIndex) { 
		List<Integer> li = new ArrayList<Integer>();
		if (rowIndex == 0){
			li.add(1);
			return li;
		}
		if (rowIndex == 1){
			li.add(1);
			li.add(1);
			return li;
		}

		li = getRow(rowIndex-1);
		List<Integer> li1 = new ArrayList<Integer>();
		li1.add(1);
		for (int i=1; i<rowIndex; i++){
			li1.add(li.get(i-1)+li.get(i));
		}
		li1.add(1);
		return li1;
	}

	
	//Given a singly linked list, return a random node's value from the linked list. Each node 
	//must have the same probability of being chosen.
	public class Solution {
	    
	    int len=0;
	    HashMap<Integer,Integer> hm = new HashMap<Integer,Integer>();
	    /** @param head The linked list's head.
	        Note that the head is guaranteed to be not null, so it contains at least one node. */
	    public Solution(MyListNode head) {
	        int cnt=0;
	        while(head!=null){
	            this.hm.put(cnt,head.val);
	            head=head.next;
	            cnt++;
	        }
	        this.len = cnt;
	    }
	    
	    /** Returns a random node's value. */
	    public int getRandom() {
	        int num = (int) (Math.random()*this.len);
	        return this.hm.get(num);
	    }
	}			

	public static void main(String[] args){

		MyListTraversal mlt = new MyListTraversal();
		mlt.combine(4, 2);

	}


	/**
	 * // This is the interface that allows for creating nested lists.
	 * // You should not implement it, or speculate about its implementation
	 * public interface NestedInteger {
	 *
	 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
	 *     public boolean isInteger();
	 *
	 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
	 *     // Return null if this NestedInteger holds a nested list
	 *     public Integer getInteger();
	 *
	 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
	 *     // Return null if this NestedInteger holds a single integer
	 *     public List<NestedInteger> getList();
	 * }
	 */
	/*	public class NestedIterator implements Iterator<Integer> {

	    List<NestedInteger> lni = new ArrayList<NestedInteger>();
	    List<Integer> fi = new ArrayList<Integer>();
	    int listIndex = -1;

	    public NestedIterator(List<NestedInteger> nestedList) {
	        this.lni = nestedList;

	        List<NestedInteger> tmp1 = lni;
	        List<NestedInteger> tmp2 = new ArrayList<NestedInteger>();
	        NestedInteger tmpInt = new NestedInteger();

	        while(true){
	            boolean moreNest = false;
	            for(int i=0;i<tmp1.size();i++){
	                tmpInt = tmp1.get(i);
	                if (tmpInt.isInteger()){
	                    tmp2.add(tmpInt);
	                }else{
	                    tmp2.addAll(tmpInt.getList());
	                    moreNest = true;
	                }
	            }
	            tmp1 = tmp2;
	            tmp2 = new ArrayList<NestedInteger>();
	            if (moreNest==false)
	                break;
	        }

	        for(int i=0;i<tmp1.size();i++){
	            fi.add(tmp1.get(i).getInteger());
	        }

	    }

	    @Override
	    public Integer next() {
	        int len = fi.size();
	        if ((len==0)||(len==listIndex +1))
	            return null;
	        else{
	            listIndex++;
	            return fi.get(listIndex);
	        }
	    }

	    @Override
	    public boolean hasNext() {
	        int len = fi.size();
	        if ((len==0)||(len==listIndex +1))
	            return false;
	        else{
	            return true;
	        }

	    }
	} */

	/**
	 * Your NestedIterator object will be instantiated and called as such:
	 * NestedIterator i = new NestedIterator(nestedList);
	 * while (i.hasNext()) v[f()] = i.next();
	 */



}
