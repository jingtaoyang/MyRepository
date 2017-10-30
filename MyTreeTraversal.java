import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Stack;

public class MyTreeTraversal {

	public List<Integer> inorderTraversal(MyTreeNode root) {
		List<Integer> li = new ArrayList<Integer>();
		if (root == null) return li;
		/*if (root.left!=null) li.addAll(inorderTraversal(root.left));
        li.add(root.val);
        if (root.right!=null) li.addAll(inorderTraversal(root.right));
        return li;*/
		Stack<MyTreeNode> st = new Stack<MyTreeNode>();
		MyTreeNode tn=root;
		while(tn != null){
			do {
				st.push(tn);
				tn = tn.left;
			} while (tn != null);
			while(tn == null && !st.empty()) {
				MyTreeNode tmp = st.pop();
				li.add(tmp.val);
				tn = tmp.right;
			}
		}
		return li;
	}



	public List<Integer> preorderTraversal(MyTreeNode root) {
		List<Integer> li = new ArrayList<Integer>();
		if (root == null) return li;
		/*   li.add(root.val);
    if (root.left!=null) li.addAll(preorderTraversal(root.left));
    if (root.right!=null) li.addAll(preorderTraversal(root.right));
    return li;*/
		MyTreeNode cur = root;
		Stack<MyTreeNode> st = new Stack<MyTreeNode>();
		while((cur!=null)||(!st.empty())){
			while(cur!=null){
				li.add(cur.val);
				st.push(cur);
				cur = cur.left;
			}
			while(!st.empty()&&(cur==null)){
				MyTreeNode tn = st.pop();
				if (tn.right!=null)
					cur=tn.right;
			}    
		}
		return li;
	}


	public List<List<Integer>> levelOrder(MyTreeNode root) {

		ArrayList<List<Integer>> ret = new ArrayList<List<Integer>>();
		MyTreeNode parent = root;
		ArrayList<MyTreeNode> childrenL = new ArrayList<MyTreeNode>();
		List<Integer> pL = new ArrayList<Integer>();
		if (root==null) return ret;
		if (root.left!=null) childrenL.add(root.left);
		if (root.right!=null) childrenL.add(root.right);
		pL.add(root.val);
		ret.add(pL);

		while(!childrenL.isEmpty()){
			ArrayList<MyTreeNode> parentL = new ArrayList<MyTreeNode>();
			pL = new ArrayList<Integer>(); 
			for (int i=0; i<childrenL.size();i++){
				pL.add(childrenL.get(i).val);
				if (childrenL.get(i).left!=null) parentL.add(childrenL.get(i).left);
				if (childrenL.get(i).right!=null) parentL.add(childrenL.get(i).right);
			}
			ret.add(pL);
			childrenL = parentL;
		}
		return ret;
	}


	public List<Integer> rightSideView(MyTreeNode root) {

		List<Integer> li = new ArrayList<Integer>();
		if (root==null) return li;
		Queue<MyTreeNode> pQueue = new LinkedList<MyTreeNode>();
		Queue<MyTreeNode> cQueue = new LinkedList<MyTreeNode>();
		Queue<MyTreeNode> tQueue = pQueue;
		pQueue.offer(root);

		while(pQueue.peek()!=null){    
			MyTreeNode tn = pQueue.poll();
			if (tn.left!=null)
				cQueue.offer(tn.left);
			if (tn.right!=null)
				cQueue.offer(tn.right);
			if (pQueue.peek()==null){
				li.add(tn.val);
				tQueue = pQueue;
				pQueue = cQueue;
				cQueue = tQueue;
			}
		}
		return li;
	}


	public MyTreeNode invertTree(MyTreeNode root) {
		if (root==null) return null;
		MyTreeNode left = invertTree(root.left);
		MyTreeNode right = invertTree(root.right);
		root.left = right;
		root.right = left;
		return root;
	}


	public boolean isSameTree(MyTreeNode p, MyTreeNode q) {
		if ((p==null)&&(q==null)) return true;
		if ((p==null)||(q==null)) return false;
		if (p.val != q.val) return false;
		if (isSameTree(p.left,q.left)){
			return isSameTree(p.right,q.right);
		}else{
			return false;
		}
	}


	public int maxDepth(MyTreeNode root) {
		if (root==null) return 0;
		if ((root.left==null)&&(root.right==null)) return 1;

		if (root.left==null) return maxDepth(root.right)+1;
		if (root.right==null) return maxDepth(root.left)+1;

		int maxR = maxDepth(root.right);
		int maxL = maxDepth(root.left);

		int depth = (maxR<maxL)? maxL+1:maxR+1;
		return depth;   
	}

	public int minDepth(MyTreeNode root) {

		if (root==null) return 0;
		if ((root.left==null)&&(root.right==null)) return 1;

		if (root.left==null) return minDepth(root.right)+1;
		if (root.right==null) return minDepth(root.left)+1;

		int minR = minDepth(root.right);
		int minL = minDepth(root.left);

		int depth = (minR>minL)? minL+1:minR+1;
		return depth;

	}

	public MyTreeNode lowestCommonAncestor(MyTreeNode root, MyTreeNode p, MyTreeNode q) {
		if ((root==null)||(p==null)||(q==null))
			return null;
		int val = root.val;
		int pval = p.val;
		int qval = q.val;   
		if ((val==pval)||(val==qval)){
			return root;
		}
		if ((val>pval)&&(val>qval)) 
			return lowestCommonAncestor(root.left, p, q);
		if ((val<pval)&&(val<qval)) 
			return lowestCommonAncestor(root.right, p, q);
		return root;
	}



	/* Given a binary tree, return all root-to-leaf paths.For example, given the following binary tree: 
	 *  1
	 * /   \
	 *2     3
	 * \
	 *	5
	 *  results : ["1->2->5", "1->3"]
	 */
	public List<String> binaryTreePaths(MyTreeNode root) {
		if (root == null) return new ArrayList<String>();
		List<String> ls = new ArrayList<String>();
		List<String> lsl = new ArrayList<String>();
		List<String> lsr = new ArrayList<String>();
		String rootS = root.val + "";

		if ((root.left==null)&&(root.right==null)){
			ls.add(rootS);
			return ls;
		}    
		if (root.left!=null){
			lsl = appendList(root.val, binaryTreePaths(root.left));
		}
		if (root.right!=null) {
			lsr = appendList(root.val, binaryTreePaths(root.right));
		} 
		for (int i=0; i<lsl.size(); i++){
			ls.add(lsl.get(i));
		}
		for (int j=0; j<lsr.size(); j++){
			ls.add(lsr.get(j));
		}
		return ls;
	}

	public List<String> appendList(int val, List<String> ls){
		List<String> ret = new ArrayList<String>();
		for (int i=0; i<ls.size(); i++){
			ret.add(val+"->"+ls.get(i));
		}
		return ret; 
	}

	/*
	 * Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.
	 * Note: You may assume k is always valid, 1 ? k ? BST's total elements.
	 * Follow up: What if the BST is modified (insert/delete operations) often and you need to find 
	 * the kth smallest frequently? How would you optimize the kthSmallest routine?
	 */

	public int kthSmallest(MyTreeNode root, int k) {
		if (root==null) return 0;
		int leftCnt = findNodes(root.left);
		int rightCnt = findNodes(root.right);  
		if (leftCnt>k-1)
			return kthSmallest(root.left, k);
		else if (leftCnt==k-1)
			return root.val;
		else 
			return kthSmallest(root.right, k-leftCnt-1); 
	}

	public int findNodes(MyTreeNode root){
		if (root==null)
			return 0;
		if ((root.left==null)&&(root.right==null)) 
			return 1;
		return findNodes(root.left)+findNodes(root.right)+1;
	}

	/*
	 * Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its bottom-up level order traversal as:
[[15,7],
  [9,20],
  [3]]
	 * 
	 */
	public List<List<Integer>> levelOrderBottom(MyTreeNode root) {

		Queue<MyTreeNode> p = new LinkedList<MyTreeNode>();
		Queue<MyTreeNode> c = new LinkedList<MyTreeNode>();
		List<List<Integer>> lli = new ArrayList<List<Integer>>();
		Queue<Integer> qi = new LinkedList<Integer>();
		Stack<Queue<Integer>> sli = new Stack<Queue<Integer>>();

		if (root==null) return lli;
		p.offer(root);
		qi.offer(root.val);
		sli.push(qi);
		qi =  new LinkedList<Integer>();
		while(!p.isEmpty()){
			MyTreeNode tmp = p.poll();

			if (tmp!=null){
				if (tmp.left!=null){
					c.offer(tmp.left);
					qi.offer(tmp.left.val);
				}
				if (tmp.right!=null){
					c.offer(tmp.right);
					qi.offer(tmp.right.val);
				}
			}
			if (p.isEmpty()){
				sli.push(qi);
				qi =  new LinkedList<Integer>();
				p = c;
				c = new LinkedList<MyTreeNode>();
			}
		}

		while(!sli.empty()){
			Queue<Integer> lt = sli.pop();
			List<Integer> li = new ArrayList<Integer>();
			while(!lt.isEmpty()){
				li.add(lt.poll());
			}
			lli.add(li);
		}
		return lli;

	}

	/*
	 * Given a binary tree, return the tilt of the whole tree.
	The tilt of a tree node is defined as the absolute difference between the sum of all left subtree node
	values and the sum of all right subtree node values. Null node has tilt 0.
	The tilt of the whole tree is defined as the sum of all nodes' tilt.
	Example:Input: 
         1
       /   \
      2     3
	Output: 1
	Explanation: 
	Tilt of node 2 : 0
	Tilt of node 3 : 0
	Tilt of node 1 : |2-3| = 1
	Tilt of binary tree : 0 + 0 + 1 = 1
	 */
	public int findTilt(MyTreeNode root) {
		if (root==null) return 0;
		int ret = Math.abs(sumTree(root.left)-sumTree(root.right)) + findTilt(root.left)+ findTilt(root.right);
		return ret;
	}

	public int sumTree(MyTreeNode root){
		if (root==null) return 0;
		return sumTree(root.left)+sumTree(root.right)+root.val;
	}


	/*
	 *Given the root of a tree, you are asked to find the most frequent subtree sum. 
	 *The subtree sum of a node is defined as the sum of all the node values formed by the subtree 
	 *rooted at that node (including the node itself). So what is the most frequent subtree sum value?
	 * If there is a tie, return all the values with the highest frequency in any order.
  		  5
 		 /  \
		2   -3
	 return [2, -3, 4], since all the values happen only once, return all of them in any order.
  		   5
 		 /  \
		2   -5
	 * return [2], since 2 happens twice, however -5 only occur once. 
	 */
	 public int[] findFrequentTreeSum(MyTreeNode root) {     
	        ArrayList<Integer> freq = new ArrayList<Integer>();
	        HashMap<Integer,Integer> hm = new HashMap<Integer, Integer>();
	        getSum(root, hm);	        
	        int max = -1;
	        for(Integer val:hm.values()){
	            if (val>max)
	                max = val;
	        }
	        for(Integer key:hm.keySet()){
	            if (hm.get(key)==max)
	                freq.add(key);
	        }
	        int[] freqL = new int[freq.size()];
	        for(int i=0;i<freq.size();i++){
	            freqL[i] = freq.get(i);
	        }
	        return freqL;
	    }
	    
	    public Integer getSum(MyTreeNode root, HashMap<Integer,Integer> hm){
	        if (root==null)  return 0;
	        Integer sum = root.val+ getSum(root.left, hm) + getSum(root.right, hm);
	        if (hm.get(sum)==null)
	                hm.put(sum,1);
	            else
	                hm.put(sum, hm.get(sum)+1);
	        return sum;  
	    }
	

	public static void main(String[] args){	   
		MyTreeTraversal mtt = new MyTreeTraversal();
		MyTreeNode n = new MyTreeNode(1);
		MyTreeNode n1 = new MyTreeNode(2);
		MyTreeNode n2 = new MyTreeNode(3);

		n.right = n1;
		n1.left = n2;

		mtt.levelOrderBottom(n);

		List<Integer> li = mtt.inorderTraversal(n);
		for(int i:li){
			System.out.println(i);
		}
	}
}
