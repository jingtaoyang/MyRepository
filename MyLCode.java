import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

/**
 * This class contains all leetcode questions and accepted answers
 *
*/
public class MyLCode {

    /*
     * OJ's Binary Tree Serialization: The serialization of a binary tree
     * follows a level order traversal, where '#' signifies a path terminator
     * where no node exists below.
     *
    //@formatter:off
    Here's an example:
       1
      / \
     2   3
        /
       4
        \
         5
    The above binary tree is serialized as "{1,2,3,#,#,4,#,#,5}".
     */
    //@formatter:on

    // Note: Preorder is easy with recursive; Level order is bit difficult in
    // that
    // level order serialize - the core is how to define the position of
    // dummy
    public List<String> serializeBinaryTree_level_order(TreeNode node) {
        // level order traversal
        // use null to represent '#' dummy node
        if (node == null) {
            return new ArrayList<String>();
        }
        List<String> list = new ArrayList<String>();
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(node);
        while (!q.isEmpty()) {
            TreeNode n = q.poll();
            if (n == null) {
                list.add("#");
                continue;
            } else {
                list.add(String.valueOf(n.val));
            }
            // use null to represent '#'
            q.add(n.left == null ? null : n.left);
            q.add(n.right == null ? null : n.right);
        }
        return list;
    }

    public TreeNode deSerializeBinaryTree_level_order(List<String> list) {
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        TreeNode root = new TreeNode(Integer.parseInt(list.remove(0)));
        q.offer(root);

        while (!q.isEmpty()) {
            TreeNode n = q.poll();
            if (!list.isEmpty()) {
                String s = list.remove(0);
                if (!s.equals("#")) {
                    TreeNode left = new TreeNode(Integer.parseInt(s));
                    n.left = left;
                    q.offer(left);
                }
            }
            if (!list.isEmpty()) {
                String s = list.remove(0);
                if (!s.equals("#")) {
                    TreeNode right = new TreeNode(Integer.parseInt(s));
                    n.right = right;
                    q.offer(right);
                }
            }
        }
        return root;
    }

    // serialize tree node list
    List<String> serialiedList = new ArrayList<String>();

    // use pre-order traversal
    public List<String> serializeBinaryTree_preorder(TreeNode node) {
        if (node == null) {
            serialiedList.add("#");
            return serialiedList;
        } else {
            serialiedList.add(String.valueOf(node.val));
        }
        serializeBinaryTree_preorder(node.left);
        serializeBinaryTree_preorder(node.right);
        return serialiedList;
    }

    // desirialized root
    TreeNode deserializedRoot = null;

    // deserialize preorder
    public TreeNode deSerializeBinaryTree_preorder(List<String> list) {
        if (list.size() == 0) {
            return deserializedRoot;
        }

        String s = list.remove(0);
        if (s.equals("#")) {
            return null;
        }

        TreeNode node = new TreeNode(Integer.parseInt(s));
        if (deserializedRoot == null) {
            deserializedRoot = node;
        }
        node.left = deSerializeBinaryTree_preorder(list);
        node.right = deSerializeBinaryTree_preorder(list);
        return node;
    }

    /**
     * Max product - a subarray has maxmiaml product
     *
     */
    public int maxProduct(int[] A) {
        if (A == null) {
            throw new IllegalArgumentException();
        }
        if (A.length == 0) {
            return 0;
        }

        int max = A[0];
        int min = A[0];
        int result = A[0]; // or Integer.MIN_VALUE

        for (int i = 1; i < A.length; i++) {
            if (A[i] < 0) {
                // swap max and min
                // that is because to multiple negative number the small number
                // becomes bigger and big number becomes smaller, keep recording
                // smallest and biggest both as smallest is neg and has max
                // abs() value, hence could flip to positive with next neg
                int tmp = max;
                max = min;
                min = tmp;
            }
            max = Math.max(A[i], A[i] * max);
            min = Math.min(A[i], A[i] * min);
            result = Math.max(result, max);
        }

        return result;
    }

    /**
     * Largest Rectangle histogram
     *
     * Given n non-negative integers representing the histogram's bar height
     * where the width of each bar is 1, find the area of largest rectangle in
     * the histogram.
     *
     */
    public int largestRectangleArea(int[] arr) {
        // use a stack to hold index, keep pushing when height increase;
        // otherwise pop until height is less, and compute
        // historical max;
        // return two element array index (start, end, area)
        int[] result = new int[3];
        result[0] = 0; // start index
        result[1] = 0; // end index
        result[2] = 0; // max area = arr[result[0]] * arr[result[1]]

        // stack holds the array index
        Stack<Integer> indexStack = new Stack<Integer>();

        // push to stack and update max
        for (int i = 0; i < arr.length; i++) {
            if (indexStack.isEmpty() || arr[i] >= arr[indexStack.peek()]) {
                indexStack.push(i);
                continue;
            }

            // pop until top of stack is smaller than i in terms of arr[i]
            int stopIndex = indexStack.peek();
            while (!indexStack.isEmpty() && arr[indexStack.peek()] > arr[i]) {
                // trick- compute the value between two stack elements [start,
                // stop]
                // current element should expand left to the next top of stack
                // element + 1, and right to the stopIndex (always)
                int currentIndex = indexStack.pop();
                int startIndex = indexStack.isEmpty() ? 0 : indexStack.peek() + 1;
                // update local max
                int area = arr[currentIndex] * (stopIndex - startIndex + 1);
                if (area > result[2]) {
                    result[0] = startIndex;
                    result[1] = stopIndex;
                    result[2] = area;
                }
            }
            indexStack.push(i);
        }

        // compute the elements in stack
        if (indexStack.isEmpty()) {
            return result[2];
        }
        int stopIndex = indexStack.peek();
        while (!indexStack.isEmpty()) {
            int currentIndex = indexStack.pop();
            int startIndex = indexStack.isEmpty() ? 0 : indexStack.peek() + 1;
            // update local max
            int area = arr[currentIndex] * (stopIndex - startIndex + 1);
            if (area > result[2]) {
                result[0] = startIndex;
                result[1] = stopIndex;
                result[2] = area;
            }
        }
        return result[2];
    }

    /**
     * Two Sum - 2Sum
     *
     * Given an array of integers, find two numbers such that they add up to a
     * specific target number. The function twoSum should return indices of the
     * two numbers such that they add up to the target, where index1 must be
     * less than index2. Please note that your returned answers (both index1 and
     * index2) are not zero-based.
     *
     * You may assume that each input would have exactly one solution. Input:
     * numbers={2, 7, 11, 15}, target=9 Output: index1=1, index2=2
     *
     * @param numbers
     * @param target
     * @return
     *
     * @status ACCEPTED
     */
    public int[] twoSum(int[] numbers, int target) {
        if (numbers == null || numbers.length == 0) {
            return null;
        }

        int[] result = new int[2];
        // <value, index>
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();

        for (int i = 0; i < numbers.length; i++) {
            if (map.containsKey(target - numbers[i])) {
                int index = map.get(target - numbers[i]);
                int index1 = Math.min(index, i);
                int index2 = Math.max(index, i);
                result[0] = ++index1;
                result[1] = ++index2;
                return result;
            } else {
                if (!map.containsKey(numbers[i])) {
                    map.put(numbers[i], i);
                }
            }
        }

        return null;
    }

    /**
     * Evaluate Reverse Poland Notation
     *
     * Evaluate the value of an arithmetic expression in Reverse Polish
     * Notation. Valid operators are +, -, *, /. Each operand may be an integer
     * or another expression.
     *
     * Some examples: ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9 ["4",
     * "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            throw new IllegalArgumentException();
        }
        Stack<String> stack = new Stack<String>();
        for (String s : tokens) {
            if (isOperand(s)) {
                int op2 = Integer.parseInt(stack.pop());
                int op1 = Integer.parseInt(stack.pop());
                int result = execute(op1, op2, s);
                stack.push(String.valueOf(result));
            } else {
                stack.push(s);
            }
        }

        return (Integer.parseInt(stack.pop()));

    }

    private boolean isOperand(String s) {
        return s.equals("+") || s.equals("-") || s.equals("*") || s.equals("/");
    }

    private int execute(int op1, int op2, String s) {
        if (s.equals("+")) {
            return op1 + op2;
        }
        if (s.equals("-")) {
            return op1 - op2;
        }
        if (s.equals("*")) {
            return op1 * op2;
        }
        if (s.equals("/")) {
            return op1 / op2;
        }
        throw new IllegalArgumentException();
    }

    /**
     * Post order traversal
     *
     * Post order traverse a binary tree using iterative
     *
     */
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    public List<Integer> postorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<Integer>();
        }
        Stack<TreeNode> stack = new Stack<TreeNode>();
        List<Integer> result = new ArrayList<Integer>();
        TreeNode visited = null;
        TreeNode node = root;

        while (!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.push(node);
                node = node.left;
            } else {
                TreeNode peek = stack.peek();
                if (peek.right != null && peek.right != visited) {
                    node = peek.right;
                } else { // visit
                    peek = stack.pop();
                    result.add(peek.val);
                    visited = peek;
                }
            }
        }
        return result;
    }

    /**
     * Preorder traversal with iterative
     *
     */
    public List<Integer> preorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<Integer>();
        }
        List<Integer> result = new ArrayList<Integer>();
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode node = root;
        stack.push(node);
        while (!stack.isEmpty()) {
            node = stack.pop();
            result.add(node.val);
            if (node.right != null) {
                stack.push(node.right);
            }
            if (node.left != null) {
                stack.push(node.left);
            }
        }
        return result;
    }

    /**
     * Given a m x n matrix, if an element is 0, set its entire row and column
     * to 0. Do it in place.
     */
    public void zeroRow(int i, int[][] matrix) {
        for (int k = 0; k < matrix[i].length; k++) {
            matrix[i][k] = 0;
        }
    }

    public void zeroColum(int j, int[][] matrix) {
        for (int k = 0; k < matrix.length; k++) {
            matrix[k][j] = 0;
        }
    }

    public void setZeroes(int[][] matrix) {
        Set<Integer> rows = new HashSet<Integer>();
        Set<Integer> cols = new HashSet<Integer>();

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                if (matrix[i] != null) {
                    if (matrix[i][j] == 0) {
                        rows.add(i);
                        cols.add(j);
                    }
                }
            }
        }
        for (int i : rows) {
            zeroRow(i, matrix);
        }
        for (int j : cols) {
            zeroColum(j, matrix);
        }
    }

    /**
     * LRUCache
     *
     * Design and implement a data structure for Least Recently Used (LRU)
     * cache. It should support the following operations: get and set.
     *
     * get(key) - Get the value (will always be positive) of the key if the key
     * exists in the cache, otherwise return -1.
     *
     * set(key, value) - Set or insert the value if the key is not already
     * present. When the cache reached its capacity, it should invalidate the
     * least recently used item before inserting a new item.
     *
     */

    // add following line in leetcode since it doesn't add linkedhashmap class
    // by default
    // import java.util.LinkedHashMap;

    public class LRUCache {
        private LinkedHashMap<Integer, Integer> cache = null;
        private int capacity = 0;

        public LRUCache(int capacity) {
            cache = new LinkedHashMap<Integer, Integer>();
            this.capacity = capacity;
        }

        public int get(int key) {
            int value = -1;
            if (cache.containsKey(key)) {
                // use this key and shuffle it by adding it to the end of the
                // linked hashmap, thus it becomes most recently used, and the
                // head of linked hashmap is always unused.
                value = cache.get(key);
                cache.remove(key);
                cache.put(key, value);
            }
            return value;
        }

        public void set(int key, int value) {
            if (cache.containsKey(key)) {
                // flush away stale data
                cache.remove(key); // remove old key to reflect recent usage
                cache.put(key, value);
                return;
            }
            if (cache.size() < capacity) {
                cache.put(key, value);
            } else {
                // evict LUR
                java.util.Map.Entry<Integer, Integer> entry = cache.entrySet().iterator().next();
                int firstKey = entry.getKey();
                cache.remove(firstKey);
                cache.put(key, value);
            }
        }
    }

    /**
     * Insert last half list in reversed order
     *
     * Given a singly linked list L: L0°˙L1°˙°≠°˙Ln-1°˙Ln, reorder it to:
     * L0°˙Ln°˙L1°˙Ln-1°˙L2°˙Ln-2°˙°≠ You must do this in-place without altering the
     * nodes' values.
     *
     * For example Given {1,2,3,4}, reorder it to {1,4,2,3}.
     *
     */

    public static class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    public ListNode reverseList(ListNode head) {
        ListNode node = head;
        ListNode prev = null;
        while (node != null) {
            ListNode tmp = node.next;
            node.next = prev;
            prev = node;
            node = tmp;
        }
        return prev;
    }

    public ListNode getMiddleNode(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode r1 = head;
        ListNode r2 = head;
        while (r2 != null) {
            r1 = r1.next;
            r2 = r2.next;
            if (r2 == null) {
                return r1;
            } else {
                r2 = r2.next;
            }
        }
        return r1;
    }

    public void reorderList(ListNode head) {
        if (head == null || head.next == null || head.next.next == null) {
            return;
        }
        // assuming list has more than 2 elements
        // three steps:
        // 1. find the middle of the list
        // 2. reverse the middle of the list till last node
        // 3. insert the reversed list
        ListNode middle = getMiddleNode(head);
        ListNode insertHead = reverseList(middle);

        ListNode node = head;
        ListNode insert = insertHead;
        while (node.next != insert && insert != null) {
            ListNode next = node.next;
            node.next = insert;
            ListNode insertNext = insert.next;
            insert.next = next;
            node = next;
            insert = insertNext;
        }
        if (node.next != null && node.next.next == node) {
            // remove last circular link
            node.next = null;
        }
    }

    /**
     * Copy List with Random Pointer
     *
     * A linked list is given such that each node contains an additional random
     * pointer which could point to any node in the list or null. Return a deep
     * copy of the list.
     *
     * Definition for singly-linked list with a random pointer.
     */
    class RandomListNode {
        int label;
        RandomListNode next, random;

        RandomListNode(int x) {
            this.label = x;
        }
    };

    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) {
            return null;
        }
        Queue<RandomListNode> randomNodes = new LinkedList<RandomListNode>();
        Queue<RandomListNode> originalNodes = new LinkedList<RandomListNode>();

        RandomListNode copyHead = new RandomListNode(head.label);
        RandomListNode it = head;
        RandomListNode copy = copyHead;
        while (it != null) {
            RandomListNode next = it.next;
            it.next = copy;
            RandomListNode copyNext = (next == null) ? null : new RandomListNode(next.label);
            copy.next = copyNext;
            copy = copyNext;
            randomNodes.add(it.random);
            it = next;
            originalNodes.add(it);
        }

        // now reset random nodes
        copy = copyHead;
        while (copy != null) {
            RandomListNode r = randomNodes.poll();
            copy.random = (r == null) ? null : r.next;
            copy = copy.next;
        }

        // restore original node list
        it = head;
        while (it != null) {
            RandomListNode n = originalNodes.poll();
            it.next = n;
            it = n;
        }
        return copyHead;
    }

    /**
     * Merge two sorted Arrays
     *
     * Given two sorted integer arrays A and B, merge B into A as one sorted
     * array.
     *
     * Note: You may assume that A has enough space (size that is greater or
     * equal to m + n) to hold additional elements from B. The number of
     * elements initialized in A and B are m and n respectively.
     *
     * @param A
     * @param m
     * @param B
     * @param n
     */
    public void mergeTwoSortedArrays(int[] A, int m, int[] B, int n) {
        if (A == null || B == null) {
            return;
        }
        int len = m + n;
        int i = m - 1; // A's index
        int j = n - 1; // B's index
        int k = len - 1; // new array's index

        while (i >= 0 && j >= 0) {
            if (A[i] > B[j]) {
                A[k--] = A[i--];
            } else {
                A[k--] = B[j--];
            }
        }
        while (j >= 0) {
            A[k--] = B[j--];
        }
    }

    /**
     * Linked List Cycle
     *
     * Given a linked list, determine if it has a cycle in it.
     *
     * Definition for singly-linked list. class ListNode { int val; ListNode
     * next; ListNode(int x) { val = x; next = null; } }
     */
    public boolean hasCycle(ListNode head) {
        if (head == null) {
            return false;
        }
        ListNode first = head;
        ListNode second = head;

        while (first != null && second != null) {
            first = first.next;
            second = second.next;
            if (second == null) {
                return false;
            }
            second = second.next;
            if (second == first) {
                return true;
            }
        }
        return false;
    }

    /**
     * Linked List Cycle II
     *
     * Given a linked list, return the node where the cycle begins. If there is
     * no cycle, return null.
     *
     * Definition for singly-linked list. class ListNode { int val; ListNode
     * next; ListNode(int x) { val = x; next = null; } }
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode first = head;
        ListNode second = head;

        while (first != null && second != null) {
            first = first.next;
            second = second.next;
            if (second == null) {
                return null;
            }
            second = second.next;
            // find the cycle
            if (second == first) {
                // move first to head and restart
                first = head;
                while (first != second) {
                    first = first.next;
                    second = second.next;
                }
                return first;
            }
        }
        return null;
    }

    /**
     * Single Number
     *
     * Given an array of integers, every element appears twice except for one.
     * Find that single one. Your algorithm should have a linear runtime
     * complexity. Could you implement it without using extra memory?
     */
    public int singleNumber(int[] A) {
        // exclusive or
        if (A == null || A.length == 0) {
            throw new IllegalArgumentException();
        }

        for (int i = 1; i < A.length; i++) {
            A[0] ^= A[i];
        }
        return A[0];
    }

    /**
     * Single Number II
     *
     * Given an array of integers, every element appears three times except for
     * one. Find that single one.
     *
     * Your algorithm should have a linear runtime complexity. Could you
     * implement it without using extra memory?
     *
     * Following impl is very trick, it uses concept of Boolean algabra, I don't
     * understand it
     */
    public int singleNumber2(int[] A) {
        int ones = 0, twos = 0;
        for (int element : A) {
            ones = (ones ^ element) & ~twos;
            twos = (twos ^ element) & ~ones;
        }
        return ones;
    }

    /**
     * Minimum Depth of Binary Tree
     *
     * Given a binary tree, find its minimum depth. The minimum depth is the
     * number of nodes along the shortest path from the root node down to the
     * nearest leaf node.
     *
     */
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left != null && root.right == null) {
            return 1 + minDepth(root.left);
        } else if (root.right != null && root.left == null) {
            return 1 + minDepth(root.right);
        } else if (root.left == null && root.right == null) {
            return 1;
        } else {
            return 1 + Math.min(minDepth(root.left), minDepth(root.right));
        }
    }

    /**
     * Maximum Depth of Binary Tree
     *
     * Given a binary tree, find its maximum depth.
     *
     * The maximum depth is the number of nodes along the longest path from the
     * root node down to the farthest leaf node.
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left != null && root.right != null) {
            return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
        } else if (root.left != null) {
            return 1 + maxDepth(root.left);
        } else if (root.right != null) {
            return 1 + maxDepth(root.right);
        } else {
            return 1;
        }
    }

    /**
     * Word Ladder
     *
     * Given two words (start and end), and a dictionary, find the length of
     * shortest transformation sequence from start to end, such that:
     *
     * Only one letter can be changed at a time Each intermediate word must
     * exist in the dictionary
     *
     * For example, Given: start = "hit"end = "cog" dict =
     * ["hot","dot","dog","lot","log"] As one shortest transformation is "hit"
     * -> "hot" -> "dot" -> "dog" -> "cog", return its length 5.
     *
     */
    public int ladderLength(String start, String end, Set<String> dict) {
        Map<String, String> backtrack = new HashMap<String, String>();
        Set<String> visited = new HashSet<String>();
        Queue<String> q = new LinkedList<String>();
        q.offer(start);
        visited.add(start);

        while (!q.isEmpty()) {
            String current = q.poll();
            for (String neighbor : getNeighbors(current, dict)) {
                if (neighbor.equals(end)) {
                    List<String> list = new ArrayList<String>();
                    list.add(neighbor);
                    while (current != null) {
                        list.add(0, current);
                        current = backtrack.get(current);
                    }
                    return list.size();
                }
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    q.offer(neighbor);
                    backtrack.put(neighbor, current);
                }
            }
        }
        return 0;
    }

    private static List<String> getNeighbors(String s, Set<String> dict) {
        List<String> result = new ArrayList<String>();
        for (int i = 0; i < s.length(); i++) {
            for (char it = 'a'; it <= 'z'; it++) {
                char[] word = s.toCharArray();
                if (it == word[i]) {
                    continue;
                }
                word[i] = it;
                String str = new String(word);
                if (dict.contains(str)) {
                    result.add(str);
                }
            }
        }
        return result;
    }

    /**
     * Word Ladder II
     *
     * Find all shortest path
     *
     */
    public List<List<String>> findLadders(String start, String end, Set<String> dict) {
        Map<String, String> backtrack = new HashMap<String, String>();
        Set<String> visited = new HashSet<String>();
        Queue<String> q = new LinkedList<String>();
        List<List<String>> result = new ArrayList<List<String>>();

        q.offer(start);
        visited.add(start);

        while (!q.isEmpty()) {
            String current = q.poll();
            for (String neighbor : getNeighbors(current, dict)) {
                if (neighbor.equals(end)) {
                    List<String> list = new ArrayList<String>();
                    list.add(neighbor);
                    while (current != null) {
                        list.add(0, current);
                        current = backtrack.get(current);
                    }
                    result.add(list);
                    continue;
                }
                if (!visited.contains(neighbor)) {
                    visited.add(neighbor);
                    q.offer(neighbor);
                    backtrack.put(neighbor, current);
                }
            }
        }
        // return only shorted path, so compare list size
        int min = Integer.MAX_VALUE;
        for (List<String> list : result) {
            if (list.size() < min) {
                min = list.size();
            }
        }
        List<List<String>> output = new ArrayList<List<String>>();
        for (List<String> list : result) {
            if (list.size() == min) {
                output.add(list);
            }
        }
        return output;
    }

    /**
     * Palindrome
     *
     * For example, "A man, a plan, a canal: Panama" is a palindrome.
     * "race a car" is not a palindrome.
     *
     * Note: Have you consider that the string might be empty? This is a good
     * question to ask during an interview.
     *
     * For the purpose of this problem, we define empty string as valid
     * palindrome.
     *
     */
    public boolean isPalindrome(String s) {
        if (s == null) {
            throw new IllegalArgumentException();
        }
        if (s.length() == 0) {
            return true;
        }

        int head = 0;
        int tail = s.length() - 1;
        while (head < tail) {
            while (!isAlphanumeric(s.charAt(head))) {
                head++;
                if (head > s.length() - 1) {
                    return true;
                }
                if (head == tail) {
                    break;
                }
            }
            while (!isAlphanumeric(s.charAt(tail))) {
                tail--;
                if (tail < 0) {
                    return true;
                }
                if (head == tail) {
                    break;
                }
            }
            if (head >= tail) {
                break;
            } else if (Character.toLowerCase(s.charAt(head++)) != Character.toLowerCase(s.charAt(tail--))) {
                return false;
            }
        }

        return true;
    }

    private boolean isAlphanumeric(char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
    }

    /**
     * Anagrams
     *
     * Given an array of strings, return all groups of strings that are
     * anagrams.
     *
     */
    public List<String> anagrams(String[] strs) {
        if (strs == null) {
            return null;
        }

        Map<String, ArrayList<String>> signedTable = new HashMap<String, ArrayList<String>>();
        for (String s : strs) {
            char[] arr = s.toCharArray();
            Arrays.sort(arr);
            String signed = new String(arr);
            ArrayList<String> list = signedTable.get(signed);
            if (list == null) {
                list = new ArrayList<String>();
            }
            list.add(s);
            signedTable.put(signed, list);
        }
        List<String> result = new ArrayList<String>();
        for (Entry<String, ArrayList<String>> entry : signedTable.entrySet()) {
            List<String> tmp = entry.getValue();
            if (tmp.size() > 1) {
                result.addAll(tmp);
            }
        }
        return result;
    }

    /**
     * Climbing Stairs
     *
     * You are climbing a stair case. It takes n steps to reach to the top.
     *
     * Each time you can either climb 1 or 2 steps. In how many distinct ways
     * can you climb to the top?
     *
     * Note: this is essentially Fibnacci
     *
     */
    public int climbStairs_Recursive_Fib(int n) {
        if (n <= 2) {
            return n;
        }
        return climbStairs_Recursive_Fib(n - 1) + climbStairs_Recursive_Fib(n - 2);
    }

    // iterative solution - DP
    public int climbStairs(int n) {
        if (n <= 2) {
            return n;
        }
        int[] fib = new int[n];
        fib[0] = 1;
        fib[1] = 2;
        for (int i = 2; i < n; i++) {
            fib[i] = fib[i - 1] + fib[i - 2];
        }
        return fib[n - 1];
    }

    /**
     * Minimum Window Substring
     *
     * Given a string S and a string T, find the minimum window in S which will
     * contain all the characters in T in complexity O(n).
     *
     * For example, S = "ADOBECODEBANC" T = "ABC" Minimum window is "BANC".
     *
     * Note: If there is no such window in S that covers all characters in T,
     * return the emtpy string "".
     *
     * If there are multiple such windows, you are guaranteed that there will
     * always be only one unique minimum window in S.
     *
     * @param S
     *            input string
     * @param T
     *            pattern
     * @return
     */
    // approach 1: windows
    public String minWindow(String S, String T) {
        if (T == null || T.length() == 0 || S == null || S.length() == 0) {
            return "";
        }
        if (S.length() < T.length()) {
            return "";
        }

        int totalMatchCount = T.length();
        int matchCountSofar = 0;
        Map<Character, Integer> shouldFind = new HashMap<Character, Integer>();
        // compute for each char, how many match should we do
        for (int i = 0; i < T.length(); i++) {
            if (shouldFind.containsKey(T.charAt(i))) {
                int num = shouldFind.get(T.charAt(i));
                shouldFind.put(T.charAt(i), ++num);
            } else {
                shouldFind.put(T.charAt(i), 1);
            }
        }
        int beginIndex = 0; // minimal substring index
        int endIndex = Integer.MAX_VALUE;

        // constraint, move end pointer first, increase count by 1 if move over
        // hit char;
        // if find so far is less than or greater than shouldFind, increase
        // matchCountSofar
        // if matchCountSofar == totalMatchCount, then a substring found, keep
        // this constraint
        // and move begin pointer
        int i = 0, j = 0; // i: index for begin pointer; j: index for end
        // pointer
        Map<Character, Integer> findSofar = new HashMap<Character, Integer>();

        for (; j < S.length(); j++) {
            // move end pointer
            while (j < S.length()) {
                char ch = S.charAt(j);
                if (shouldFind.containsKey(ch)) {
                    int n = findSofar.containsKey(ch) ? findSofar.get(ch) + 1 : 1;
                    findSofar.put(ch, n);
                    // for duplicate chars, only increase found count when the
                    // so far found number is not exceeding desired count,
                    // otherwise it doesn't contribute to total count
                    // eg shoul found ="abb", you have "bb", and encounter
                    // another b so it is "bbb" now, but it doesn't help
                    if (n <= shouldFind.get(ch)) {
                        matchCountSofar++;
                    }
                    // check if found all the chars to match
                    if (matchCountSofar == totalMatchCount) {
                        // update index interval
                        if (j - i < endIndex - beginIndex) {
                            endIndex = j;
                            beginIndex = i;
                        }
                        break;
                    }
                }
                j++;
            }
            // move begin pointer
            while (i <= j) {
                // no any match found while j reached end of string
                if (matchCountSofar != totalMatchCount) {
                    return "";
                }
                // How do we check if it is maintaining the constraint?
                // Assume that begin points to an element x, we check if
                // hasFound[x]
                // is greater than needToFind[x]. If it is, we can decrement
                // hasFound[x] by one
                // and advancing begin pointer without breaking the constraint.
                // On the other hand, if it is not, we stop immediately as
                // advancing begin pointer breaks the window constraint.
                char ch = S.charAt(i);
                if (!shouldFind.containsKey(ch)) {
                    // update interval and advance tail pointer
                    i++;
                    if (j - i < endIndex - beginIndex) {
                        endIndex = j;
                        beginIndex = i;
                    }
                    continue;
                }
                int count = findSofar.get(ch);
                // we have exact match for current char, we cannot advance
                // current begin pointer
                if (count <= shouldFind.get(ch)) {
                    break;
                }
                // advance being pointer
                findSofar.put(ch, --count);
                i++;
                if (j - i < endIndex - beginIndex) {
                    endIndex = j;
                    beginIndex = i;
                }
            }
        }

        return S.substring(beginIndex, endIndex + 1);

    }

    // approach 2: Kway merge, it works but is sub-optimal
    // Complexity: O(MlgN), unfortunately Time Limit Exceeded (TLE)
    /**
     * min window to cover a char set string
     *
     */
    private static class KeyCharPair {
        int key;
        char value;

        public KeyCharPair(int i, char ch) {
            key = i;
            value = ch;
        }
    }

    private static class Interval {
        int start;
        int end;

        public Interval(int s, int e) {
            start = s;
            end = e;
        }
    }

    public String minWindow2(String S, String T) {
        if (T == null || T.length() == 0 || S == null || S.length() == 0) {
            return "";
        }
        if (S.length() < T.length()) {
            return "";
        }

        Set<Character> charSet = new HashSet<Character>();
        Map<Character, ArrayList<Integer>> map = new HashMap<Character, ArrayList<Integer>>();
        // build map for each char set
        for (int i = 0; i < T.length(); i++) {
            charSet.add(T.charAt(i));
        }
        for (int i = 0; i < S.length(); i++) {
            if (!charSet.contains(S.charAt(i))) {
                continue;
            }
            ArrayList<Integer> list = map.get(S.charAt(i));
            if (list == null) {
                list = new ArrayList<Integer>();
            }
            list.add(i);
            map.put(S.charAt(i), list);
        }
        // check if map has all char set
        if (map.size() != charSet.size()) {
            return "";
        }

        // get minimal interval out of map which covers charset
        List<Interval> intervals = new ArrayList<Interval>();
        int minInterval = Integer.MAX_VALUE;
        PriorityQueue<KeyCharPair> minQ = new PriorityQueue<KeyCharPair>(map.size(), new Comparator<KeyCharPair>() {
            @Override
            public int compare(KeyCharPair v1, KeyCharPair v2) {
                return v1.key - v2.key;
            }
        });
        PriorityQueue<KeyCharPair> maxQ = new PriorityQueue<KeyCharPair>(map.size(), new Comparator<KeyCharPair>() {
            @Override
            public int compare(KeyCharPair v1, KeyCharPair v2) {
                return v2.key - v1.key;
            }
        });
        // input head of each string into Heap
        for (Entry<Character, ArrayList<Integer>> entry : map.entrySet()) {
            KeyCharPair p = new KeyCharPair(entry.getValue().get(0), entry.getKey());
            minQ.add(p);
            maxQ.add(p);
        }
        while (!minQ.isEmpty()) {
            KeyCharPair v1 = minQ.remove();
            KeyCharPair v2 = maxQ.peek();
            // update minimal interval
            if (v2.key - v1.key < minInterval) {
                intervals.clear();
                intervals.add(new Interval(v1.key, v2.key));
                minInterval = v2.key - v1.key;
            } else if (v2.key - v1.key == minInterval) {
                intervals.add(new Interval(v1.key, v2.key));
            }
            // get next element from arrays
            ArrayList<Integer> list = map.get(v1.value);
            list.remove(0);
            if (list.isEmpty()) { // one of the arrays is done, we can break
                break;
            }
            KeyCharPair p = new KeyCharPair(list.get(0), v1.value);
            minQ.add(p);
            // remove smallest element from maxQ and add new element too
            maxQ.remove(v1); // O(lgM)
            maxQ.add(p); // O(lgM)
        }
        if (intervals.size() == 0) {
            return "";
        }
        String[] result = new String[intervals.size()];
        int i = 0;
        for (Interval it : intervals) {
            result[i++] = S.substring(it.start, it.end + 1);
        }
        return result[0];
    }

    /**
     * Regular Expression Match
     *
     * Implement regular expression matching with support for '.' and '*'.
     *
     * The matching should cover the entire input string (not partial).
     *
     * The function prototype should be: bool isMatch(const char *s, const char
     * *p)
     *
     * Some examples: isMatch("aa","a") °˙ false isMatch("aa","aa") °˙ true
     * isMatch("aaa","aa") °˙ false isMatch("aa", "a*") °˙ true isMatch("aa",
     * ".*") °˙ true isMatch("ab", ".*") °˙ true isMatch("aab", "c*a*b") °˙ true
     *
     */
    //@formatter:off
    /*
     * code in c++:
     *
     * bool matchHead(const char *s, const char *p) {
     *      return *s == *p ||   (*p=='.' && *s != '\0');
     * }
     *
     * bool isMatch(const char *s, const char *p) {
     *      if (*p=='\0') return *s=='\0'; // match whole string
     *
     *      if (*(p+1) != '*') {
     *          if (!matchHead(s, p)) return false;
     *          return  isMatch(++s, ++p);
     *      } else { // p+1 is *
     *          if (isMatch(s, p+2)) { // 0 match for *
     *              return true;
     *          }
     *          else {
     *              while(matchHead(s, p)) { // one or more match for *
     *                  if(isMatch(++s, p+2)) return true;
     *              }
     *          }
     *      }
     *      return false;
     * }
     */
    //@formatter:off

    /**
     * Wild match '?' Matches any single character. '*' Matches any sequence of
     * characters (including the empty sequence).
     *
     * The matching should cover the entire input string (not partial).
     *
     * The function prototype should be: bool isMatch(const char *s, const char
     * *p)
     *
     * Some examples: isMatch("aa","a") °˙ false isMatch("aa","aa") °˙ true
     * isMatch("aaa","aa") °˙ false isMatch("aa", "*") °˙ true isMatch("aa", "a*")
     * °˙ true isMatch("ab", "?*") °˙ true isMatch("aab", "c*a*b") °˙ false
     *
     * c++ code: it has time exception, we need dp
     *
     *
     * bool matchHead(const char*s, const char*p) { return (*s == *p) ||
     * (*p=='?' && *s != '\0'); }
     *
     * bool isMatch(const char *s, const char *p) { if (*p=='\0') return
     * *s=='\0'; // match complete string
     *
     * if (*p != '*') { if (!matchHead(s, p)) return false; return isMatch(s+1,
     * p+1); } else { // handle * matching // if * is last char, then matched
     * successfully // otherwise skip * and look for whatever after * // skip
     * multiple * while (*(p+1) == '*') p++; if (*(p+1)=='\0') return true; // *
     * can match string // now *(p+1) is a non * char, do the match if
     * (*(p+1)=='?') return *s != '\0'; // ? needs at least one char to match
     * and * handles rest // brute force search rest of s and match *(p+1) with
     * each char while (*s != '\0') { if (*s==*(p+1)) { if (isMatch(s+1, p+2))
     * return true; } s++; } return false; // nothing matches } }
     */

    // below is Yucoding's c++ algo, good job
    // @formatter:off
    //    bool isMatch(const char *s, const char *p) {
    //        const char* star=NULL;
    //        const char* ss=s;
    //        while (*s){
    //            //advancing both pointers when (both characters match) or ('?' found in pattern)
    //            //note that *p will not advance beyond its length
    //            if ((*p=='?')||(*p==*s)){s++;p++;continue;}
    //
    //            // * found in pattern, track index of *, only advancing pattern pointer
    //            if (*p=='*'){star=p++; ss=s;continue;}
    //
    //            //current characters didn't match, last pattern pointer was *, current pattern pointer is not *
    //            //only advancing pattern pointer
    //            if (star){ p = star+1; s=++ss;continue;}
    //
    //            //current pattern pointer is not star, last patter pointer was not *
    //            //characters do not match
    //            return false;
    //        }
    //
    //        //check for remaining characters in pattern
    //        while (*p=='*'){p++;}
    //
    //        return !*p;
    //    }
    // @formatter:on

    /**
     * Balanced Binary Tree
     *
     * This problem is generally believed to have two solutions: the top down
     * approach and the bottom up way.
     *
     * 1.The first method checks whether the tree is balanced strictly according
     * to the definition of balanced binary tree: the difference between the
     * heights of the two sub trees are not bigger than 1, and both the left sub
     * tree and right sub tree are also balanced. With the helper function
     * depth(), we could easily write the code;
     *
     * 2.The second method is based on DFS. Instead of calling depth()
     * explicitly for each child node, we return the height of the current node
     * in DFS recursion. When the sub tree of the current node (inclusive) is
     * balanced, the function dfsHeight() returns a non-negative value as the
     * height. Otherwise -1 is returned. According to the leftHeight and
     * rightHeight of the two children, the parent node could check if the sub
     * tree is balanced, and decides its return value.
     *
     */
    // way 1 - Based on top down O(n^2) like
    public int depth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left != null && root.right != null) {
            return 1 + Math.max(depth(root.left), depth(root.right));
        } else if (root.right != null) {
            return 1 + depth(root.right);
        } else if (root.left != null) {
            return 1 + depth(root.left);
        } else {
            return 1;
        }
    }

    public boolean isBalanced(TreeNode root) {
        if (root == null) {
            return true;
        }

        int depthLeft = depth(root.left);
        int depthRight = depth(root.right);
        // check every node and sub node for heighth
        if (Math.abs(depthLeft - depthRight) > 1) {
            return false;
        } else {
            return isBalanced(root.left) && isBalanced(root.right);
        }
    }

    // way 2 - O(n) return values: -1 means false, otherwise it represents
    // height
    private int dfsHeight(TreeNode root) {
        if (root == null) {
            return 0; // true and height = 0
        }
        int lh = dfsHeight(root.left);
        if (lh == -1) {
            return -1;
        }
        int rh = dfsHeight(root.right);
        if (rh == -1) {
            return -1;
        }
        if (Math.abs(rh - lh) > 1) {
            return -1;
        }
        return Math.max(lh, rh) + 1;
    }

    public boolean isBalanced_bottom_up(TreeNode root) {
        if (root == null) {
            return true;
        }

        return dfsHeight(root) != -1;
    }

    /**
     * Plus One
     *
     */
    public int[] plusOne(int[] digits) {
        int len = digits.length;
        boolean carry = false;
        List<Integer> list = new ArrayList<Integer>();
        int first = digits[len - 1] + 1;
        if (first >= 10) {
            list.add(0, first - 10);
            carry = true;
        } else {
            list.add(0, first);
        }
        for (int i = len - 2; i >= 0; i--) {
            int result = carry ? ++digits[i] : digits[i];
            if (result >= 10) {
                list.add(0, result - 10);
                carry = true;
            } else {
                list.add(0, result);
                carry = false;
            }
        }
        if (carry) {
            list.add(0, 1);
        }

        int[] ret = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            ret[i] = list.get(i);
        }
        return ret;
    }

    /**
     * Level order traversal in bottom up sequence
     */
    public List<List<Integer>> levelOrderBottom(TreeNode root) {
        if (root == null) {
            return new ArrayList<List<Integer>>();
        }

        Stack<ArrayList<Integer>> stack = new Stack<ArrayList<Integer>>();
        ArrayList<Integer> list = new ArrayList<Integer>();
        Queue<TreeNode> q = new LinkedList<TreeNode>();

        int current = 1;
        int next = 0;
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode n = q.poll();
            current--;
            list.add(n.val);
            if (n.left != null) {
                q.offer(n.left);
                next++;
            }
            if (n.right != null) {
                q.offer(n.right);
                next++;
            }
            // check end of level
            if (current == 0) {
                current = next;
                next = 0;
                stack.push((ArrayList) list.clone());
                list.clear();
            }
        }
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        while (!stack.isEmpty()) {
            ArrayList k = stack.pop();
            ret.add(k);
        }
        return ret;
    }

    /**
     * Merge k Sorted Lists
     *
     * Merge k sorted linked lists and return it as one sorted list. Analyze and
     * describe its complexity.
     *
     */
    public ListNode mergeKLists(List<ListNode> lists) {
        if (lists == null || lists.size() == 0) {
            return null;
        }

        PriorityQueue<ListNode> q = new PriorityQueue<ListNode>(lists.size(), new Comparator<ListNode>() {
            @Override
            public int compare(ListNode n1, ListNode n2) {
                return n1.val - n2.val;
            }
        });

        for (ListNode n : lists) {
            if (n != null) {
                q.offer(n);
            }
        }
        if (q.isEmpty()) {
            return null;
        }

        ListNode root = q.remove(); // the smallest node is root
        if (root.next != null) {
            q.add(root.next);
        }
        ListNode it = root;
        while (!q.isEmpty()) {
            ListNode node = q.remove();
            it.next = node;
            if (node.next != null) {
                q.add(node.next);
            }
            it = it.next;
        }
        return root;
    }

    /**
     * Sqrt(x)
     *
     * compute the square root of x
     */
    // @formatter:on
    public int sqrt(int x) {
        if (x < 0) {
            throw new IllegalArgumentException();
        }

        if (x < 2) {
            return x;
        }

        // simulate binary search
        int low = 0;
        int high = x;

        while (low <= high) {
            int mid = (low + high) >>> 1;
            // careful, number might overflow, so use division instead of
            // multiplication
            if (mid == x / mid) {
                return mid;
            } else if (mid < x / mid && x / (mid + 1) < (mid + 1)) {
                return mid;
            } else if (mid < x / mid) {
                low = mid + 1;
            } else if (mid > x / mid) {
                high = mid - 1;
            }
        }

        return -1; // error
    }

    /**
     * is valid number
     */
    public boolean isNumber(String s) {
        String str = s.trim(); // remove leading and trailing empty
        // valid no digit chars:
        // +, cannot be single +
        // -, cannot be single 1
        // .,
        // e,
        // space in middle if not valid
        boolean hasNeg = false;
        boolean hasPos = false;
        boolean hasE = false;
        boolean hasDot = false;

        if (str.length() == 0) {
            return false;
        }

        for (int i = 0; i < str.length(); i++) {
            if (i == 0) {
                if (str.charAt(i) == '+') {
                    hasPos = true;
                    continue;
                } else if (str.charAt(i) == '-') {
                    hasNeg = true;
                    continue;
                }
            }
            char ch = str.charAt(i);
            if (!isValidChar(ch)) {
                return false;
            }
            if (i - 1 >= 0 && (str.charAt(i - 1) == 'e' || str.charAt(i - 1) == '.')) {
                if (!isNumberDigit(ch)) {
                    return false;
                }
            }
            if (ch == 'e') {
                if (i - 1 < 0 || (!isNumberDigit(str.charAt(i - 1)) && str.charAt(i - 1) != '.')) {
                    return false;
                } else if (i == str.length() - 1) {
                    return false;
                }
                hasE = true;
            }
            if (ch == '.') {
                if (hasDot) {
                    return false;
                }
                if (i == str.length() - 1 && i - 1 >= 0 && !isNumberDigit(str.charAt(i - 1))) {
                    return false;
                }
                hasDot = true;
            }
        }

        if (hasPos || hasNeg || hasDot || hasE) {
            return str.length() != 1;
        }
        return true;
    }

    public boolean isValidChar(char ch) {
        // + and - is only allowed in leading
        return isNumberDigit(ch) || ch == '.' || ch == 'e';
    }

    public boolean isNumberDigit(char ch) {
        return (ch >= '0' && ch <= '9');
    }

    /**
     * Interleaving String
     *
     * Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and
     * s2.
     *
     * For example, Given: s1 = "aabcc", s2 = "dbbca",
     *
     * When s3 = "aadbbcbcac", return true. When s3 = "aadbbbaccc", return
     * false.
     */
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s3.length() != s1.length() + s2.length()) {
            return false;
        }

        boolean[][] table = new boolean[s1.length() + 1][s2.length() + 1];

        for (int i = 0; i < s1.length() + 1; i++) {
            for (int j = 0; j < s2.length() + 1; j++) {
                if (i == 0 && j == 0) {
                    table[i][j] = true;
                } else if (i == 0) {
                    table[i][j] = (table[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                } else if (j == 0) {
                    table[i][j] = (table[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1));
                } else {
                    table[i][j] = (table[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1))
                            || (table[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
                }
            }
        }

        return table[s1.length()][s2.length()];
    }

    // recursive cause Time exception
    public boolean isInterleave_recursive(String s1, String s2, String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }

        if (s1.length() + s2.length() != s3.length()) {
            return false;
        }

        if (s1.length() == 0 && s2.length() == 0) {
            return s3.length() == 0;
        } else if (s1.length() == 0) {
            return s2.equals(s3);
        } else if (s2.length() == 0) {
            return s1.equals(s3);
        }

        // String subs1 = "";
        // String subs2 = "";

        if (s1.charAt(0) == s3.charAt(0) && s2.charAt(0) != s3.charAt(0)) {
            String subs1 = (1 == s1.length()) ? "" : s1.substring(1);
            String subs3 = (1 == s3.length()) ? "" : s3.substring(1);
            return isInterleave(subs1, s2, subs3);
        } else if (s2.charAt(0) == s3.charAt(0) && s1.charAt(0) != s3.charAt(0)) {
            String subs2 = (1 == s2.length()) ? "" : s2.substring(1);
            String subs3 = (1 == s3.length()) ? "" : s3.substring(1);
            return isInterleave(s1, subs2, subs3);
        } else if (s1.charAt(0) == s3.charAt(0) && s2.charAt(0) == s3.charAt(0)) {
            String subs1 = (1 == s1.length()) ? "" : s1.substring(1);
            String subs2 = (1 == s2.length()) ? "" : s2.substring(1);
            String subs3 = (1 == s3.length()) ? "" : s3.substring(1);
            return isInterleave(s1, subs2, subs3) || isInterleave(subs1, s2, subs3);
        } else { // no match at head
            return false;
        }
    }

    /**
     * Is scrambled string
     *
     * Below is one possible representation of s1 = "great":
     *
     * Define scrambled string: cut a string at any position then build binary
     * tree recursively, switch children of non-leaf nodes to get another
     * string, this string is called scrambled string
     *
     * Given two strings s1 and s2 of the same length, determine if s2 is a
     * scrambled string of s1.
     */
    public boolean isScramble(String s1, String s2) {
        if (s1 == null || s2 == null || s1.length() != s2.length()) {
            return false;
        }
        if (s1.equals(s2)) {
            return true;
        }
        char[] c1 = s1.toCharArray();
        char[] c2 = s2.toCharArray();
        Arrays.sort(c1);
        Arrays.sort(c2);
        if (!Arrays.equals(c1, c2)) {
            return false;
        }
        for (int i = 1; i < s1.length(); i++) {
            if (isScramble(s1.substring(0, i), s2.substring(0, i)) && isScramble(s1.substring(i), s2.substring(i))) {
                return true;
            }
            if (isScramble(s1.substring(0, i), s2.substring(s2.length() - i))
                    && isScramble(s1.substring(i), s2.substring(0, s2.length() - i))) {
                return true;
            }
        }
        return false;
    }

    /**
     * Is anagram given two strings
     *
     */
    public boolean isAnagram(String s1, String s2) {
        if (s1 == null || s2 == null) {
            return false;
        }
        if (s1.length() != s2.length()) {
            return false;
        }

        Map<Character, Integer> set = new HashMap<Character, Integer>();
        for (int i = 0; i < s1.length(); i++) {
            if (set.containsKey(s1.charAt(i))) {
                int count = set.get(s1.charAt(i));
                set.put(s1.charAt(i), ++count);
            } else {
                set.put(s1.charAt(i), 1);
            }
        }
        for (int j = 0; j < s2.length(); j++) {
            if (!set.containsKey(s2.charAt(j))) {
                return false;
            }
            int count = set.get(s2.charAt(j));
            if (count == 1) {
                set.remove(s2.charAt(j));
            } else {
                set.put(s2.charAt(j), --count);
            }
        }
        return true;
    }

    /**
     * Multiply Strings
     *
     * Given two numbers represented as strings, return multiplication of the
     * numbers as a string.
     *
     * Note: The numbers can be arbitrarily large and are non-negative.
     */
    public String multiply(String num1, String num2) {
        if (num1 == null || num2 == null) {
            return "0";
        }

        String mul = "";

        // use short string for the loop
        int len = Math.min(num1.length(), num2.length());
        boolean useNum1ForLoop = num1.length() > num2.length();

        for (int i = len - 1; i >= 0; i--) {
            if (useNum1ForLoop) {
                String tmp = multiplySingle(num2.charAt(i), num1);
                // handle overflow, must add by string
                int power = num2.length() - 1 - i;
                StringBuilder sb = new StringBuilder();
                sb.append(tmp);
                while (power-- > 0) {
                    sb.append("0");
                }
                mul = addString(mul, sb.toString());
            } else {
                String tmp = multiplySingle(num1.charAt(i), num2);
                // handle overflow, must add by string
                int power = num1.length() - 1 - i;
                StringBuilder sb = new StringBuilder();
                sb.append(tmp);
                while (power-- > 0) {
                    sb.append("0");
                }
                mul = addString(mul, sb.toString());
            }
        }

        return mul;
    }

    // multiple string and a single digit
    public String multiplySingle(char ch, String num) {
        // must use incremental string multiply, or may overflow
        int n = ch - '0';
        String sum = "";

        int len = num.length();
        for (int i = len - 1; i >= 0; i--) {
            int tmp = (num.charAt(i) - '0') * n;
            // sum + tmp * 10^ power
            int power = len - 1 - i;
            // Integer.MAX_VALUE = 2147483647
            StringBuilder sb = new StringBuilder();
            sb.append(tmp);
            while (power-- > 0) {
                sb.append("0");
            }
            sum = addString(sum, sb.toString());
        }
        return sum;
    }

    // add two strings and return the result in string format - this is used to
    // handle overflow
    public String addString(String sum, String base) {
        StringBuilder sb = new StringBuilder();
        if (base.length() == 0) {
            return sum;
        }

        if (sum.length() == 0) {
            return base;
        }

        int i = sum.length() - 1;
        int j = base.length() - 1;
        int carry = 0;
        while (i >= 0 && j >= 0) {
            int a = (sum.charAt(i) - '0') + (base.charAt(j) - '0') + carry;
            int digit = a > 9 ? (a - 10) : a;
            carry = a > 9 ? 1 : 0;
            sb.insert(0, digit);
            i--;
            j--;
        }
        // if any left
        while (i >= 0) {
            int a = (sum.charAt(i) - '0') + carry;
            int digit = a > 9 ? (a - 10) : a;
            carry = a > 9 ? 1 : 0;
            sb.insert(0, digit);
            i--;
        }
        while (j >= 0) {

            int a = (base.charAt(j) - '0') + carry;
            int digit = a > 9 ? (a - 10) : a;
            carry = a > 9 ? 1 : 0;
            sb.insert(0, digit);
            j--;
        }
        if (carry == 1) {
            sb.insert(0, 1);
        }
        return sb.toString();
    }

    /**
     * First Missing Positive
     *
     * Given an unsorted integer array, find the first missing positive integer.
     *
     * For example, Given [1,2,0] return 3, and [3,4,-1,1] return 2.
     *
     * Your algorithm should run in O(n) time and uses constant space.
     */

    public int firstMissingPositive(int[] A) {
        if (A == null || A.length == 0) {
            return 1;
        }

        int len = A.length;
        for (int i = 0; i < len; i++) {
            // found is in the array so we only need to move found to A[found-1]
            int found = A[i];
            while (found <= len && found > 0 && A[found - 1] != found) {
                int other = A[found - 1];
                A[found - 1] = found;
                found = other; // continue check if other is in A[other-1]
            }
        }
        // pass 2: find first A[i] != i
        for (int i = 0; i < len; i++) {
            if (A[i] != i + 1) {
                return i + 1;
            }
        }
        return len + 1;
    }

    /**
     * Longest Substring Without Repeating Characters
     *
     * Given a string, find the length of the longest substring without
     * repeating characters. For example, the longest substring without
     * repeating letters for "abcabcbb" is "abc", which the length is 3. For
     * "bbbbb" the longest substring is "b", with the length of 1.
     */
    public int lengthOfLongestSubstring(String s) {
        // use tow pointers, head and tail, and a linkedhashmap to track
        // visited char, move tail to end of string, if a duplicate is found,
        // move head to the duplicate index+1, and delete linkedhashmap index
        // times
        // so we have a new non duplicate substring
        // keep this process until end of the string
        if (s == null || s.length() == 0) {
            return 0;
        }

        int head = 0;
        int tail = 1;
        int max = 1;
        // track char and its index
        Map<Character, Integer> map = new LinkedHashMap<Character, Integer>();
        map.put(s.charAt(head), 0);
        for (; tail < s.length(); tail++) {
            char ch = s.charAt(tail);
            if (!map.containsKey(ch)) {
                map.put(ch, tail);
                max = Math.max(max, tail - head + 1);
                continue;
            }
            // handle duplicate - duplicate first happend at index dupIndex
            int dupIndex = map.get(ch);
            // compute how many step we should move head to dupIndex+1, and
            // clean map before dupIndex+1
            int dist = dupIndex - head + 1;
            head = dupIndex + 1;
            // move head
            for (int j = 0; j < dist; j++) {
                Character key = map.keySet().iterator().next();
                map.remove(key);
            }
            // if duplicate is at end of previous string, then we need to add
            // head to the map, eg, "bcaa", two "a"'s are removed
            // if duplicate is at beginning of previous string, then we need to
            // add tail to the map, eg, "abca", two "a"'s are removed
            // so check both of them
            if (!map.containsKey(s.charAt(head))) {
                map.put(s.charAt(head), head);
            }
            if (!map.containsKey(s.charAt(tail))) {
                map.put(s.charAt(tail), tail);
            }
        }

        return max;
    }

    // in this impl we don't delete entry form hashmap to save time
    public int lengthOfLongestSubstring_perf_enhance_HashMap_No_delete(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int head = 0;
        int tail = 1;
        int max = 1;
        // track char and its index using hashmap instead of LinkedHashMap
        Map<Character, Integer> map = new HashMap<Character, Integer>();
        map.put(s.charAt(head), 0);
        for (; tail < s.length(); tail++) {
            char ch = s.charAt(tail);
            // if map.get(ch) < head, it says the current char was in previous
            // discarded chars due to duplicate
            if (!map.containsKey(ch) || map.get(ch) < head) {
                map.put(ch, tail);
                max = Math.max(max, tail - head + 1);
                continue;
            }
            // handle duplicate - duplicate first happend at index dupIndex
            int dupIndex = map.get(ch);
            head = dupIndex + 1;
            // no need to physically delete key from map as it is slow, just
            // ignore if the index is before head

            // if duplicate is at end of previous string, then we need to add
            // head to the map
            // if duplicate is at beginning of previous string, then we need to
            // add tail to the map
            // reset head and tail
            map.put(s.charAt(head), head);
            map.put(s.charAt(tail), tail);
        }

        return max;
    }

    public int lengthOfLongestSubstring_perf_enhance_Use_array(String s) {
        // use tow pointers, head and tail, and a linkedhashmap to track
        // visited char, move tail to end of string, if a duplicate is found,
        // move head to the duplicate index+1, and delete linkedhashmap index
        // times
        // so we have a new non duplicate substring
        // keep this process until end of the string
        if (s == null || s.length() == 0) {
            return 0;
        }

        int head = 0;
        int tail = 1;
        int max = 1;
        // track char and its index
        // use char[] instead of map to improve performance
        int[] table = new int[256];
        Arrays.fill(table, -1);
        table[s.charAt(head)] = 0;

        for (; tail < s.length(); tail++) {
            char ch = s.charAt(tail);
            if (table[ch] == -1 || table[ch] < head) {
                table[ch] = tail;
                max = Math.max(max, tail - head + 1);
                continue;
            }
            // handle duplcate - duplicate first happend at index dupIndex
            int dupIndex = table[ch];
            // int dist = dupIndex - head + 1; // compute how many step we
            // should

            // move head to dupIndex+1, and clean map before dupIndex+1
            head = dupIndex + 1;

            table[s.charAt(head)] = head;
            table[s.charAt(tail)] = tail;
        }

        return max;
    }

    /**
     * AtoI
     *
     * Convert string to int
     *
     * Requirements for atoi: The function first discards as many whitespace
     * characters as necessary until the first non-whitespace character is
     * found. Then, starting from this character, takes an optional initial plus
     * or minus sign followed by as many numerical digits as possible, and
     * interprets them as a numerical value.
     *
     * The string can contain additional characters after those that form the
     * integral number, which are ignored and have no effect on the behavior of
     * this function.
     *
     * If the first sequence of non-whitespace characters in str is not a valid
     * integral number, or if no such sequence exists because either str is
     * empty or it contains only whitespace characters, no conversion is
     * performed.
     *
     * If no valid conversion could be performed, a zero value is returned. If
     * the correct value is out of the range of representable values, INT_MAX
     * (2147483647) or INT_MIN (-2147483648) is returned.
     */
    public int atoi(String str) {
        if (str == null || str.length() == 0) {
            return 0;
        }

        str = str.trim(); // trim leading and tailing white space
        int base = 0;
        int sign = 0; // pos:0, neg:1

        int i = 0;
        if (str.charAt(i) == '+') {
            sign = 0;
            i++;
        } else if (str.charAt(i) == '-') {
            sign = 1;
            i++;
        }

        for (; i < str.length(); i++) {
            char ch = str.charAt(i);
            if (ch < '0' || ch > '9') {
                return sign == 0 ? base : -base; // return whatever value before
                // this illegal char
            }
            // the trick is check 7 for fine grain increment before overflow
            if (base > Integer.MAX_VALUE / 10 || (base == Integer.MAX_VALUE / 10 && ch - '0' > 7)) {
                return sign == 0 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
            } else {
                base = base * 10 + ch - '0';
            }
        }

        return sign == 0 ? base : -base;
    }

    /**
     * Substring with Concatenation of All Words You are given a string, S, and
     * a list of words, L, that are all of the same length. Find all starting
     * indices of substring(s) in S that is a concatenation of each word in L
     * exactly once and without any intervening characters.
     *
     * For example, given: S: "barfoothefoobarman" L: ["foo", "bar"]
     *
     * You should return the indices: [0,9]. (order does not matter).
     *
     */

    // Naive impl with Time exception
    public List<Integer> findSubstring(String S, String[] L) {
        // use two hashtable, one records L, the other records visited so far
        if (S == null || L == null || S.length() == 0 || L.length == 0) {
            return new ArrayList<Integer>();
        }

        // note that L might has duplicte strings
        Map<String, Integer> visited = new HashMap<String, Integer>();
        Map<String, Integer> dict = new HashMap<String, Integer>();
        List<Integer> list = new ArrayList<Integer>();
        int start = -1;
        int count = 0;

        for (String element : L) {
            if (dict.containsKey(element)) {
                int val = dict.get(element);
                dict.put(element, ++val);
            } else {
                dict.put(element, 1);
            }
        }

        int inc = L[0].length(); // pattern string's common len
        int i = 0;
        while (i < S.length()) {
            if (i + inc > S.length()) {
                break; // out of range
            }
            String s = S.substring(i, i + inc);
            if (!dict.containsKey(s)) {
                // no match or already found, move i to i+1
                visited.clear();
                start = -1; // reset head
                i++;
                continue;
            }
            // found a match
            if (!visited.containsKey(s)) {
                visited.put(s, 1);
                count++;
            } else {
                int found = visited.get(s);
                int toMatch = dict.get(s);
                if (found == toMatch) {
                    // already has enough match
                    visited.clear();
                    start = -1; // reset head
                    i++;
                    continue;
                } else {
                    visited.put(s, ++found);
                    count++;
                }
            }

            if (count == 1) {
                // first pattern match
                start = i;
            }
            if (count == L.length) {
                // all paterns are matched
                list.add(start);
                visited.clear();
                count = 0;
                // rewind i to the next of match
                i = start + 1;
            } else {
                i += inc;
            }
        }

        return list;
    }

    /**
     * Longest Palindromic Substring
     *
     * Given a string S, find the longest palindromic substring in S. You may
     * assume that the maximum length of S is 1000, and there exists one unique
     * longest palindromic substring.
     *
     */
    // Manacher's algorithm - O(N)
    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0) {
            return s;
        }

        String ss = preProcessForPalindrome(s);
        int len = ss.length();
        int[] p = new int[len];
        int id = 0;
        int mx = 0; // mx = id + p[id]
        for (int i = 1; i < len; i++) {
            p[i] = (mx > i) ? Math.min(p[2 * id - i], mx - i) : 1;
            while (i + p[i] < len && i - p[i] >= 0) {
                if (ss.charAt(i + p[i]) == ss.charAt(i - p[i])) {
                    p[i]++;
                } else {
                    break;
                }
            }
            if (i + p[i] > mx) {
                mx = i + p[i];
                id = i;
            }
        }
        // find max p[i], then
        int max = 0;
        for (int i = 1; i < len; i++) {
            if (p[i] > p[max]) {
                max = i;
            }
        }
        return s.substring((max - p[max] + 1) / 2, (max + p[max]) / 2);
    }

    // convert "abc" to "#a#b#c#"
    private String preProcessForPalindrome(String s) {
        StringBuilder sb = new StringBuilder();
        sb.append("#");
        for (int i = 0; i < s.length(); i++) {
            sb.append(s.charAt(i));
            sb.append("#");
        }
        return sb.toString();
    }

    // below is naive O(N^3)
    public String longestPalindrome_Time_Excpetion(String s) {
        if (s.length() == 1) {
            return s;
        }

        int len = s.length();
        for (int n = len; n > 1; n--) {
            for (int i = 0; i <= len - n; i++) {
                String sub = s.substring(i, i + n);
                if (isPalin(sub)) {
                    return sub;
                }
            }
        }
        // no palin for size 2 and above, so just return lenght 1
        return s.substring(0, 1);
    }

    public boolean isPalin(String s) {
        int len = s.length();
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(i) != s.charAt(len - i - 1)) {
                return false;
            }
        }
        return true;
    }

    /**
     * Merge Two Sorted Lists
     *
     * Merge two sorted linked lists and return it as a new list. The new list
     * should be made by splicing together the nodes of the first two lists.
     *
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        } else if (l2 == null) {
            return l1;
        }

        ListNode head = (l2.val < l1.val) ? l2 : l1;
        ListNode curr = new ListNode(0);
        while (l1 != null && l2 != null) {
            if (l2.val < l1.val) {
                curr.next = l2;
                curr = l2;
                l2 = l2.next;
            } else {
                curr.next = l1;
                curr = l1;
                l1 = l1.next;
            }
        }
        if (l1 == null) {
            curr.next = l2;
        } else if (l2 == null) {
            curr.next = l1;
        }

        return head;
    }

    /**
     * Pascal's Triangle
     *
     * Given numRows, generate the first numRows of Pascal's triangle.
     *
     * For example, given numRows = 5, Return
     *
     * [ [1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1] ]
     *
     */
    public List<List<Integer>> generate(int numRows) {
        if (numRows == 0) {
            return new ArrayList<List<Integer>>();
        }

        List<List<Integer>> result = new LinkedList<List<Integer>>();

        List<Integer> last = new LinkedList<Integer>();
        for (int i = 0; i < numRows; i++) {
            // only add new object next to list array
            List<Integer> next = new LinkedList<Integer>();
            if (i == 0) {
                next.add(1);
                result.add(next);
                last.clear();
                last.addAll(next);
                continue;
            }
            int prev = 0;
            int count = 0;
            // to avoid java.util.ConcurrentModificationException exception
            // it happens when we iterate through a list while also modify it.
            for (int n : last) {
                next.add(prev + n);
                prev = n;
                count++;
                if (count == last.size()) {
                    next.add(n);
                }
            }
            result.add(next);
            // list = next;
            last.clear();
            last.addAll(next);
        }

        return result;
    }

    /**
     * Pascal's Triangle II
     *
     * Given an index k, return the kth row of the Pascal's triangle.
     *
     * For example, given k = 3, Return [1,3,3,1].
     *
     * Note: Could you optimize your algorithm to use only O(k) extra space?
     */
    public List<Integer> getRow(int rowIndex) {

        List<Integer> last = new LinkedList<Integer>();
        for (int i = 0; i <= rowIndex; i++) {
            // only add new object next to list array
            List<Integer> next = new LinkedList<Integer>();
            if (i == 0) {
                next.add(1);
                last.clear();
                last.addAll(next);
                continue;
            }
            int prev = 0;
            int count = 0;
            // to avoid java.util.ConcurrentModificationException exception
            // it happens when we iterate through a list while also modify it.
            for (int n : last) {
                next.add(prev + n);
                prev = n;
                count++;
                if (count == last.size()) {
                    next.add(n);
                }
            }
            last.clear();
            last.addAll(next);
        }

        return last;
    }

    /**
     * Level order traversal
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return null;
        }

        List<List<Integer>> result = new ArrayList<List<Integer>>();
        List<TreeNode> list = new ArrayList<TreeNode>();
        int current = 1;
        int next = 0;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);

        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            current--;
            if (node.left != null) {
                q.offer(node.left);
                next++;
            }
            if (node.right != null) {
                q.offer(node.right);
                next++;
            }
            list.add(node);
            if (current == 0) {
                current = next;
                next = 0;
                List<Integer> copy = new ArrayList<Integer>();
                for (TreeNode n : list) {
                    copy.add(n.val);
                }
                result.add(copy);
                list.clear();
            }
        }
        return result;
    }

    /**
     * Symmetric binary tree
     *
     * Given a binary tree, check whether it is a mirror of itself (ie,
     * symmetric around its center).
     *
     * For example, this binary tree is symmetric:
     *
     * 1 / \ 2 2 / \ / \ 3 4 4 3 But the following is not: 1 / \ 2 2 \ \ 3 3
     *
     * idea: 1. recursive: check left and right child 2. iterative: use two
     * stacks, while one push in left child, the other push in right child, then
     * pop and compare
     */
    public boolean isSymmetricBinaryTree(TreeNode root) {
        if (root == null) {
            return true;
        }

        if (root.left != null && root.right != null) {
            return checkSym(root.left, root.right);
        } else if (root.left != null && root.right == null) {
            return false;
        } else if (root.left == null && root.right != null) {
            return false;
        } else { // root is leaf
            return true;
        }
    }

    private boolean checkSym(TreeNode left, TreeNode mirror) {
        if (left.val != mirror.val) {
            return false;
        }

        if (left.left != null && left.right != null) {
            if (mirror.left == null || mirror.right == null) {
                return false;
            }
            return checkSym(left.left, mirror.right) && checkSym(left.right, mirror.left);
        } else if (left.left != null) {
            if (mirror.right == null || mirror.left != null) {
                return false;
            }
            return checkSym(left.left, mirror.right);
        } else if (left.right != null) {
            if (mirror.right != null || mirror.left == null) {
                return false;
            }
            return checkSym(left.right, mirror.left);
        } else { // leaf
            if (mirror.left != null || mirror.right != null) {
                return false;
            } else {
                return true;
            }
        }
    }

    /**
     * Same tree
     *
     * Given two binary trees, write a function to check if they are equal or
     * not.
     *
     * Two binary trees are considered equal if they are structurally identical
     * and the nodes have the same value.
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null) {
            return q == null;
        }
        if (q == null) {
            return p == null;
        }

        if (p.val != q.val) {
            return false;
        }
        if (p.left != null && p.right != null) {
            if (q.left == null || q.right == null) {
                return false;
            }
            return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
        } else if (p.left != null) {
            if (q.left == null || q.right != null) {
                return false;
            }
            return isSameTree(p.left, q.left);
        } else if (p.right != null) {
            if (q.right == null || q.left != null) {
                return false;
            }
            return isSameTree(p.right, q.right);
        } else { // leaf
            return true;
        }
    }

    /**
     * Remove duplicate from sorted list
     *
     */
    public ListNode deleteDuplicatesFromSortedList(ListNode head) {
        if (head == null) {
            return null;
        }
        if (head.next == null) {
            return head;
        }

        // find new head if head has duplicate.
        ListNode node = head;

        while (node.next != null) {
            if (node.val == node.next.val) {
                node = node.next;
            } else {
                break;
            }
        }
        ListNode newHead = node;
        ListNode front = node.next;
        while (front != null) {
            if (front.next == null) {
                break;
            }
            if (front.val == front.next.val) {
                node.next = front.next;
                front = front.next;
            } else {
                node = front;
                front = front.next;
            }
        }

        return newHead;
    }

    /**
     * Remove Duplicates from Sorted List II
     *
     * Given a sorted linked list, delete all nodes that have duplicate numbers,
     * leaving only distinct numbers from the original list.
     *
     * For example, Given 1->2->3->3->4->4->5, return 1->2->5. Given
     * 1->1->1->2->3, return 2->3.
     */
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return head;
        }
        ListNode dummy = new ListNode(0); // dummy
        dummy.next = head;
        ListNode n = dummy;
        while (n != null) {
            ListNode next = n.next;
            boolean dup = false;
            while (next != null) {
                if (next.next == null) {
                    break;
                } else if (next.val != next.next.val) {
                    if (!dup) {
                        break;
                    } else {
                        dup = false; // reset and start new round of dup check
                    }
                } else {
                    dup = true;
                }
                next = next.next;
            }
            n.next = dup ? next.next : next;
            n = n.next;
        }
        return dummy.next;
    }

    /**
     * Add Binary
     *
     * Given two binary strings, return their sum (also a binary string). For
     * example, a = "11" b = "1" Return "100".
     */
    public String addBinary(String a, String b) {
        // validate a, b
        if (a == null || a.length() == 0) {
            return b;
        }
        if (b == null || b.length() == 0) {
            return a;
        }

        int i = a.length() - 1;
        int j = b.length() - 1;
        int carry = 0;
        List<Integer> list = new ArrayList<Integer>();

        while (i >= 0 && j >= 0) {
            int n = a.charAt(i) - '0' + b.charAt(j) - '0' + carry;
            carry = n / 2;
            int digit = n % 2;
            list.add(0, digit);
            i--;
            j--;
        }
        while (i >= 0) {
            int n = a.charAt(i) - '0' + carry;
            carry = n / 2;
            int digit = n % 2;
            list.add(0, digit);
            i--;
        }
        while (j >= 0) {
            int n = b.charAt(j) - '0' + carry;
            carry = n / 2;
            int digit = n % 2;
            list.add(0, digit);
            j--;
        }
        if (carry > 0) {
            list.add(0, 1);
        }
        StringBuilder sb = new StringBuilder();
        for (Integer m : list) {
            sb.append(m);
        }
        return sb.toString();
    }

    /**
     * Length of Last Word
     *
     * Given a string s consists of upper/lower-case alphabets and empty space
     * characters ' ', return the length of last word in the string.
     *
     * If the last word does not exist, return 0.
     *
     * Note: A word is defined as a character sequence consists of non-space
     * characters only.
     *
     * For example, Given s = "Hello World", return 5.
     */
    // idea: count from last words
    public int lengthOfLastWord(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int count = 0;
        int i = s.length() - 1;

        while (i >= 0) {
            if (s.charAt(i) != ' ') {
                count++;
            } else {
                if (count != 0) {
                    return count;
                }
                i--;
                continue;
            }
            i--;
        }
        return count;
    }

    /**
     * Count and Say
     *
     * The count-and-say sequence is the sequence of integers beginning as
     * follows: 1, 11, 21, 1211, 111221, ...
     *
     * 1 is read off as "one 1" or 11. 11 is read off as "two 1s" or 21. 21 is
     * read off as "one 2, then one 1" or 1211. Given an integer n, generate the
     * nth sequence.
     *
     * Note: The sequence of integers will be represented as a string.
     *
     */
    public String countAndSay(int n) {
        String s = "1";
        for (int i = 0; i < n - 1; i++) {
            s = getNextString(s);
        }
        return s;
    }

    private String getNextString(String s) {
        StringBuilder sb = new StringBuilder();
        int prev = s.charAt(0) - '0';
        int count = 1;
        for (int i = 1; i < s.length(); i++) {
            int d = s.charAt(i) - '0';
            if (d == prev) {
                count++;
            } else {
                sb.append(count);
                sb.append(prev);
                prev = d;
                count = 1;
            }
        }
        // add last element to sb
        sb.append(count);
        sb.append(prev);
        return sb.toString();
    }

    /**
     * Count and Say II
     *
     * The count-and-say II sequence is the sequence of integers beginning as
     * follows: 1, 11, 21, 1211, 1231, 131221, ... ie, count number of max valu,
     * till smallest
     *
     * 1 is read off as "one 1" or 11. 11 is read off as "two 1s" or 21. 21 is
     * read off as "one 2, then one 1" or 1211. Given an integer n, generate the
     * nth sequence.
     *
     * Note: The sequence of integers will be represented as a string.
     *
     */
    public String countAndSay2(int n) {
        String s = "1";
        for (int k = 0; k < n - 1; k++) {
            StringBuilder sb = new StringBuilder();
            Map<Integer, Integer> lookup = new TreeMap<Integer, Integer>(Collections.reverseOrder());
            for (int i = 0; i < s.length(); i++) {
                int d = s.charAt(i) - '0';
                lookup.put(d, lookup.containsKey(d) ? lookup.get(d) + 1 : 1);
            }
            for (Entry<Integer, Integer> e : lookup.entrySet()) {
                s = sb.append(e.getValue()).append(e.getKey()).toString();
            }
        }
        return s;
    }

    /**
     * check if this is valid Sudoku
     *
     * 1. row 2. col 3. 9 blocks
     */
    public boolean isValidSudoku(char[][] board) {
        Map<Character, Integer> col = new HashMap<Character, Integer>();
        Map<Character, Integer> row = new HashMap<Character, Integer>();
        Map<Character, Integer> blk1 = new HashMap<Character, Integer>();
        Map<Character, Integer> blk2 = new HashMap<Character, Integer>();
        Map<Character, Integer> blk3 = new HashMap<Character, Integer>();

        if (board.length != 9) {
            return false;
        }
        // check row
        for (int i = 0; i < board.length; i++) {
            if (board[i].length != 9) {
                return false;
            }
            for (int j = 0; j < board[i].length; j++) {
                int count = row.containsKey(board[i][j]) ? row.get(board[i][j]) + 1 : 1;
                row.put(board[i][j], count);
                count = col.containsKey(board[j][i]) ? col.get(board[j][i]) + 1 : 1;
                col.put(board[j][i], count);

                // fit int to 3 blk
                int k = j / 3;
                switch (k) {
                case 0:
                    count = blk1.containsKey(board[i][j]) ? blk1.get(board[i][j]) + 1 : 1;
                    blk1.put(board[i][j], count);
                    break;
                case 1:
                    count = blk2.containsKey(board[i][j]) ? blk2.get(board[i][j]) + 1 : 1;
                    blk2.put(board[i][j], count);
                    break;
                case 2:
                    count = blk3.containsKey(board[i][j]) ? blk3.get(board[i][j]) + 1 : 1;
                    blk3.put(board[i][j], count);
                    break;
                default:
                    break;
                }
            }
            if (!checkSudukoRule(row) || !checkSudukoRule(col)) {
                return false;
            }
            row.clear();
            col.clear();
            // check 3-block
            if ((i + 1) % 3 == 0) {
                if (!checkSudukoRule(blk1) || !checkSudukoRule(blk2) || !checkSudukoRule(blk3)) {
                    return false;
                }
                blk1.clear();
                blk2.clear();
                blk3.clear();
            }
        }
        return true;
    }

    private boolean checkSudukoRule(Map<Character, Integer> map) {
        // '.' count plus '1-9' must equal 9
        int dotCount = 0;
        int nonDotCount = 0;
        for (Entry<Character, Integer> entry : map.entrySet()) {
            if (entry.getKey() == '.') {
                dotCount += entry.getValue();
            } else {
                int d = entry.getValue();
                if (d > 1) {
                    return false;
                }
                nonDotCount += entry.getValue();
            }
        }
        return (dotCount + nonDotCount) == 9;
    }

    /**
     * remove an element from an Array using inplace
     *
     * @return
     */
    public int removeElement(int[] A, int elem) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int index = 0;
        // sort works better - but do inplace sort
        Arrays.sort(A);
        for (int i = 0; i < A.length; i++) {
            if (A[i] == elem) {
                continue;
            } else {
                if (index < i) {
                    A[index++] = A[i];
                } else {
                    index++;
                }
            }
        }
        return index;
    }

    /**
     * Remove Duplicates from Sorted Array
     *
     * same as above
     */
    public int removeDuplicates(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int index = 1;
        // sort works better - but do inplace sort
        int prev = A[0];
        for (int i = 1; i < A.length; i++) {
            if (A[i] == prev) {
                continue;
            } else {
                A[index++] = A[i];
                prev = A[i];
            }
        }
        // check end duplicate
        return index;
    }

    /**
     * Remove Duplicates from Sorted Array II
     *
     * Follow up for "Remove Duplicates": What if duplicates are allowed at most
     * twice?
     *
     * For example, Given sorted array A = [1,1,1,2,2,3],
     *
     * Your function should return length = 5, and A is now [1,1,2,2,3].
     */
    public int removeDuplicates2(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int index = 1;
        int count = 1;

        for (int i = 1; i < A.length; i++) {
            if (A[i] != A[i - 1]) {
                count = 1;
                A[index++] = A[i];
                continue;
            }
            // handle duplicate
            if (count < 2) {
                count++;
                A[index++] = A[i];
            }
        }
        return index;
    }

    /**
     * reverse a number's digit
     *
     * Reverse digits of an integer.
     *
     * Example1: x = 123, return 321 Example2: x = -123, return -321
     *
     * click to show spoilers.
     *
     * Have you thought about this? Here are some good questions to ask before
     * coding. Bonus points for you if you have already thought through this!
     *
     * If the integer's last digit is 0, what should the output be? ie, cases
     * such as 10, 100.
     *
     * Did you notice that the reversed integer might overflow? Assume the input
     * is a 32-bit integer, then the reverse of 1000000003 overflows. How should
     * you handle such cases?
     *
     * Throw an exception? Good, but what if throwing an exception is not an
     * option? You would then have to re-design the function (ie, add an extra
     * parameter).
     */
    public int reverse(int x) {
        boolean neg = false;
        if (x < 0) {
            neg = true;
            x *= -1;
        }
        // need to handle overflow
        int n = 0;
        while (x > 0) {
            int d = x % 10;
            n = n * 10 + d;
            x /= 10;
        }

        return neg ? -n : n;
    }

    /**
     * Find media of two sorted array
     *
     * There are two sorted arrays A and B of size m and n respectively. Find
     * the median of the two sorted arrays. The overall run time complexity
     * should be O(log (m+n)).
     *
     * A smart iteratife algo:
     *
     * This is my iterative solution using binary search. The main idea is to
     * find the approximate location of the median and compare the elements
     * around it to get the final result.
     *
     * do binary search. suppose the shorter list is A with length n. the
     * runtime is O(log(n)) which means no matter how large B array is, it only
     * depends on the size of A. It makes sense because if A has only one
     * element while B has 100 elements, the median must be one of A[0], B[49],
     * and B[50] without check everything else. If A[0] <= B[49], B[49] is the
     * answer; if B[49] < A[0] <= B[50], A[0] is the answer; else, B[50] is the
     * answer.
     *
     * After binary search, we get the approximate location of median. Now we
     * just need to compare at most 4 elements to find the answer. This step is
     * O(1).
     *
     * the same solution can be applied to find kth element of 2 sorted arrays.
     */
    public double findMedianSortedArrays(int A[], int B[]) {
        int n = A.length;
        int m = B.length;
        // the following call is to make sure len(A) <= len(B).
        // yes, it calls itself, but at most once, shouldn't be
        // consider a recursive solution
        if (n > m) {
            return findMedianSortedArrays(B, A);
        }

        // now, do binary search
        int k = (n + m - 1) / 2;
        int l = 0; // left
        int r = Math.min(k, n); // right - r is n, NOT n-1, this is important!!
        while (l < r) {
            int midA = (l + r) / 2;
            int midB = k - midA;
            // if (midB > k || A[midA] < B[midB]) { // original code, but I
            // think midB > k is unnecessary
            if (A[midA] < B[midB]) {
                l = midA + 1;
            } else {
                r = midA;
            }
        }

        // after binary search, we almost get the median because it must be
        // between
        // these 4 numbers: A[l-1], A[l], B[k-l], and B[k-l+1]

        // if (n+m) is odd, the median is the larger one between A[l-1] and
        // B[k-l].
        // and there are some corner cases we need to take care of.
        int a = Math.max(l > 0 ? A[l - 1] : Integer.MIN_VALUE, k - l >= 0 ? B[k - l] : Integer.MIN_VALUE);
        if (((n + m) & 1) == 1) {
            return a;
        }

        // if (n+m) is even, the median can be calculated by
        // median = (max(A[l-1], B[k-l]) + min(A[l], B[k-l+1]) / 2.0
        // also, there are some corner cases to take care of.
        int b = Math.min(l < n ? A[l] : Integer.MAX_VALUE, k - l + 1 < m ? B[k - l + 1] : Integer.MAX_VALUE);
        return (a + b) / 2.0;
    }

    /**
     * Best Time to Buy and Sell Stock - this is equivalent to running sum
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i.
     *
     * If you were only permitted to complete at most one transaction (ie, buy
     * one and sell one share of the stock), design an algorithm to find the
     * maximum profit.
     */
    public int maxProfit(int[] prices) {
        // find and keep updating lowest
        if (prices == null || prices.length == 0) {
            return 0;
        }

        int low = prices[0];
        int max = 0;

        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < low) {
                low = prices[i];
                continue;
            }
            if (prices[i] - low > max) {
                max = prices[i] - low;
            }
        }
        return max;
    }

    /**
     * Best Time to Buy and Sell Stock II - this is equivalent to sum up all
     * increasing trend
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i.
     *
     * Design an algorithm to find the maximum profit. You may complete as many
     * transactions as you like (ie, buy one and sell one share of the stock
     * multiple times). However, you may not engage in multiple transactions at
     * the same time (ie, you must sell the stock before you buy again).
     */
    public int maxProfit2(int[] prices) {
        // sum up all increasing graph
        if (prices == null || prices.length == 0) {
            return 0;
        }

        int prev = prices[0];
        int sum = 0;

        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prev) {
                sum += prices[i] - prev;
            }
            prev = prices[i];
        }
        return sum;
    }

    /**
     * Best Time to Buy and Sell Stock III
     *
     * Say you have an array for which the ith element is the price of a given
     * stock on day i.
     *
     * Design an algorithm to find the maximum profit. You may complete at most
     * two transactions.
     *
     * Note: You may not engage in multiple transactions at the same time (ie,
     * you must sell the stock before you buy again).
     */
    //@formatter:off
    /* this is someone's dp proposal for k transactions, great work!
     *
        int maxProfit(vector<int> &prices) {
        // f[k, i] represents the max profit up until prices[i] (Note: NOT ending with prices[i]) using at most k transactions.
        // f[k, i] = max(f[k, i-1], prices[i] - prices[j] + f[k-1, j]) { j in range of [0, i-1] }
        //          = max(f[k, i-1], prices[i] + max(f[k-1, j] - prices[j]))
        // f[0, i] = 0; 0 times transation makes 0 profit
        // f[k, 0] = 0; if there is only one price data point you can't make any money no matter how many times you can trade
        if (prices.size() <= 1) return 0;
        else {
            int totoalTransCount = 2; // number of max transation allowed
            int maxProf = 0;
            vector<vector<int>> f(totoalTransCount + 1, vector<int>(prices.size(), 0));
            for (int k = 1; k <= totoalTransCount; k++) {
                int tmpMax = f[k-1][0] - prices[0];
                for (int i = 1; i < prices.size(); i++) {
                    f[k][i] = max(f[k][i-1], prices[i] + tmpMax);
                    tmpMax = max(tmpMax, f[k-1][i] - prices[i]);
                    maxProf = max(f[k][i], maxProf);
                }
            }
            return maxProf;
        }
     */
    //@formatter:on

    // this impl is wrong, it computes the maximal two bands, unforutunatley
    // this is not right for this problem
    public int maxProfit3(int[] prices) {
        if (prices == null || prices.length == 0) {
            return 0;
        }

        // count every increasing graph and calculate largest two
        int prev = prices[0];
        int sum = 0; // local sum piece by piece
        int max = 0;
        int secondMax = 0;
        int upBandCount = 0;

        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prev) { // increasing
                sum += prices[i] - prev;
                if (upBandCount == 0) { // first band, update max only
                    max = Math.max(sum, max);
                } else {
                    secondMax = Math.max(Math.min(sum, max), secondMax);
                    max = Math.max(sum, max);
                }
                prev = prices[i];
            } else { // decreasing
                upBandCount++;
                // reset sum
                sum = 0;
                prev = prices[i];
            }
        }
        return max + secondMax;
    }

    /**
     * Binary Tree Inorder Traversal
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        if (root == null) {
            return new ArrayList<Integer>();
        }
        List<Integer> list = new ArrayList<Integer>();
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode node = root;

        while (!stack.isEmpty() || node != null) {
            if (node != null) {
                stack.push(node);
                node = node.left;
            } else { // pop
                node = stack.pop();
                list.add(node.val);
                node = node.right;
            }
        }
        return list;

    }

    /**
     * strstr - find index of a substring in string
     *
     * Returns the index of the first occurrence of needle in haystack, or -1 if
     * needle is not part of haystack.
     *
     * KMP- the core part is to build the KMP prefix table, the failure table,
     * which dictates how many steps you can backtrak when failure occurs
     */
    public int[] build_kmp_prefix_table(String p) {
        if (p == null || p.length() == 0) {
            return null;
        }

        int len = p.length();
        // the kmp prefix table
        int[] t = new int[len];
        t[0] = -1; // first element is -1 as initial failure has to rewind
        // working table needs two more elements as t[1] is always 0 - it has no
        // *proper* suffix
        if (len == 1) {
            return t;
        }
        t[1] = 0;

        // populate the t[i] table,
        // 1. when we fail at i, find a suffix stops at i-1, which is a also
        // a prefix of string p;
        // 2. t[i] can be deduced from t[i-1] by check one more elment befor
        // t[i-1], because t[i-1] is the maxmial suffix match (also a prefix
        // of p) stops at i-2
        //
        // taking example: p="abcdabd"
        // t[0]=-1, t[1]=0, so t(2) try p(1) and its
        // previous t(1) values (0), end up with 0, it 'c' is not a prefix of p,
        // and so on

        // this is actually a dp
        for (int i = 2; i < len; i++) {
            // trackback of t[i-1] plus new char at p[i-1]
            if (p.charAt(t[i - 1]) == p.charAt(i - 1)) {
                t[i] = t[i - 1] + 1;
            } else if (p.charAt(i - 1) == p.charAt(0)) { // only p[i-1] match
                // head
                t[i] = 1;
            } else { // no match between proper suffix and prefix
                t[i] = 0;
            }
        }
        return t;
    }

    public int strStr_kmp(String haystack, String needle) {
        if (haystack == null || needle == null) {
            return -1;
        }
        if (needle.length() == 0) {
            return 0;
        }

        int[] t = build_kmp_prefix_table(needle);
        int i = 0; // index of haystack
        int p = 0; // index of pattern - needle

        while (i + p < haystack.length()) {
            if (haystack.charAt(i + p) == needle.charAt(p)) { // matched head
                if (p == needle.length() - 1) {
                    return i;
                } else {
                    p++;
                }
            } else { // no match found skip
                if (p > 0) {
                    i = i + p - t[p]; // skip unmatch part
                    p = t[p]; // move pattern based on fail table
                } else { // miss head, move to +1
                    p = 0;
                    i++;
                }
            }
        }
        return -1;
    }

    /**
     * Gas station
     *
     * There are N gas stations along a circular route, where the amount of gas
     * at station i is gas[i].
     *
     * You have a car with an unlimited gas tank and it costs cost[i] of gas to
     * travel from station i to its next station (i+1). You begin the journey
     * with an empty tank at one of the gas stations.
     *
     * Return the starting gas station's index if you can travel around the
     * circuit once, otherwise return -1.
     *
     * Note: The solution is guaranteed to be unique.
     */
    public int canCompleteCircuit(int[] gas, int[] cost) {
        // compute pure left at each station, form a new
        // circular array, the sum up of new array has to be
        // positive to get a rout, otherwise return -1
        // then compute running sum of the array, return the starting index
        // of the new array

        if (gas == null || cost == null) {
            return -1;
        }
        if (cost.length == 0) {
            return 0;
        }

        int N = gas.length;
        int[] balance = new int[N];
        int sum = 0;
        int start = 0;
        boolean setStart = false;
        for (int i = 0; i < N; i++) {
            balance[i] = gas[i] - cost[i];
            sum += balance[i];
            if (!setStart) {
                start = i;
                setStart = true;
            }
            // start index is set, check futher - if the sum here is negative
            // and
            // balance is negative, we cannot start from here
            if (sum < 0 && balance[i] < 0) { // cannot reach here
                setStart = false;
            }
        }
        if (sum < 0) {
            return -1;
        } else {
            return start;
        }
    }

    /*
     * Populating Next Right Pointers in Each Node or Chain sibling tree node
     * 
     * - also known as Level order sibling chain
     */
    public static class TreeLinkNode {
        int val;
        TreeLinkNode left, right, next;

        TreeLinkNode(int x) {
            val = x;
        }
    }

    public void connect(TreeLinkNode root) {
        if (root == null) {
            return;
        }
        int currentLevelCount = 1;
        int nextLevelCount = 0;
        Queue<TreeLinkNode> q = new LinkedList<TreeLinkNode>();
        TreeLinkNode prev = null;
        q.offer(root);

        while (!q.isEmpty()) {
            TreeLinkNode node = q.poll();
            currentLevelCount--;
            if (node.left != null) {
                q.offer(node.left);
                nextLevelCount++;
            }
            if (node.right != null) {
                q.offer(node.right);
                nextLevelCount++;
            }
            if (currentLevelCount == 0) { // end of node
                currentLevelCount = nextLevelCount;
                nextLevelCount = 0;
                if (prev != null) {
                    prev.next = node;
                    prev = null;
                }
                node.next = null;
            } else { // in the middle of the level
                if (prev != null) {
                    prev.next = node;
                }
                prev = node;
            }
        }
    }

    /**
     * Valid Parentheses Given a string containing just the characters '(', ')',
     * '{', '}', '[' and ']', determine if the input string is valid.
     *
     * The brackets must close in the correct order, "()" and "()[]{}" are all
     * valid but "(]" and "([)]" are not.
     *
     */
    public boolean isValidParentheses(String s) {
        if (s == null || s.length() == 0) {
            return false;
        }

        Stack<Character> stack = new Stack<Character>();
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == '(' || ch == '[' || ch == '{') {
                stack.push(ch);
            } else if (ch == ')') {
                if (stack.size() == 0 || stack.pop() != '(') {
                    return false;
                }
            } else if (ch == ']') {
                if (stack.size() == 0 || stack.pop() != '[') {
                    return false;
                }
            } else if (ch == '}') {
                if (stack.size() == 0 || stack.pop() != '{') {
                    return false;
                }
            }
        }
        return stack.size() == 0;
    }

    /**
     * simplify path of directory
     *
     * Given an absolute path for a file (Unix-style), simplify it.
     *
     * For example, path = "/home/", => "/home" path = "/a/./b/../../c/", =>
     * "/c"
     */
    public String simplifyPath(String path) {
        if (path == null || path.length() == 0) {
            return path;
        }

        String[] strs = path.trim().split("/");
        Stack<String> stack = new Stack<String>();

        for (String str : strs) {
            if (str.equals("") || str.equals(".")) {
                continue;
            } else if (str.equals("..")) {
                if (stack.size() > 0) {
                    stack.pop();
                }
            } else {
                stack.push(str);
            }
        }
        StringBuilder sb = new StringBuilder();
        if (stack.isEmpty()) {
            return "/";
        }

        while (!stack.isEmpty()) {
            sb.insert(0, stack.pop()).insert(0, "/");
        }
        return sb.toString();
    }

    /**
     * Trapping Rain Water
     *
     * Given n non-negative integers representing an elevation map where the
     * width of each bar is 1, compute how much water it is able to trap after
     * raining.
     *
     * For example, Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
     */
    // it has at least three solutions, stack, below and a left/right expanding
    // dp
    public int trap(int[] A) {
        if (A == null || A.length < 3) {
            return 0;
        }

        int left = 0;
        int right = A.length - 1;
        int secHeight = 0;
        int max = 0;
        while (left < right) {
            if (A[left] < A[right]) {
                secHeight = Math.max(secHeight, A[left]);
                max += secHeight - A[left];
                left++;
            } else { // handle right side
                secHeight = Math.max(secHeight, A[right]);
                max += secHeight - A[right];
                right--;
            }
        }
        return max;
    }

    // stack solution
    public int trap_stack(int[] A) {
        if (A == null || A.length < 3) {
            return 0;
        }

        int sum = 0;
        // use a stack to store descreasing number
        Stack<Integer> stack = new Stack<Integer>();
        for (int i = 0; i < A.length; i++) {
            // is A[i] is smaller than top of stack then push, otheriwse
            // pop until find the lager on in stack

            // pop and calculate rain until larger stack elem found
            while (!stack.isEmpty() && A[stack.peek()] < A[i]) {
                int k = stack.pop();
                if (stack.isEmpty()) {
                    break;
                }
                int height = A[stack.peek()] < A[i] ? A[stack.peek()] - A[k] : A[i] - A[k];
                sum += height * (i - stack.peek() - 1);
            }
            stack.push(i);
        }
        // the decreasing trend in stack won't hold water
        return sum;
    }

    /**
     * Min Stack
     *
     * Design a stack that supports push, pop, top, and retrieving the minimum
     * element in constant time.
     *
     * push(x) -- Push element x onto stack. pop() -- Removes the element on top
     * of the stack. top() -- Get the top element. getMin() -- Retrieve the
     * minimum element in the stack.
     */
    static class MinStack {
        Stack<Integer> stack = new Stack<Integer>();
        Stack<Integer> minStack = new Stack<Integer>();

        public void push(int x) {
            if (minStack.isEmpty() || x <= minStack.peek()) {
                minStack.push(x);
            }

            stack.push(x);
        }

        public void pop() {
            int a = stack.peek();
            int b = minStack.peek();
            // below if not working, strange
            // if (stack.peek() == minStack.peek()) {
            if (a == b) {
                minStack.pop();
            }
            stack.pop();
        }

        public int top() {
            return stack.peek();
        }

        public int getMin() {
            return minStack.peek();
        }
    }

    /**
     * Text Justification
     *
     * Given an array of words and a length L, format the text such that each
     * line has exactly L characters and is fully (left and right) justified.
     * You should pack your words in a greedy approach; that is, pack as many
     * words as you can in each line. Pad extra spaces ' ' when necessary so
     * that each line has exactly L characters. Extra spaces between words
     * should be distributed as evenly as possible. If the number of spaces on a
     * line do not divide evenly between words, the empty slots on the left will
     * be assigned more spaces than the slots on the right. For the last line of
     * text, it should be left justified and no extra space is inserted between
     * words.
     *
     * For example, words: ["This", "is", "an", "example", "of", "text",
     * "justification."] L: 16. Return the formatted lines as: [
     * "This    is    an", "example  of text", "justification.  " ]
     *
     * Note: Each word is guaranteed not to exceed L in length.
     */

    public List<String> fullJustify(String[] words, int L) {

        if (words == null || words.length == 0) {
            // return new ArrayList<String>();
            throw new IllegalArgumentException("words is null or mepty");
        }

        List<String> list = new ArrayList<String>();
        char[] line = new char[L];
        int index = 0;

        int i = 0;
        while (i < words.length) {
            // middle words needs extra ' '
            int requiredLen = (index == 0) ? words[i].length() : words[i].length() + 1;
            if (requiredLen <= line.length - index) {
                if (index == 0) {
                    copyStringToCharArray(words[i], line, index, true);
                } else {
                    copyStringToCharArray(words[i], line, index, false);
                }
                index += requiredLen; // new index starts
                i++;
                if (i == words.length) { // stop when last word hits
                    String s = redistributeStr(line, index, true);
                    list.add(s);
                    break;
                }
            } else { // not enough space for word[i]
                if (index == 0) {
                    // error: word[i] cannot fit into line
                    throw new IllegalArgumentException("size is too short for word: " + words[i]);
                }
                // in the middle
                // 1. stretch last words to end of line
                // 2. move word[i] to next line
                String s = redistributeStr(line, index, false);
                list.add(s);
                index = 0; // only reset index, i points to same word
            }
        }
        return list;
    }

    // index points to next available spot
    public String redistributeStr(char[] arr, int index, boolean isLastLine) {
        // string head and tail should be at 0 and arr.length
        // middle spaces are evenly distributed
        if (index == arr.length) {// just fit, don't do anything
            // arr[index] = ' ';
            return new String(arr);
        } else {
            // compute word blocks and avaialbe spaces
            int intervals = 0;
            for (int i = 0; i < index; i++) {
                if (arr[i] == ' ') {
                    intervals++;
                }
            }
            if (intervals == 0 || isLastLine) {
                // one word in a line, how do we divide?
                // pad ' ' to the string
                for (int i = index; i < arr.length; i++) {
                    arr[i] = ' ';
                }
                return new String(arr);
            }

            int totalAvailableSpace = intervals + arr.length - index;
            int spaceSize = totalAvailableSpace / intervals;
            int padCount = totalAvailableSpace % intervals;
            // now move words to end with proper space
            int end = arr.length - 1;
            int start = index - 1;
            for (int n = 0; n < intervals; n++) {
                while (arr[start] != ' ') {
                    arr[end--] = arr[start--];
                }
                start--;
                // copy ' ' to end by spaceSize times
                for (int j = 0; j < spaceSize; j++) {
                    arr[end--] = ' ';
                }
                // pad extra space to the begining of array
                if (n >= intervals - padCount) {
                    arr[end--] = ' ';
                }
            }
            return new String(arr);
        }
    }

    // copy string to char array starting from index
    public void copyStringToCharArray(String s, char[] arr, int index, boolean isBegining) {
        if (!isBegining) {
            arr[index++] = ' ';
        }

        for (int i = 0; i < s.length(); i++) {
            arr[index++] = s.charAt(i);
        }
    }

    /**
     * Longest Common Prefix
     *
     * Write a function to find the longest common prefix string amongst an
     * array of strings.
     */
    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }

        StringBuilder result = new StringBuilder();
        int index = 0;
        while (true) {
            // loop through strings
            if (strs[0].length() == 0) {
                return "";
            }
            if (index >= strs[0].length()) {
                return result.toString();
            }

            char ch = strs[0].charAt(index);
            boolean stop = false;
            if (strs.length == 1) {
                return strs[0];
            }

            for (int i = 1; i < strs.length; i++) {
                if (index >= strs[i].length()) {
                    return result.toString();
                }
                if (ch != strs[i].charAt(index)) {
                    stop = true;
                    break;
                }
            }
            if (!stop) {
                result.append(ch);
            } else {
                return result.toString();
            }
            index++;
        }
    }

    /**
     * Restore IP Addresses
     *
     * Given a string containing only digits, restore it by returning all
     * possible valid IP address combinations.
     *
     * For example: Given "25525511135",
     *
     * return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
     *
     */
    public List<String> restoreIpAddresses(String s) {
        if (s == null || s.length() == 0 || s.length() > 12) {
            return new ArrayList<String>();
        }

        List<String> result = cutString(s, 3);
        return result;
    }

    // cut the string to n pieces (n=cuts), each piece is a valid ip intger
    private List<String> cutString(String s, int cuts) {
        if (s == null || s.length() == 0) {
            return null;
        }

        List<String> result = new ArrayList<String>();
        if (cuts == 0) {
            int k = Integer.parseInt(s);
            // valid ip int is [0, 255] and converted number should not trim
            // leading 0, hence string and number must have same number of
            // digits
            if (k >= 0 && k <= 255 && (s.length() == Integer.toString(k).length())) {
                result.add(s);
                return result;
            }
            return null;
        }
        for (int i = 1; i <= 3; i++) { // 255 has ony 3 digits
            if (i > s.length()) {
                break; // out of range
            }
            String head = s.substring(0, i);
            int ip = Integer.parseInt(head);
            if (ip >= 0 && ip <= 255 && (head.length() == Integer.toString(ip).length())) {
                List<String> partial = cutString(s.substring(i), cuts - 1);
                if (partial != null) {
                    // attach head ip to each string
                    for (String substr : partial) {
                        StringBuilder sb = new StringBuilder();
                        sb.append(ip).append(".").append(substr);
                        result.add(sb.toString());
                    }
                }
            }
        }
        return result;
    }

    /**
     * Letter Combinations of a Phone Number
     *
     * Given a digit string, return all possible letter combinations that the
     * number could represent.
     *
     * A mapping of digit to letters (just like on the telephone buttons) is
     * given below.
     *
     *
     *
     * Input:Digit string "23" Output: ["ad", "ae", "af", "bd", "be", "bf",
     * "cd", "ce", "cf"].
     */
    char[][] dict = { {}, {}, { 'a', 'b', 'c' }, { 'd', 'e', 'f' }, { 'g', 'h', 'i' }, { 'j', 'k', 'l' },
            { 'm', 'n', 'o' }, { 'p', 'q', 'r', 's' }, { 't', 'u', 'v' }, { 'w', 'x', 'y', 'z' } };

    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<String>();
        if (digits == null || digits.length() == 0) {
            result.add("");
            return result;
        }

        char first = digits.charAt(0);
        char[] headChars = dict[first - '0'];
        if (digits.length() == 1) {
            for (char ch : headChars) {
                result.add(String.valueOf(ch));
            }
            return result;
        }
        List<String> part = letterCombinations(digits.substring(1));
        for (char ch : headChars) {
            for (String s : part) {
                StringBuilder sb = new StringBuilder();
                sb.append(String.valueOf(ch));
                sb.append(s);
                result.add(sb.toString());
            }
        }

        return result;
    }

    /**
     * insertion sort linked list
     *
     */
    public ListNode insertionSortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode curr = head;
        ListNode newHead = curr;
        while (curr.next != null) {
            if (curr.next.val >= curr.val) {
                curr = curr.next;
                continue;
            }
            // next is smaller, should swap with previous nodes
            ListNode currnext = curr.next;
            curr.next = curr.next.next; // link the second half first

            ListNode n = newHead;
            // search all node before curr
            // currnext is smallest
            if (currnext.val < newHead.val) {
                currnext.next = newHead;
                newHead = currnext;
            } else { // currnext is in the middle
                while (n != curr) {
                    if (n.val <= currnext.val && n.next.val > currnext.val) {
                        ListNode nnext = n.next;
                        n.next = currnext;
                        currnext.next = nnext;
                        break;
                    }
                    n = n.next;
                }
            }
        }
        return newHead;
    }

    /**
     * reserver words in a string
     *
     * idea: do it in one loop with a stack to track strings, thus it is better
     * than reverse strings twice 1. move start to next non empty, and set end
     * correspondingly 2. add substring(start, end) to stack 3. pop up stack to
     * stringbuilder
     */

    public String reverseWords(String s) {
        if (s == null || s.length() == 0) {
            return s;
        }

        StringBuilder sb = new StringBuilder();
        Stack<String> stack = new Stack<String>();

        // reverse words in arr
        int start = 0;
        int end = 0;
        while (start < s.length()) {
            while (start < s.length() && s.charAt(start) == ' ') {
                start++;
            }
            if (start > s.length() - 1) {
                break;
            }
            end = start;
            while (end < s.length() && s.charAt(end) != ' ') {
                end++;
            }

            if (end > s.length() - 1) {
                end = s.length();
            }
            stack.push(s.substring(start, end));
            // important - advance start to head of next word to avoid stuck in
            // first word
            start = end;
        }
        while (!stack.isEmpty()) {
            sb.append(stack.pop());
            sb.append(" ");
        }
        sb.setLength(Math.max(sb.length() - 1, 0));
        return sb.toString();
    }

    // swap string from index start to end
    public void swap(char[] arr, int start, int end) {
        if (start > arr.length - 1 || end < 0 || end > arr.length - 1 || end <= start) {
            return;
        }
        for (int i = start; i < (start + end + 1) / 2; i++) {
            char ch = arr[i];
            arr[i] = arr[start + end - i];
            arr[start + end - i] = ch;
        }
    }

    /**
     * Clone graph
     */
    class UndirectedGraphNode {
        int label;
        List<UndirectedGraphNode> neighbors;

        UndirectedGraphNode(int x) {
            label = x;
            neighbors = new ArrayList<UndirectedGraphNode>();
        }
    };

    // idea: use a map to track <originalNode, copyNode>, and a queue for BFS
    // walk through
    public UndirectedGraphNode cloneGraph_BFS_Iterative(UndirectedGraphNode node) {
        // BFS - a queue to walk through nodes, a hashmap<node, newCopy> to
        // record visited
        if (node == null) {
            return null;
        }

        Map<UndirectedGraphNode, UndirectedGraphNode> visited = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
        Queue<UndirectedGraphNode> q = new LinkedList<UndirectedGraphNode>();
        q.offer(node);
        while (!q.isEmpty()) {
            UndirectedGraphNode n = q.poll(); // BFS
            UndirectedGraphNode copy = null;
            if (!visited.containsKey(n)) {
                copy = new UndirectedGraphNode(n.label); // clone of node n
            } else {
                copy = visited.get(n);
            }
            visited.put(n, copy);
            // clone n's neighbors
            for (UndirectedGraphNode neighbor : n.neighbors) {
                UndirectedGraphNode neighborCopy = visited.containsKey(neighbor) ? visited.get(neighbor)
                        : new UndirectedGraphNode(neighbor.label);
                copy.neighbors.add(neighborCopy);
                if (!visited.containsKey(neighbor)) {
                    q.add(neighbor);
                }
                visited.put(neighbor, neighborCopy);
            }
        }
        return visited.get(node);
    }

    // use DFS - preorder
    public UndirectedGraphNode cloneGraph_DFS_Interative(UndirectedGraphNode node) {
        // DFS - a queue to walk through nodes, a hashmap<node, newCopy> to
        // record visited
        if (node == null) {
            return null;
        }

        Map<UndirectedGraphNode, UndirectedGraphNode> visited = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
        Stack<UndirectedGraphNode> stack = new Stack<UndirectedGraphNode>();
        stack.push(node);

        // preorder
        while (!stack.isEmpty()) {
            UndirectedGraphNode curr = stack.pop();
            UndirectedGraphNode currCopy = visited.containsKey(curr) ? visited.get(curr) : new UndirectedGraphNode(
                    curr.label);
            visited.put(curr, currCopy);

            for (UndirectedGraphNode neighbor : curr.neighbors) {
                UndirectedGraphNode neighborCopy = visited.containsKey(neighbor) ? visited.get(neighbor)
                        : new UndirectedGraphNode(neighbor.label);
                currCopy.neighbors.add(neighborCopy);
                if (!visited.containsKey(neighbor)) {
                    stack.push(neighbor);
                }
                visited.put(neighbor, neighborCopy);

            }

        }

        return visited.get(node);
    }

    /**
     * Swap Nodes in Pairs
     *
     * Given a linked list, swap every two adjacent nodes and return its head.
     *
     * For example, Given 1->2->3->4, you should return the list as 2->1->4->3.
     *
     * Your algorithm should use only constant space. You may not modify the
     * values in the list, only nodes itself can be changed.
     *
     * Idea - set a dummy node on head to avoid corner case
     */
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode prevEnd = dummy;
        ListNode first = head;

        while (first != null) {
            if (first.next == null) {
                break;
            }
            ListNode second = first.next;
            // swap first and second
            ListNode secNext = second.next;
            second.next = first;
            first.next = secNext;
            // connecting prev end with this new swapped pair
            prevEnd.next = second;
            prevEnd = first;
            first = first.next;
        }
        return dummy.next;
    }

    /**
     * Reverse linked list nodes between m and n
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        if (head == null || head.next == null) {
            return head;
        }

        if (m == n) {
            return head;
        }
        // 1 °‹ m °‹ n °‹ length of list.
        int count = 1;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode startprev = dummy;
        while (count++ < m) {
            startprev = startprev.next;
        }
        // now prev points to previous node of m-th node
        // reverse prev.next to n-th element
        ListNode start = startprev.next;
        ListNode prev = startprev;
        ListNode startnext = null;
        // after reverse, startprev.next = start && start.next = startprev, a
        // circle, we reset
        while (count <= n + 1) {
            startnext = start.next;
            start.next = prev;
            prev = start;
            start = startnext;

            count++;
        }
        startprev.next.next = start;
        startprev.next = prev;

        return dummy.next;
    }

    /**
     * Add Two Numbers
     *
     * You are given two linked lists representing two non-negative numbers. The
     * digits are stored in reverse order and each of their nodes contain a
     * single digit. Add the two numbers and return it as a linked list.
     *
     * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4) Output: 7 -> 0 -> 8
     *
     * Key point: remember to add last carry when both list are gone
     *
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }

        int carry = 0;
        ListNode ret = new ListNode(0);
        ListNode it = ret;
        while (l1 != null && l2 != null) {
            int sum = l1.val + l2.val + carry;
            int d = sum % 10;
            carry = sum / 10;
            ListNode node = new ListNode(d);
            it.next = node;
            l1 = l1.next;
            l2 = l2.next;
            it = it.next;
        }
        while (l1 != null) {
            int sum = l1.val + carry;
            int d = sum % 10;
            carry = sum / 10;
            ListNode node = new ListNode(d);
            it.next = node;
            l1 = l1.next;
            it = it.next;
        }
        while (l2 != null) {
            int sum = l2.val + carry;
            int d = sum % 10;
            carry = sum / 10;
            ListNode node = new ListNode(d);
            it.next = node;
            l2 = l2.next;
            it = it.next;
        }
        if (carry == 1) { // don't forget last carry
            ListNode node = new ListNode(1);
            it.next = node;
        }

        return ret.next;
    }

    /**
     * Minimum Path Sum
     *
     * Given a m x n grid filled with non-negative numbers, find a path from top
     * left to bottom right which minimizes the sum of all numbers along its
     * path.
     *
     * Note: You can only move either down or right at any point in time.
     *
     * Idea - dp, dp[i][j] =
     * Math.min(dp[i-1][j]+grid[i][j],dp[i][j-1]+grid[i][j]), but need to
     * consider corner case when i-1 and j-1 is out of array boundary
     *
     */
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        // use dp
        int[][] dp = new int[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                dp[i][j] = i == 0 ? (j == 0 ? grid[i][j] : dp[i][j - 1] + grid[i][j]) : (j == 0 ? dp[i - 1][j]
                        + grid[i][j] : Math.min(dp[i - 1][j] + grid[i][j], dp[i][j - 1] + grid[i][j]));
            }
        }
        return dp[grid.length - 1][grid[0].length - 1];
    }

    /**
     * Running sum - maximal subarray
     */
    public int maxSubArray(int[] A) {
        int max = Integer.MIN_VALUE;
        int maxsofar = 0;

        for (int element : A) {
            maxsofar = Math.max(maxsofar + element, element);
            max = Math.max(max, maxsofar);
        }
        return max;
    }

    /*
     * Edit word distance
     * 
     * Given two words word1 and word2, find the minimum number of steps
     * required to convert word1 to word2. (each operation is counted as 1
     * step.)
     * 
     * You have the following 3 operations permitted on a word:
     * 
     * a) Insert a character b) Delete a character c) Replace a character
     * 
     * 
     * Idea: dp, compare replace/delete/insert and + 1, also init add first row
     * and column to the length to word1 and word1 separately, meaning
     * converting from empty string to one or the other is essentially the
     * length of each word -- see "algorithm design manual" book ,p284
     */
    public int minDistance(String word1, String word2) {
        // dp[i][j] record the cost to convert word1[0..i] to word2[0..j]
        if (word1 == null || word1.length() == 0) {
            return word2 == null ? 0 : word2.length();
        }
        if (word2 == null || word2.length() == 0) {
            return word1 == null ? 0 : word1.length();
        }

        int[][] cost = new int[word1.length() + 1][word2.length() + 1];

        // cost[i][j] = MIN of :
        // 1. match: cost[i-1][j-1] if word1[i-1][j-1] = word2[i-1][j-1] but
        // word1[i][j]==word2[i][j]
        // 2. replace: cost[i-1][j-1]+1 if word1[i-1][j-1] = word2[i-1][j-1] but
        // word1[i][j]!=word2[i][j]
        // 3. delete: cost[i-1][j]+1
        // 4. insert: cost[i][j-1]+1

        // note: init cost[0] and cost[x][0] as converting from empty to the
        // string
        // this setup saves condition when i-1 and j-1 out of range
        for (int i = 0; i < word1.length() + 1; i++) {
            cost[i][0] = i;
        }
        for (int j = 0; j < word2.length() + 1; j++) {
            cost[0][j] = j;
        }
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                int replace = (word1.charAt(i - 1) == word2.charAt(j - 1)) ? cost[i - 1][j - 1]
                        : cost[i - 1][j - 1] + 1;
                int insert = cost[i][j - 1] + 1;
                int delete = cost[i - 1][j] + 1;
                cost[i][j] = Math.min(Math.min(replace, insert), delete);
            }
        }
        return cost[word1.length()][word2.length()];
    }

    /**
     * Word Break
     *
     * Given a string s and a dictionary of words dict, determine if s can be
     * segmented into a space-separated sequence of one or more dictionary
     * words.
     *
     * For example, given s = "leetcode", dict = ["leet", "code"].
     *
     * Return true because "leetcode" can be segmented as "leet code".
     */
    public boolean wordBreak_recursive(String s, Set<String> dict) {
        // recursive
        if (s == null || s.length() == 0) {
            return true;
        }

        for (int i = 1; i <= s.length(); i++) {
            String first = s.substring(0, i);
            String rest = s.substring(i);
            if (dict.contains(first)) {
                boolean match = wordBreak_recursive(rest, dict);
                if (match) {
                    return true;
                }
            }
        }
        // nothing matched
        return false;
    }

    // dp solution for word break -
    // // dp[i] means if s[0..i] can be partitioned,
    // i=0,.. s.length()
    public boolean wordBreak(String s, Set<String> dict) {
        // recursive and convert it to dp
        // dp[i] means if s[0..i] can be partitioned,
        // i=0,.. s.length()
        if (s == null || s.length() == 0) {
            return true;
        }

        // default is false
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true; // empty string
        for (int i = 1; i <= s.length(); i++) {
            // compute [0..i]
            for (int j = 0; j < i; j++) {
                // scan all prefix in [0...i-1]
                if (dp[j] == true && dict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }

    /**
     * Word Break II
     *
     * Given a string s and a dictionary of words dict, add spaces in s to
     * construct a sentence where each word is a valid dictionary word.
     *
     * Return all such possible sentences.
     *
     * For example, given s = "catsanddog", dict = ["cat", "cats", "and",
     * "sand", "dog"].
     *
     * A solution is ["cats and dog", "cat sand dog"].
     */
    // backward
    public List<String> wordBreakII_backward(String s, Set<String> dict) {
        Map<Integer, List<String>> validMap = new HashMap<Integer, List<String>>();

        // initialize the valid values
        List<String> l = new ArrayList<String>();
        l.add("");
        validMap.put(s.length(), l);

        // generate solutions from the end
        for (int i = s.length() - 1; i >= 0; i--) {
            List<String> values = new ArrayList<String>();
            for (int j = i + 1; j <= s.length(); j++) {
                if (dict.contains(s.substring(i, j))) {
                    for (String word : validMap.get(j)) {
                        values.add(s.substring(i, j) + (word.isEmpty() ? "" : " ") + word);
                    }
                }
            }
            validMap.put(i, values);
        }
        return validMap.get(0);
    }

    // forward
    public List<String> wordBreak_forward(String s, Set<String> dict) {
        if (s == null || s.length() == 0) {
            return new ArrayList<String>();
        }

        // map index to valid word list
        Map<Integer, List<String>> map = new HashMap<Integer, List<String>>();
        List<String> ll = new ArrayList<String>();
        ll.add("");
        map.put(0, ll);

        for (int i = 1; i <= s.length(); i++) {
            // compute [0..i]
            List<String> list = new ArrayList<String>();
            for (int j = 0; j < i; j++) {
                // scan all prefix in [0...i-1]
                String lasthalf = s.substring(j, i);
                if (dict.contains(lasthalf)) {
                    for (String word : map.get(j)) {
                        list.add(word + (word.isEmpty() ? "" : " ") + lasthalf);
                    }
                }
            }
            map.put(i, list);
        }
        return map.get(s.length());
    }

    /**
     * Unique Paths
     *
     * A robot is located at the top-left corner of a m x n grid (marked 'Start'
     * in the diagram below).
     *
     * The robot can only move either down or right at any point in time. The
     * robot is trying to reach the bottom-right corner of the grid (marked
     * 'Finish' in the diagram below).
     *
     * How many possible unique paths are there?
     */
    public int uniquePaths(int m, int n) {
        // dp[i][j] is the count of ways to reach (i,j)
        // then dp[i][j] = dp[i-1][j] + dp[i][j-1]
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                dp[i][j] = (i < 1) ? 1 : ((j < 1) ? 1 : dp[i - 1][j] + dp[i][j - 1]);
            }
        }
        return dp[m - 1][n - 1];
    }

    /**
     * Unique Paths II
     *
     * Follow up for "Unique Paths":
     *
     * Now consider if some obstacles are added to the grids. How many unique
     * paths would there be?
     *
     * An obstacle and empty space is marked as 1 and 0 respectively in the
     * grid.
     *
     * For example, There is one obstacle in the middle of a 3x3 grid as
     * illustrated below.
     *
     * [ [0,0,0], [0,1,0], [0,0,0] ] The total number of unique paths is 2.
     *
     * Note: m and n will be at most 100.
     *
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        // dp[i][j] is the count of ways to reach (i,j)
        // then dp[i][j] = dp[i-1][j] + dp[i][j-1]
        if (obstacleGrid == null || obstacleGrid.length == 0) {
            return 0;
        }
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;

        // to faciliate i-1 and j-1, we add one more row and column initialized
        // to 0
        int[][] dp = new int[m + 1][n + 1];
        // first column init to 0
        for (int i = 0; i < m + 1; i++) {
            dp[i][0] = 0;
        }
        // first row init to 0
        for (int i = 0; i < n + 1; i++) {
            dp[0][i] = 0;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (obstacleGrid[i - 1][j - 1] == 1) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = (i == 1 && j == 1) ? 1 : dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * Unique Binary Search Trees
     *
     *
     *Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
     // @formatter:off
     //   For example,
     //   Given n = 3, there are a total of 5 unique BST's.

     //      1         3     3      2      1
     //       \       /     /      / \      \
     //        3     2     1      1   3      2
     //       /     /       \                 \
     //      2     1         2                 3
     */
    // @formatter: on

    public int numTrees(int n) {
        // dp - assume dp[n-1] is the ways to store n-1 numbers in binary tree,
        // the dp[n] = sigma(dp[i]*dp[n-1-i]),  i=0...n-1
        // the idea is n-1 nodes are divided to left and right, finish the combo with multiplication principel
        // so it is dp[i]*dp[n-1-i]
        int[] dp = new int[n+1];
        dp[0] = 1; // dummy for convenience of multiply
        for (int i=1; i<=n; i++) {
            for (int j=0; j<i;j++){
                dp[i] += dp[j]*dp[i-1-j]; // sum up all left (j nodes) and right (n-1-j) nodes combo
            }
        }
        return dp[n];
    }

    /**
     * Palindrome Partitioning
     *
     * Given a string s, partition s such that every substring of the partition is a palindrome.
     * Return all possible palindrome partitioning of s.
     * For example, given s = "aab",
     * Return
     *  [
            ["aa","b"],
            ["a","a","b"]
        ]
     */
    // DP - record with index i with valid string partition list
    // the expand to i+1
    public List<List<String>> partition(String s) {
        if (s==null || s.length()==0) {
            return new ArrayList<List<String>>();
        }

        // map index [0...len] to list of strings, note i is mapped to i+1
        Map<Integer, List<List<String>>> map = new HashMap<Integer, List<List<String>>>();
        List<List<String>> list = new ArrayList<List<String>>();
        list.add(null);
        map.put(0, list);

        for (int i=1; i<=s.length();i++) {
            // index i's list of strings
            List<List<String>> list2 = new ArrayList<List<String>>();
            for (int j=0; j<i;j++) {
                String lasthalf = s.substring(j,i);
                if (isPalin(lasthalf)) {
                    for (List<String> it : map.get(j)){
                        // need to make a copy of it, otherwise since it is reference,
                        // it would be written to all mapped spots
                        List<String> copy = new ArrayList<String>();
                        if (it != null) { // map[0] is dummy list, should skip
                            copy.addAll(it);
                        }
                        copy.add(lasthalf);
                        list2.add(copy);
                    }
                }
            }
            map.put(i, list2);
        }
        return map.get(s.length());
    }

    /**
     * Palindrome Partitioning II
     *
     * Given a string s, partition s such that every substring of the partition is a palindrome.
     * Return the minimum cuts needed for a palindrome partitioning of s.
     * For example, given s = "aab",
     * Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
     *
     * fantastic logic below - based on
     *
     *  it works because of the following invariance: cut[k] is correct for every k <= i. You could prove it by induction for i=0,1,2..
     */
    public int palindrome_minCut(String s) {
        if(s.length()==0) {
            return 0;
        }
        // dp cut array count number of min cuts required from s[i....len-1]
        int[] cut = new int[s.length()+1];
        //Arrays.fill(cut, Integer.MAX_VALUE);
        for(int i=0;i<s.length();i++) {
            cut[i]=Integer.MAX_VALUE;
        }
        cut[s.length()] = -1;
        for(int i=s.length()-1; i>=0; i--){
            // odd length palindrome
            for(int a=i,b=i;a>=0 && b<s.length() && s.charAt(a)==s.charAt(b);a--,b++) {
                cut[a]=Math.min(cut[a], 1 + cut[b+1]);
            }
            // even length palindrom
            for(int a=i,b=i+1;a>=0 && b<s.length() && s.charAt(a)==s.charAt(b);a--,b++) {
                cut[a]=Math.min(cut[a], 1 + cut[b+1]);
            }
        }
        return cut[0];
    }

    // time exception - my original thought
    public int minCut_TLE(String s) {
        if (s==null||s.length()==0) {
            return 0;
        }

        // dp - cut[i] counts cuts for substring[0..i]
        int[] cut = new int[s.length()];
        cut[0] = 0; // single char no need for cut

        for (int i=1; i<s.length(); i++){
            if (isPalin(s.substring(0,i+1))) {
                cut[i] = 0; // natrual palin, no cut needed
                continue;
            } else {
                cut[i] = Integer.MAX_VALUE;
            }
            for (int j=1; j<=i; j++) {
                if(isPalin(s.substring(j,i+1))) {
                    cut[i] = Math.min(cut[i], cut[j-1]+1);
                }
            }
        }
        return cut[s.length()-1];
    }

    //@formatter:off
    /**
    public boolean isPalin(String s) {
        int len = s.length();
        for (int i = 0; i < len / 2; i++) {
            if (s.charAt(i) != s.charAt(len - i - 1)) {
                return false;
            }
        }
        return true;
    }
     */
    // @formatter:on

    /**
     * Decode Ways
     *
     * A message containing letters from A-Z is being encoded to numbers using
     * the following mapping: 'A' -> 1 'B' -> 2 ... 'Z' -> 26
     *
     * Given an encoded message containing digits, determine the total number of
     * ways to decode it.
     *
     * For example, Given encoded message "12", it could be decoded as "AB" (1
     * 2) or "L" (12).
     *
     * The number of ways decoding "12" is 2.
     */
    public int numDecodings(String s) {
        // dp[i] = ways decoding s[0...i],
        // dp[i] has combo of dp[i-2]+dp[i-1], dp[i-2], or dp[i-1]
        if (s == null || s.length() == 0) {
            return 0;
        }

        int[] dp = new int[s.length() + 1];
        dp[0] = 1; // empty string, dp[1] points to s[0]
        if (!isValidNubmer(s.substring(0, 1))) { // first char is invalid, no
            // way to to decode
            return 0;
        } else {
            dp[1] = 1; // one way to decode s[0]
        }
        for (int i = 1; i < s.length(); i++) {// string loop
            if (!isValidNubmer(s.substring(i, i + 1))) { // s[i] is not valid
                if (isValidNubmer(s.substring(i - 1, i + 1))) { // s[i-1..i] is
                    // valid
                    dp[i + 1] = dp[i - 1];
                } else { // s[i-1..i] is not valid number too
                    return 0;
                }
            } else { // s[i] is valid nubmer
                if (isValidNubmer(s.substring(i - 1, i + 1))) { // s[i-1..i] is
                    // valid
                    dp[i + 1] = dp[i] + dp[i - 1];
                } else { // s[i-1..i] is not valid number
                    dp[i + 1] = dp[i];
                }
            }
        }
        return dp[s.length()];
    }

    public boolean isValidNubmer(String s) {
        if (s == null || s.length() == 0 || s.length() > 2) {
            return false;
        }

        if (s.charAt(0) == '0') {
            return false; // no leading 0
        }

        int n = Integer.parseInt(s);
        return n >= 1 && n <= 26;

    }

    /**
     * Sum Root to Leaf Numbers
     *
     * Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

 // @formatter:off
An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.

For example,

    1
   / \
  2   3
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.

Return the sum = 12 + 13 = 25.
     */
    // @formatter:on

    public int sumNumbers(TreeNode root) {
        int total = 0;
        LinkedList<TreeNode> q = new LinkedList<TreeNode>();
        LinkedList<Integer> sumq = new LinkedList<Integer>();
        if (root != null) {
            q.addLast(root);
            sumq.addLast(root.val);
        }
        while (!q.isEmpty()) {
            TreeNode current = q.removeFirst();
            int partialSum = sumq.removeFirst();
            if (current.left == null && current.right == null) {
                total += partialSum;
            } else {
                if (current.right != null) {
                    q.addLast(current.right);
                    sumq.addLast(partialSum * 10 + current.right.val);
                }
                if (current.left != null) {
                    q.addLast(current.left);
                    sumq.addLast(partialSum * 10 + current.left.val);
                }

            }

        }
        return total;
    }

    /**
     * Spiral Matrix
     *
     * Given a matrix of m x n elements (m rows, n columns), return all elements
     * of the matrix in spiral order.
     *
     * For example, Given the following matrix:
     */
    //@formatter:off
    /* [
         [ 1, 2, 3 ],
         [ 4, 5, 6 ],
         [ 7, 8, 9 ]
        ]
        You should return [1,2,3,6,9,8,7,4,5].
      @formatter:on
     */
    public List<Integer> spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0) {
            return new ArrayList<Integer>();
        }

        List<Integer> ret = new ArrayList<Integer>();
        int startRow = 0;
        int endRow = matrix.length - 1;
        int startCol = 0;
        int endCol = matrix[0].length - 1;

        int i = 0;
        int j = 0;
        int step = 0;
        while (startRow <= endRow && startCol <= endCol) {
            // if (i >= startRow && i <= endRow && j >= startCol && j <=endCol)
            // ret.add(matrix[i][j]);

            // step 1 move right,
            // step 2 move left
            // step 3 move up,
            // step 4 move down
            // to avoid dup, rewrite visited number with uniqe number, say
            // max_int

            if (step == 0) {
                if (j <= endCol) {
                    if (matrix[i][j] != Integer.MAX_VALUE) {
                        ret.add(matrix[i][j]);
                        matrix[i][j] = Integer.MAX_VALUE;
                    }
                    j++;
                } else { // end of row
                    startRow++;
                    j--; // back to endCol
                    step = (++step) % 4;
                }
                continue;
            }
            if (step == 1) {
                if (i <= endRow) {
                    if (matrix[i][j] != Integer.MAX_VALUE) {
                        ret.add(matrix[i][j]);
                        matrix[i][j] = Integer.MAX_VALUE;
                    }
                    i++;
                } else { // last row
                    endCol--;
                    i--;
                    step = (++step) % 4;
                }
                continue;
            }
            if (step == 2) {
                if (j >= startCol) {
                    if (matrix[i][j] != Integer.MAX_VALUE) {
                        ret.add(matrix[i][j]);
                        matrix[i][j] = Integer.MAX_VALUE;
                    }
                    j--;
                } else { // first col
                    endRow--;
                    j++;
                    step = (++step) % 4;
                }
                continue;
            }
            if (step == 3) {
                if (i >= startRow) {
                    if (matrix[i][j] != Integer.MAX_VALUE) {
                        ret.add(matrix[i][j]);
                        matrix[i][j] = Integer.MAX_VALUE;
                    }
                    i--;
                } else { // touched start row
                    startCol++;
                    i++;
                    step = (++step) % 4;
                }
                continue;
            }
        }

        return ret;
    }

    /**
     *
     */
    static class Point {
        int x;
        int y;

        Point() {
            x = 0;
            y = 0;
        }

        Point(int a, int b) {
            x = a;
            y = b;
        }
    }

    /**
     * Max Points on a Line
     *
     * Given n points on a 2D plane, find the maximum number of points that lie
     * on the same straight line.
     */
    // complexity: O(N^2)
    // this algorithm has a bug, see below input/output - I guess might be
    // glitch using double as key
    /*
     * Input:
     * [(-435,-347),(-435,-347),(609,613),(-348,-267),(-174,-107),(87,133)
     * ,(-87,-
     * 27),(-609,-507),(435,453),(-870,-747),(-783,-667),(0,53),(-174,-107
     * ),(783,
     * 773),(-261,-187),(-609,-507),(-261,-187),(-87,-27),(87,133),(783,773
     * ),(-783
     * ,-667),(-609,-507),(-435,-347),(783,773),(-870,-747),(87,133),(87,133
     * ),(870
     * ,853),(696,693),(0,53),(174,213),(-783,-667),(-609,-507),(261,293),(
     * 435,453),(261,293),(435,453)]
     * 
     * Output: 23 Expected: 37
     */
    public int maxPoints(Point[] points) {
        // idea: use slope and y-intercept as key to build a map
        if (points == null || points.length == 0) {
            return 0;
        }

        int max = Integer.MIN_VALUE;

        // first key is slope; the second key in internal map is y-intercept,
        // thus the list mapping is unique
        Map<Double, HashMap<Double, HashSet<Point>>> map = new HashMap<Double, HashMap<Double, HashSet<Point>>>();

        // scan the points to check how man duplicate we have
        // first key is x, second key if y - we should use Point as key but that
        // needs rewrite equals and hashcode() in Point class
        // we build a list without duplicate thus reduce a lot headache
        List<Point> noduplist = new ArrayList<Point>();
        Map<Integer, HashMap<Integer, Integer>> dupMap = new HashMap<Integer, HashMap<Integer, Integer>>();
        int totalDup = 0;
        for (int i = 0; i < points.length; i++) {
            if (!dupMap.containsKey(points[i].x)) { // new point
                HashMap<Integer, Integer> m = new HashMap<Integer, Integer>();
                m.put(points[i].y, 0); // no dup = 0
                dupMap.put(points[i].x, m);
                noduplist.add(points[i]);
            } else {
                HashMap<Integer, Integer> m = dupMap.get(points[i].x);
                if (m.containsKey(points[i].y)) {
                    int count = m.get(points[i].y);
                    m.put(points[i].y, ++count);
                    totalDup++;
                } else { // new point
                    m.put(points[i].y, 0);
                    noduplist.add(points[i]);
                }
                dupMap.put(points[i].x, m);
            }
        }

        Point[] nodup = new Point[noduplist.size()];
        int kk = 0;
        for (Point p : noduplist) {
            nodup[kk++] = p;
        }

        if (nodup.length < 3) {
            return nodup.length + totalDup;
        }

        for (int i = 0; i < nodup.length; i++) {
            for (int j = i + 1; j < nodup.length; j++) {
                double base = nodup[j].x - nodup[i].x;
                double slope = (base == 0) ? Double.MAX_VALUE : (nodup[j].y - nodup[i].y) / base;
                double intercept = (base == 0) ? (double) nodup[j].x : nodup[i].y - (double) nodup[i].x
                        * (double) (nodup[j].y - nodup[i].y) / base;

                // to avoid -0.0 vs 0.0, this is bizzar but as key -0.0 is
                // different from 0.0
                slope = Math.abs(slope) == 0 ? 0 : slope;
                intercept = Math.abs(intercept) == 0 ? 0 : intercept;

                HashMap<Double, HashSet<Point>> inmap = map.get(slope);
                if (inmap == null) { // new slope
                    inmap = new HashMap<Double, HashSet<Point>>();
                }
                HashSet<Point> set = inmap.get(intercept);
                if (set == null) { // new intercept
                    // first time - add p[i]
                    set = new HashSet<Point>();
                    set.add(nodup[i]);
                }
                // if this is duplicate, then simple increase max only, but do
                // NOT adding dup to the list
                // say, input if [1,1],[1,1], [1,2], when we reached [1,1] here
                // need to check both i and j's dup count
                if (nodup[j].x == nodup[i].x && nodup[j].y == nodup[i].y) {
                    int dupCount1 = dupMap.get(nodup[i].x).get(nodup[i].y);
                    int dupCount2 = dupMap.get(nodup[j].x).get(nodup[j].y);
                    max = Math.max(max, set.size() + dupCount1 + dupCount2);
                    continue;
                }
                // only add p[j] is it is not already in the list
                // use hashset to make the lookup O(1)
                if (!set.contains(nodup[j])) {
                    set.add(nodup[j]);
                }
                inmap.put(intercept, set);
                map.put(slope, inmap);
                // still need to check dup with point[i] even this Point is no
                // dup, then update max
                // say, input if [1,1],[1,1], [1,2], when we reached [1,2], we
                // need to add dup to the max
                int dupCount1 = dupMap.get(nodup[i].x).get(nodup[i].y);
                int dupCount2 = dupMap.get(nodup[j].x).get(nodup[j].y);
                max = Math.max(max, set.size() + dupCount1 + dupCount2);
            }
        }
        return max;
    }

    /**
     * Sort List
     *
     * Sort a linked list in O(n log n) time using constant space complexity.
     *
     * User merge sort
     */
    // return longer last half
    public ListNode getMiddle(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }

        ListNode n1 = head;
        ListNode n2 = head;
        while (n2 != null) {
            n2 = n2.next;
            if (n2 == null) {
                return n1;
            }
            n1 = n1.next;
            n2 = n2.next;
        }
        return n1;
    }

    public ListNode sortList(ListNode head) {
        // use merge sort - top down or bottom up
        if (head == null || head.next == null) {
            return head;
        }
        ListNode middle = getMiddle(head);
        ListNode tail = head;
        while (tail.next != middle) {
            tail = tail.next;
        }
        // now set first half list to null terminated
        tail.next = null;
        ListNode first = sortList(head);
        ListNode second = sortList(middle);
        // merge first and second lists
        ListNode dummy = new ListNode(0);
        ListNode n1 = first;
        ListNode n2 = second;
        ListNode run = dummy;
        while (n1 != null && n2 != null) {
            if (n1.val < n2.val) {
                run.next = n1;
                n1 = n1.next;
                run = run.next;
            } else {
                run.next = n2;
                n2 = n2.next;
                run = run.next;
            }
        }
        if (n1 != null) {
            run.next = n1;
        } else if (n2 != null) {
            run.next = n2;
        }

        return dummy.next;
    }

    /**
     * Candy
     *
     */

    //@formatter:off
    /*      Ratings:
                                P = max(0,c)+1
              P = max(a,b)+1  | |
              |<-----       | | |
            | |     |     | | | |       P = max(d,0)+1
          | | | |   |   | | | | |       | | |
        | | | | | | ^ | | | | | | |     |<|-|--
      | | | | | | | | | | | | | | | | | | | | |
      Candies:                                ^
      1 2 3 4 5 3 2 1 2 3 4 5 6 3 2 1 1 2 1 1 END
      |-----|   |---|---------| * |-| *   * *
            a   b               1 c   d
     */
    //@formatter:on

    public int candy(int[] ratings) {
        // this is like a wave, each bottom is 1, the peak's value
        // depends on longest up or down from bottom
        // algo : scan the array
        // if value is going up, update up counter;
        // if value is going down, update down counter until hit bottom or end
        // of array
        // compare the length of up and down counter, the largest the peak's
        // value;
        // then updating the rest tild done
        if (ratings == null || ratings.length < 2) {
            return ratings.length;
        }

        int up = 0;
        int down = 0;
        int i = 1;
        int lastTrend = 0; // 0-even, 1-up, -1 - down
        int sum = 0;
        while (i < ratings.length) {
            int trend = 0;
            if (ratings[i] > ratings[i - 1]) {
                trend = 1; // up
            } else if (ratings[i] < ratings[i - 1]) {
                trend = -1; // down
            }
            // when the trend hits bottom or even, compute up and down then
            // reset
            if (lastTrend < 0 && trend >= 0 || lastTrend > 0 && trend == 0) {
                // max(up, down) + 1 ? No. The next mountain start at the end.
                sum += sendCandy(up) + sendCandy(down) + Math.max(up, down);
                up = 0;
                down = 0;
            }
            if (trend > 0) {
                up++;
            } else if (trend < 0) {
                down++;
            } else {
                // even, reset to 1
                sum++;
            }
            lastTrend = trend;
            i++;
        }
        // max(up, down) + 1 ? Yes. There is no mountain next to it.
        sum += sendCandy(up) + sendCandy(down) + Math.max(up, down) + 1;
        return sum;
    }

    public int sendCandy(int n) {
        return n * (n + 1) / 2;
    }

    /*
     * Surrounded Regions
     * 
     * Flip surrounded regon
     */
    //@formatter:off
    /*
     * Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.

        A region is captured by flipping all 'O's into 'X's in that surrounded region.

        For example,
        X X X X
        X O O X
        X X O X
        X O X X
        After running your function, the board should be:

        X X X X
        X X X X
        X X X X
        X O X X
     *
     */
    //@formatter:on

    // this impl used too much space and caused time exception, a better idea is
    // to start to edges with 'O' using BFS
    // and mark those points as non-convertable, then scan the whole array
    public void surroundedRegion(char[][] board) {
        // this is wei qi
        // find all connected area with O, then flip them if they don't touch
        // edge or corner - meaning they are connected
        if (board == null || board.length == 0) {
            return;
        }

        // algo -
        // 1. scan array, use visited to track unsurrounded (sticky) and
        // connected for current area handling
        // 2. if we find 'O'and it is NOT in visited set, then explore it
        // 3. then find all its connected neighbors,
        // 4. if any neighor touches
        // edge/corner then the whole connected set is not surrounded, add them
        // to visited set (sticky)
        // 5. otherwise flip the whole surrounded area
        // 6. clear the connected data set

        // catch all visited 'O' Point but not surrended
        // if a point 'O' appears in visited, but
        // means it is visited but not surrounded
        // key=x, val=y
        Map<Integer, HashSet<Integer>> visited = new HashMap<Integer, HashSet<Integer>>();
        Map<Integer, HashSet<Integer>> connected = new HashMap<Integer, HashSet<Integer>>();

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == 'X' || contains(visited, i, j)) {
                    continue;
                }
                // new 'O' to be explored
                // need to return a list of connected area AND a flag showing
                // surrounded or not (touch edge)
                // use BFS
                HashSet<Integer> set = new HashSet<Integer>();
                set.add(j);
                connected.put(i, set);
                boolean surrounded = true;
                // find all the neighbors
                Queue<Point> q = new LinkedList<Point>();
                q.offer(new Point(i, j));
                // visited list for this BFS search
                Map<Integer, HashSet<Integer>> vv = new HashMap<Integer, HashSet<Integer>>();
                vv.put(i, set);

                while (!q.isEmpty()) {
                    Point p = q.poll();
                    // (x-1,y), (x+1,y), (x,y-1), (x,y+1)
                    if (p.x - 1 < 0) {
                        surrounded = false;
                    } else {
                        if (board[p.x - 1][p.y] == 'O' && !contains(vv, p.x - 1, p.y)) {
                            // this point must not be in visited, otherwise they
                            // are all connected
                            HashSet<Integer> hs = new HashSet<Integer>();
                            set.add(p.y);
                            connected.put(p.x - 1, hs);
                            q.offer(new Point(p.x - 1, p.y));
                            // add to current visited set, you have to retrive
                            // the internal set for y, which is ugly
                            // if we have Point, then we don't need this hassle
                            HashSet<Integer> ss = vv.get(p.x - 1);
                            if (ss == null) {
                                ss = new HashSet<Integer>();
                            }
                            ss.add(p.y);
                            vv.put(p.x - 1, ss);
                        }
                    }
                    // (x+1, y)
                    if (p.x + 1 >= board.length) {
                        surrounded = false;
                    } else {
                        if (board[p.x + 1][p.y] == 'O' && !contains(vv, p.x + 1, p.y)) {
                            // this point must not be in visited, otherwise they
                            // are all connected
                            HashSet<Integer> hs = new HashSet<Integer>();
                            set.add(p.y);
                            connected.put(p.x + 1, hs);
                            q.offer(new Point(p.x + 1, p.y));

                            HashSet<Integer> ss = vv.get(p.x + 1);
                            if (ss == null) {
                                ss = new HashSet<Integer>();
                            }
                            ss.add(p.y);
                            vv.put(p.x + 1, ss);
                        }
                    }
                    // (x, y-1)
                    if (p.y - 1 < 0) {
                        surrounded = false;
                    } else {
                        if (board[p.x][p.y - 1] == 'O' && !contains(vv, p.x, p.y - 1)) {
                            // this point must not be in visited, otherwise they
                            // are all connected
                            HashSet<Integer> hs = new HashSet<Integer>();
                            set.add(p.y - 1);
                            connected.put(p.x, hs);
                            q.offer(new Point(p.x, p.y - 1));

                            HashSet<Integer> ss = vv.get(p.x);
                            if (ss == null) {
                                ss = new HashSet<Integer>();
                            }
                            ss.add(p.y - 1);
                            vv.put(p.x, ss);
                        }
                    }
                    // (x, y+1)
                    if (p.y + 1 >= board[i].length) {
                        surrounded = false;
                    } else {
                        if (board[p.x][p.y + 1] == 'O' && !contains(vv, p.x, p.y + 1)) {
                            // this point must not be in visited, otherwise they
                            // are all connected
                            HashSet<Integer> hs = new HashSet<Integer>();
                            set.add(p.y + 1);
                            connected.put(p.x, hs);
                            q.offer(new Point(p.x, p.y + 1));

                            HashSet<Integer> ss = vv.get(p.x);
                            if (ss == null) {
                                ss = new HashSet<Integer>();
                            }
                            ss.add(p.y + 1);
                            vv.put(p.x, ss);
                        }
                    }
                }
                // now connected area is build
                if (surrounded) {
                    // flip connected
                    for (Entry<Integer, HashSet<Integer>> entry : connected.entrySet()) {
                        int x = entry.getKey();
                        for (Integer y : entry.getValue()) {
                            board[x][y] = 'X';
                        }
                    }
                } else { // not surrounded, need to copy to visited then clean
                    // connected
                    // copy connected to visisted
                    /*
                     * for (Entry<Integer, HashSet<Integer>> entry :
                     * connected.entrySet()) { int x = entry.getKey();
                     * HashSet<Integer> ss = visited.get(x); if (ss == null) {
                     * ss = new HashSet<Integer>(); } for (Integer y :
                     * entry.getValue()) { if (!ss.contains(y)) { ss.add(y); } }
                     * visited.put(x, ss); }
                     */
                    Map<Integer, HashSet<Integer>> tmp = new HashMap(connected);
                    tmp.keySet().removeAll(connected.keySet());
                    visited.putAll(tmp);
                }
                // clearn connected
                connected.clear();
            }
        }

    }

    // test if the set contains (x,y)
    public boolean contains(Map<Integer, HashSet<Integer>> set, int x, int y) {
        if (set == null || set.get(x) == null) {
            return false;
        }
        HashSet<Integer> s = set.get(x);
        return s.contains(y);
    }

    /**
     * Longest continouse Longest Consecutive Sequence
     *
     * Given an unsorted array of integers, find the length of the longest
     * consecutive elements sequence.
     *
     * For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive
     * elements sequence is [1, 2, 3, 4]. Return its length: 4.
     *
     * Your algorithm should run in O(n) complexity.
     */
    public int longestConsecutive(int[] num) {
        // algo
        // 1. map to a hashset
        // 2. for each set entry, check and expand continuous entry
        // 3. return longest
        if (num == null || num.length == 0) {
            return 0;
        }

        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int longest = Integer.MIN_VALUE;

        // since we cannot update entry during iterate, so we use value count
        // 0 - unavailable; 1 - available
        for (int element : num) {
            map.put(element, 1);
        }
        for (Entry<Integer, Integer> e : map.entrySet()) {
            if (e.getValue() == 0) {
                continue; // visited
            }
            int n = e.getKey();
            map.put(n, 0); // set it off
            // check n++ and n--
            int up = 0;
            int down = 0;
            int i = n;
            while (map.containsKey(++i)) {
                up++;
                map.put(i, 0);
            }
            while (map.containsKey(--n)) {
                down++;
                map.put(n, 0);
            }
            longest = Math.max(longest, up + down + 1);
        }
        return longest;
    }

    /**
     * Binary Tree Maximum Path Sum
     *
     * Given a binary tree, find the maximum path sum.
     *
     * The path may start and end at any node in the tree.
     *
     * For example: Given the below binary tree,
     */
    //@formatter:off
    /*
     *         1
              / \
             2   3
        Return 6.
     */
    //@formatter:on
    // java cannot pass ref, so use it to track historical max
    private int max = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        // algo - recursively check left and right subtree sum
        // use that to combine with current node to return to itsparent
        sum(root);
        return max;
    }

    // recursive call
    public int sum(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int l = sum(root.left);
        int r = sum(root.right);

        // check one side
        int a = Math.max(root.val, root.val + Math.max(l, r));
        // check both sides - which means local max update, but not transferable
        // to parents
        int b = Math.max(a, root.val + l + r);
        max = Math.max(max, b); // historical max
        return a;
    }

    /**
     * Triangle
     *
     * Given a triangle, find the minimum path sum from top to bottom. Each step
     * you may move to adjacent numbers on the row below.
     *
     * For example, given the following triangle
     */
    //@formatter:off
    /*
     *  [
             [2],
            [3,4],
           [6,5,7],
          [4,1,8,3]
        ]
        The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).

        Note:
        Bonus point if you are able to do this using only O(n) extra space, where n is the total number of rows in the triangle.
     */
    //@formatter:on
    public int minimumTotal(List<List<Integer>> triangle) {
        // dp bottom up:
        // only only one-dim table to record the dp process for each layer
        // previous layer is only needed
        // assuming triangle has k rows:
        // t[i] (i=0,..., k-1) is the cost at layer j (k-1...0), at layer j-1,
        // we don't need data on j, so we reuse t[i] and upate it, then update
        // t[i+1]..
        // recursive:
        // t[i] = min(t[i], t[i+1]) + triangle[k][i]

        if (triangle == null || triangle.size() == 0) {
            return 0;
        }

        int[] t = new int[triangle.size()];
        // init t to last row of triangle
        int k = 0;
        for (int n : triangle.get(triangle.size() - 1)) {
            t[k++] = n;
        }

        // bottom up, starts from the last second row n-2
        for (int row = triangle.size() - 2; row >= 0; row--) {
            for (int i = 0; i <= row; i++) {
                t[i] = Math.min(t[i], t[i + 1]) + triangle.get(row).get(i);
            }
        }
        return t[0];
    }

    /**
     * 4 sum
     *
     * Given an array S of n integers, are there elements a, b, c, and d in S
     * such that a + b + c + d = target? Find all unique quadruplets in the
     * array which gives the sum of target.
     *
     * Note: Elements in a quadruplet (a,b,c,d) must be in non-descending order.
     * (ie, a °‹ b °‹ c °‹ d) The solution set must not contain duplicate
     * quadruplets. For example, given array S = {1 0 -1 0 -2 2}, and target =
     * 0.
     *
     * A solution set is: (-1, 0, 0, 1) (-2, -1, 1, 2) (-2, 0, 0, 2)
     *
     * Idea: make pair, and store pair sum with their original index in map then
     * theck the map like 2 sum, and refer to stored index
     *
     * time complex O(N^2) space complext O(N^2)
     */
    public class Pair {
        int x, y;

        public Pair(int i, int j) {
            x = i;
            y = j;
        }
    }

    // this func convert the (i,j) (j>i) in combo to index based 1-dim array
    // n is the total length of original array num
    private int getIndex(int i, int j, int n) {
        return n * (n - 1) / 2 - (n - i - 1) * (n - i) / 2 + j - i + 1;
    }

    // check four elment
    private boolean containsList(List<List<Integer>> all, List<Integer> check) {
        for (List<Integer> list : all) {
            // check if list has same elements of set of checked
            List<Integer> copy = new ArrayList<Integer>();
            copy.addAll(check);
            for (Integer n : list) {
                if (!copy.remove(n)) {
                    break;
                }
            }
            if (copy.size() == 0) {
                return true;
            }
        }
        return false;
    }

    public List<List<Integer>> fourSum(int[] num, int target) {
        // algo
        // add every pair and store them into a map with their index list
        // from the sum result build a two dim array
        // scan the two dim-arry using two sum, then query map table for list
        if (num == null | num.length < 4) {
            return new ArrayList<List<Integer>>();
        }

        int len = num.length;
        int newlen = len * (len - 1) / 2; // combo of all pairs out of num
        int[] pair = new int[newlen]; // hold sum of every pair
        Map<Integer, List<Pair>> map = new HashMap<Integer, List<Pair>>();
        List<List<Integer>> ret = new ArrayList<List<Integer>>();

        /*
         * // preprocessing - sort it first and save lot of check int min = 0;
         * int max = 0; Arrays.sort(num); // check min and max first; int min =
         * 0; int max =0; for (int i = 0; i < 4; i++) { min += num[i]; max +=
         * num[len - 1 - i]; } if (target < min || target > max) { return ret; }
         */
        int count = 0;
        // convert (i,j) to 1-dim array, the index mapping is defined in
        // getIndex() func
        for (int i = 0; i < len - 1; i++) {
            for (int j = i + 1; j < len; j++) {
                int sum = num[i] + num[j];
                pair[count++] = sum;
                List<Pair> list = map.get(sum);
                if (list == null) {
                    list = new ArrayList<Pair>();
                }
                list.add(new Pair(i, j));
                map.put(sum, list);
            }
        }
        // now scan pair
        for (int i = 0; i < newlen - 1; i++) {
            List<Pair> list2 = map.get(target - pair[i]);
            if (list2 != null) {
                // dump the sum but no need to clean as we need all combos
                List<Pair> list1 = map.get(pair[i]);
                // no overlap, say, (1,2) and (3,4) ok, but (1,2) and (2,3) is
                // not
                for (Pair p1 : list1) {
                    for (Pair p2 : list2) {
                        if (p1.x == p2.x || p1.x == p2.y || p1.y == p2.x || p1.y == p2.y) {
                            continue;
                        }
                        List<Integer> combo = new ArrayList<Integer>();
                        // sort the four numbers, this is the stupid OJ required
                        int[] a = new int[4];
                        a[0] = num[p1.x];
                        a[1] = num[p1.y];
                        a[2] = num[p2.x];
                        a[3] = num[p2.y];
                        Arrays.sort(a);
                        for (int m = 0; m < 4; m++) {
                            combo.add(a[m]);
                        }
                        // remove dup in ret
                        if (!containsList(ret, combo)) {
                            ret.add(combo);
                        }
                    }
                }
                map.remove(pair[i]);
                map.remove(target - pair[i]);
            }
        }
        return ret;
    }

    /**
     * Path Sum II Total Accepted: 26547 Total Submissions: 99458 My Submissions
     * Question Solution Given a binary tree and a sum, find all root-to-leaf
     * paths where each path's sum equals the given sum.
     *
     * For example: Given the below binary tree and sum = 22,
     */
    //@formatter:off
    /*
     *        5
             / \
            4   8
           /   / \
          11  13  4
         /  \    / \
        7    2  5   1

        return
        [
           [5,4,11,2],
           [5,8,4,5]
        ]
     */
    //@formatter:on
    public class NodePair {
        TreeNode node;
        int val; // expected value for this node

        public NodePair(TreeNode n, int i) {
            node = n;
            val = i;
        }
    }

    // algo: use DFS + backtrack
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        // dfs + backtrack
        if (root == null) {
            return new ArrayList<List<Integer>>();
        }

        // use preorder and backtrack in iterative
        Map<TreeNode, TreeNode> backtrack = new HashMap<TreeNode, TreeNode>();
        List<List<Integer>> list = new ArrayList<List<Integer>>();
        Stack<NodePair> stack = new Stack<NodePair>();
        NodePair p = new NodePair(root, sum);
        stack.push(p);

        while (!stack.isEmpty()) {
            NodePair np = stack.pop();
            // check leaf first
            if (np.node.left == null && np.node.right == null) {
                // backtrack
                if (np.node.val != np.val) {
                    continue; // not a match
                }
                List<Integer> ll = new ArrayList<Integer>();
                TreeNode n = np.node;
                while (n != null) {
                    ll.add(0, n.val); // add parent to the head of list
                    n = backtrack.get(n); // parent
                }
                list.add(ll);
            }

            int exp = np.val - np.node.val; // count new expected sum for its
            // children
            if (np.node.right != null) {
                stack.push(new NodePair(np.node.right, exp));
                backtrack.put(np.node.right, np.node);
            }
            if (np.node.left != null) {
                stack.push(new NodePair(np.node.left, exp));
                backtrack.put(np.node.left, np.node);
            }
        }

        return list;
    }

    /**
     * Flatten Binary Tree to Linked List Total Accepted: 29526 Total
     * Submissions: 104720 My Submissions Question Solution Given a binary tree,
     * flatten it to a linked list in-place.
     *
     * For example, Given
     */
    //@formatter:off
    /*
     *           1
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
    public void flatten(TreeNode root) {
        //preorder traversal
        if (root==null) {
            return;
        }

        TreeNode prev = root;
        Stack<TreeNode> st = new Stack<TreeNode>();
        st.push(root);

        while(!st.isEmpty()) {
            TreeNode n = st.pop();

            if (n.right !=null) {
                st.push(n.right);
            }
            if (n.left !=null) {
                st.push(n.left);
            }
            if (n!=root) {
                prev.right = n;
                prev.left = null;
                prev = n;
            }
        }
    }

    /**
     * max length valid parentheses
     */
    public int longestValidParentheses(String s) {
        if (s==null || s.length()==0) {
            return 0;
        }

        // algo - use a stack to store index of '(' and last failed position - could be either '(' or ')'
        Stack<Integer> st = new Stack<Integer>();
        int max = 0;

        for (int i=0;i<s.length();i++) {
            if (s.charAt(i)=='(') {
                st.push(i);
            }
            else { // ')'
                // stack has a matching '(' on top, otherwise should write new failure position for ')'
                if (!st.isEmpty() && s.charAt(st.peek()) == '(') {
                    st.pop();
                    int lastPos = st.isEmpty() ? -1 : st.peek();
                    int len = i - lastPos;
                    max = Math.max(max, len);
                } else { // write last failure position for ')'
                    st.push(i);
                }
            }
        }
        return max;
    }

    /**
     * max rectangle
     *
     * Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing all ones and return its area.
     *
     * someone coded this algorithm - NEED understanding
     */
    public int maximalRectangle(char[][] matrix) {
        if (matrix==null||matrix.length==0||matrix[0].length==0) {
            return 0;
        }
        int cLen = matrix[0].length;    // column length
        int rLen = matrix.length;       // row length
        // height array
        int[] h = new int[cLen+1];
        h[cLen]=0;
        int max = 0;


        for (int row=0;row<rLen;row++) {
            Stack<Integer> s = new Stack<Integer>();
            for (int i=0;i<cLen+1;i++) {
                if (i<cLen) {
                    if(matrix[row][i]=='1') {
                        h[i]+=1;
                    } else {
                        h[i]=0;
                    }
                }

                if (s.isEmpty()||h[s.peek()]<=h[i]) {
                    s.push(i);
                } else {
                    while(!s.isEmpty()&&h[i]<h[s.peek()]){
                        int top = s.pop();
                        int area = h[top]*(s.isEmpty()?i:(i-s.peek()-1));
                        if (area>max) {
                            max = area;
                        }
                    }
                    s.push(i);
                }
            }
        }
        return max;
    }

    /**
     * Convert Sorted List to Binary Search Tree
     *
     * Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
     */
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }

        if (head.next == null) {
            return new TreeNode(head.val);
        }
        ListNode middle = getMiddle(head);
        ListNode tail = head;
        while (tail.next != middle) {
            tail = tail.next;
        }
        // now set first half list to null terminated
        tail.next = null;

        TreeNode root = new TreeNode(middle.val);
        TreeNode left = null;
        TreeNode right = null;

        if (head != null) {
            left = sortedListToBST(head);
        }
        if (middle.next != null) {
            right = sortedListToBST(middle.next);
        }

        root.left = left;
        root.right = right;
        return root;
    }

    /**
     * int is palindrom
     * use O(1) space please
     */
    public boolean isPalindrome(int x) {
        if (x<0) {
            return false;
        }
        return true;
    }

    /**
     * Container With Most Water
     *
     * Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai).
     * n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0).
     * Find two lines, which together with x-axis forms a container,
     * such that the container contains the most water. Note: You may not slant the container.
     */
    // algo - same as rain drop, compare left and right
    public int maxArea(int[] height) {
        // use the same idea compare left and right
        // move the shorter end till they meet
        int end = height.length-1;
        int start = 0;
        int max = Integer.MIN_VALUE;
        while(start<end) {
            max = Math.max(max, (end - start) * Math.min(height[start], height[end]));
            if (height[start] < height[end]) {
                start++;
            } else {
                end--;
            }
        }
        return max;
    }

    /**
     * Subsets
     *
     * Given a set of distinct integers, S, return all possible subsets.
     * Note:
     * Elements in a subset must be in non-descending order.
     * The solution set must not contain duplicate subsets.
     * For example,
     * If S = [1,2,3], a solution is:
        [
          [3],
          [1],
          [2],
          [1,2,3],
          [1,3],
          [2,3],
          [1,2],
          []
        ]
     */
    public List<List<Integer>> subsets(int[] S) {
        if (S==null || S.length==0) {
            // return empty set
            List<Integer> l = new ArrayList<Integer>();
            List<List<Integer>> ll = new ArrayList<List<Integer>>();
            ll.add(l);
            return ll;
        }
        // recursive
        Arrays.sort(S);
        int head = S[0];
        int[] rest = new int[S.length-1];
        for (int i=0; i<S.length-1;i++) {
            rest[i] = S[i+1];
        }

        List<List<Integer>> lists = subsets(rest);
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        result.addAll(lists);

        //make combo of head
        for (List<Integer> list : lists) {
            List<Integer> l = new ArrayList<Integer>();
            l.addAll(list);
            l.add(0, head);
            result.add(l);
        }
        return result;
    }

    /**
     * Word Search
     *
     * Given a 2D board and a word, find if the word exists in the grid.
     */
    //@formmatter:off
    /*
The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

For example,
Given board =

[
  ["ABCE"],
  ["SFCS"],
  ["ADEE"]
]
word = "ABCCED", -> returns true,
word = "SEE", -> returns true,
word = "ABCB", -> returns false.
     */
    //@formatter:on

    // Keypoint
    // 1. visited set - be carefull due to recursive call it might be written by
    // child and return a dirty table back, so you need to make a local copy or
    // make sure it returns to orignal state
    // 2. get a starting piont list, then call recursive
    public boolean exist(char[][] board, String word) {
        if (board == null || word == null) {
            return false;
        }

        // find first char then recursive call
        List<Pair> startList = getFirstPosition(board, word.charAt(0));
        for (Pair p : startList) {
            boolean[][] visited = new boolean[board.length][board[0].length];
            // compare,otherwise
            visited[p.x][p.y] = true;
            boolean found = findWord(p, board, word.substring(1), visited);
            if (found) {
                return true;
            }
        }
        // found nothing
        return false;
    }

    private boolean findWord(Pair p, char[][] b, String w, boolean[][] visited) {
        if (w == null || w.length() == 0) {
            return true;
        }

        // from p, find its horizontal and vertical neighbor
        for (int i = p.x - 1; i <= p.x + 1; i++) {
            for (int j = p.y - 1; j <= p.y + 1; j++) {
                // loop through (x,y)'s horizontal and vertical neighbor
                if (Math.abs(i - p.x) + Math.abs(j - p.y) == 1 && i >= 0 && i < b.length && j >= 0 && j < b[0].length
                        && !visited[i][j]) {
                    if (b[i][j] == w.charAt(0)) {
                        // note that if we modify visited table, it would
                        // persist and populate to caller, so we need a local
                        // copy
                        boolean[][] currVisited = new boolean[b.length][b[0].length];
                        for (int k = 0; k < b.length; k++) {
                            for (int m = 0; m < b[0].length; m++) {
                                currVisited[k][m] = visited[k][m];
                            }
                        }
                        currVisited[i][j] = true;
                        if (findWord(new Pair(i, j), b, w.substring(1), currVisited)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    private List<Pair> getFirstPosition(char[][] board, char ch) {
        List<Pair> list = new ArrayList<Pair>();

        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (board[i][j] == ch) {
                    list.add(new Pair(i, j));
                }
            }
        }
        return list;
    }

    /**
     * Recover BST tree - two elements swapped accidentally, swap them back
     */
    public void recoverTree(TreeNode root) {
        // do inorder traversal, the to be swapped nodes are the first
        // and seocond nodes out of order. Note that it may be direct parent
        // and child swap.
        if (root == null) {
            return;
        }
        TreeNode first = null; // first node to be swapped
        TreeNode second = null; // second node swap
        TreeNode prev = null; // use to track previous node for comparison
        TreeNode curr = root;
        Stack<TreeNode> st = new Stack<TreeNode>();

        while (!st.isEmpty() || curr != null) {
            if (curr != null) {
                st.push(curr);
                curr = curr.left;
            } else { // visit node
                curr = st.pop();
                if (prev == null) {
                    prev = curr;
                }
                if (prev.val > curr.val) {
                    // first time enconter
                    if (first == null) {
                        first = prev;
                        second = curr;
                    } else { // second encounter, reset second node only
                        second = curr;
                    }
                }
                prev = curr;
                curr = curr.right;
            }
        }
        int tmp = first.val;
        first.val = second.val;
        second.val = tmp;
    }

    /**
     * Intersection of Two Linked Lists
     *
     * Write a program to find the node at which the intersection of two singly
     * linked lists begins.
     */
    //@formatter:off
    /**
For example, the following two linked lists:

A:          a1 °˙ a2
                                                                   ®K
                     c1 °˙ c2 °˙ c3
                                                                     ®J
B:     b1 °˙ b2 °˙ b3
begin to intersect at node c1.
     */
    //@formatter:on
    /**
     * Notes:
     *
     * If the two linked lists have no intersection at all, return null. The
     * linked lists must retain their original structure after the function
     * returns. You may assume there are no cycles anywhere in the entire linked
     * structure. Your code should preferably run in O(n) time and use only O(1)
     * memory. Credits: Special thanks to @stellari for adding this problem and
     * creating all test cases.
     */

    int nodeCount = 0; // updated by the reverse process

    public ListNode reverseListWithNodeCount(ListNode head) {
        ListNode node = head;
        ListNode prev = null;
        int count = 0;
        while (node != null) {
            ListNode tmp = node.next;
            node.next = prev;
            prev = node;
            node = tmp;
            count++;
        }
        nodeCount = count;
        return prev;
    }

    // O(N) and constant space
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        // reverse A (x), count A+C (x+z) = p
        // reverse B (y), count A+B (x+y-1) = q
        // reverse C (z), count B+C (y+z) = r
        // x = (p+q-r+1)/2, the x-th node on A is intersection
        if (headA == null || headB == null) {
            return null;
        }

        int p = 0;
        int q = 0;
        int r = 0;

        ListNode headC = reverseListWithNodeCount(headA);
        p = nodeCount; // x+z
        ListNode headAA = reverseListWithNodeCount(headB);
        q = nodeCount; // x+y
        ListNode headBB = reverseListWithNodeCount(headC);
        r = nodeCount; // y+z
        if (headBB != headB) {// no intersection
            // restore B from headAA, then return
            reverseListWithNodeCount(headAA);
            return null;
        }

        int x = (p + q - r + 1) / 2;
        int i = 0;
        while (++i < x) {
            headA = headA.next;
        }
        return headA;
    }

    /**
     * Is Valid BST
     *
     * is valid binary search tree
     */

    // short version:
    public boolean isValidBST_short(TreeNode root) {
        // the key is - comapre with WHOLE SUB tree but not only its child
        return isValidBSTHelper(root, Integer.MIN_VALUE - 1L, Integer.MAX_VALUE + 1L);
    }

    // get min / max node from this subtree
    public boolean isValidBSTHelper(TreeNode root, long low, long high) {
        if (root == null) {
            return true;
        }
        if (low < root.val && root.val < high) {
            return isValidBSTHelper(root.left, low, root.val) && isValidBSTHelper(root.right, root.val, high);
        }
        return false;
    }

    // my thought - comparing current node with all it sub tree - expensive
    public boolean isValidBST(TreeNode root) {
        // the key is - comapre with WHOLE SUB tree but not only its child
        if (root == null) {
            return true;
        }

        if (root.left != null && root.right != null) {
            TreeNode minNode = getMinMaxNode(root.right, true); // right min
            TreeNode maxNode = getMinMaxNode(root.left, false); // left max
            if (maxNode.val >= root.val || minNode.val <= root.val) {
                return false; // max on left and min on the right
            } else {
                return isValidBST(root.left) && isValidBST(root.right);
            }
        } else if (root.left != null) {
            TreeNode maxNode = getMinMaxNode(root.left, false); // left max
            return (maxNode.val >= root.val) ? false : isValidBST(root.left);
        } else if (root.right != null) {
            TreeNode minNode = getMinMaxNode(root.right, true); // right min
            return (minNode.val <= root.val) ? false : isValidBST(root.right);
        } else { // leaf
            return true;
        }
    }

    // get min / max node from this subtree
    public TreeNode getMinMaxNode(TreeNode root, boolean getMin) {
        if (root == null) {
            return root;
        }
        TreeNode resultNode = new TreeNode(getMin ? Integer.MAX_VALUE : Integer.MIN_VALUE);
        Stack<TreeNode> st = new Stack<TreeNode>();
        st.push(root);
        // preorder
        while (!st.isEmpty()) {
            TreeNode n = st.pop();
            if (getMin) {
                resultNode = resultNode.val > n.val ? n : resultNode;
            } else { // max
                resultNode = resultNode.val < n.val ? n : resultNode;
            }
            if (root.right != null) {
                st.push(root.right);
            }
            if (root.left != null) {
                st.push(root.left);
            }
        }
        return resultNode;
    }

    // third: inorder travsersal

    /**
     * Find Peak Element
     *
     * A peak element is an element that is greater than its neighbors.
     *
     * Given an input array where num[i] °Ÿ num[i+1], find a peak element and
     * return its index.
     *
     * You may imagine that num[-1] = num[n] = -°ﬁ.
     *
     * For example, in array [1, 2, 3, 1], 3 is a peak element and your function
     * should return the index number 2.
     */
    public int findPeakElement(int[] num) {
        if (num == null || num.length == 0) {
            throw new IllegalArgumentException();
        }

        int index = -1;
        int max = Integer.MIN_VALUE; // pay attention to this init

        for (int i = 0; i < num.length; i++) {
            if (i - 1 >= 0 && num[i - 1] >= num[i]) {
                continue;
            }
            if (i + 1 < num.length && num[i + 1] >= num[i]) {
                continue;
            }
            // reset max, pay attention if num has one element as min_integer,
            // so we have to take >= instead of >
            if (num[i] >= max) {
                max = num[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * Convert sorted array to BST
     *
     * Given an array where elements are sorted in ascending order, convert it
     * to a height balanced BST.
     */
    public TreeNode sortedArrayToBST(int[] num) {
        if (num == null || num.length == 0) {
            return null;
        }

        return convertToBST(num, 0, num.length - 1);
    }

    private TreeNode convertToBST(int[] num, int low, int high) {
        // low and high is the index for the range in num
        if (low > high || low < 0 || high > num.length - 1) {
            return null;
        }

        int mid = (low + high) / 2; // mid is middle index between low and high
        TreeNode node = new TreeNode(num[mid]);
        node.left = convertToBST(num, low, mid - 1);
        node.right = convertToBST(num, mid + 1, high);

        return node;
    }

    /**
     * Convert Roman to integer
     *
     */
    //@formatter:off
    /**
    I   1
    V   5
    X   10
    L   50
    C   100
    D   500
    M   1,000

    the numeral I can be placed before V and X to make 4 units (IV) and 9 units (IX) respectively
    X can be placed before L and C to make 40 (XL) and 90 (XC) respectively
    C can be placed before D and M to make 400 (CD) and 900 (CM) according to the same pattern[5]
     */
    //@fomatter:on
    public int romanToInt(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int result = 0;
        int i = 0;
        while (i < s.length()) {
            char ch = s.charAt(i);
            if (ch == 'I') {
                if (i + 1 > s.length() - 1) {
                    result += 1;
                    break;
                }
                if (s.charAt(i + 1) == 'V') {
                    result += 4;
                    i += 2;
                    continue;
                } else if (s.charAt(i + 1) == 'X') {
                    result += 9;
                    i += 2;
                    continue;
                } else {
                    result += 1;
                    i++;
                    continue;
                }
            } else if (ch == 'X') {
                if (i + 1 > s.length() - 1) {
                    result += 10;
                    break;
                }
                if (s.charAt(i + 1) == 'L') {
                    result += 40;
                    i += 2;
                    continue;
                } else if (s.charAt(i + 1) == 'C') {
                    result += 90;
                    i += 2;
                    continue;
                } else {
                    result += 10;
                    i++;
                    continue;
                }
            } else if (ch == 'C') {
                if (i + 1 > s.length() - 1) {
                    result += 100;
                    break;
                }
                if (s.charAt(i + 1) == 'D') {
                    result += 400;
                    i += 2;
                    continue;
                } else if (s.charAt(i + 1) == 'M') {
                    result += 900;
                    i += 2;
                    continue;
                } else {
                    result += 100;
                    i++;
                    continue;
                }
            } else if (ch == 'V') {
                result += 5;
                i++;
                continue;
            } else if (ch == 'L') {
                result += 50;
                i++;
                continue;
            } else if (ch == 'D') {
                result += 500;
                i++;
                continue;
            } else if (ch == 'M') {
                result += 1000;
                i++;
                continue;
            }
        }
        return result;
    }

    //@formatter:on
    /**
     * convert number to Roman number
     *
     * Given an integer, convert it to a roman numeral. Input is guaranteed to
     * be within the range from 1 to 3999.
     */
    // shortest code
    public String intToRoman2(int num) {
        String M[] = { "", "M", "MM", "MMM" };
        String C[] = { "", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM" };
        String X[] = { "", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC" };
        String I[] = { "", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX" };
        StringBuilder sb = new StringBuilder();
        return sb.append(M[num / 1000]).append(C[(num % 1000) / 100]).append(X[(num % 100) / 10]).append(I[num % 10])
                .toString();
    }

    // naive
    public String intToRoman(int num) {
        // compute each digit, then convert to roman
        int[] d = new int[4]; // 1-3999
        d[0] = num / 1000;
        d[1] = (num - d[0] * 1000) / 100;
        d[2] = (num - d[0] * 1000 - d[1] * 100) / 10;
        d[3] = num - d[0] * 1000 - d[1] * 100 - d[2] * 10;

        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < 4; i++) {
            // fall through 1-3, 4, 5, 6-8, 9
            switch (d[i]) {
            case 1:
            case 2:
            case 3:
                for (int j = 0; j < d[i]; j++) {
                    switch (i) {
                    case 0:
                        sb.append("M");
                        break;
                    case 1:
                        sb.append("C");
                        break;
                    case 2:
                        sb.append("X");
                        break;
                    case 3:
                        sb.append("I");
                        break;
                    default:
                        break;
                    }
                }
                break;
            case 4:
                switch (i) {
                case 1:
                    sb.append("CD");
                    break;
                case 2:
                    sb.append("XL");
                    break;
                case 3:
                    sb.append("IV");
                    break;
                default:
                    break;
                }
                break;
            case 5:
                switch (i) {
                case 1:
                    sb.append("D");
                    break;
                case 2:
                    sb.append("L");
                    break;
                case 3:
                    sb.append("V");
                    break;
                default:
                    break;
                }
                break;
            case 6:
            case 7:
            case 8:

                switch (i) {
                case 1:
                    sb.append("D");
                    for (int j = 0; j < d[i] - 5; j++) {
                        sb.append("C");
                    }
                    break;
                case 2:
                    sb.append("L");
                    for (int j = 0; j < d[i] - 5; j++) {
                        sb.append("X");
                    }
                    break;
                case 3:
                    sb.append("V");
                    for (int j = 0; j < d[i] - 5; j++) {
                        sb.append("I");
                    }
                    break;
                default:
                    break;

                }

                break;
            case 9:
                switch (i) {
                case 1:
                    sb.append("CM");
                    break;
                case 2:
                    sb.append("XC");
                    break;
                case 3:
                    sb.append("IX");
                    break;
                default:
                    break;
                }
                break;
            default:
                break;
            }
        }
        return sb.toString();
    }

    /**
     * zigzag Conversion
     */
    //@formatter:off
    /*
    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this:
    (you may want to display this pattern in a fixed font for better legibility)

    P   A   H   N
     A P L S I I G
      Y   I   R
    And then read line by line: "PAHNAPLSIIGYIR"
    Write the code that will take a string and make this conversion given a number of rows:

    string convert(string text, int nRows);
    convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".
     */
    //@formatter:on
    public String convert(String s, int nRows) {
        if (s == null) {
            return "";
        }

        if (nRows == 1) {
            return s;
        }
        // for index i, its same level zigzag is: first: i+2*(row-1-i) and
        // second: i+2*(row-1)
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < nRows; i++) {
            int n = i;
            while (n < s.length()) {
                sb.append(s.charAt(n));
                int firstLevelElem = n + 2 * (nRows - i - 1);
                // i=0, it is duplicate first level element
                // similarly firstLevelElem > n to avoid dupliate with n
                if (i != 0 && firstLevelElem > n && firstLevelElem >= 0 && firstLevelElem < s.length()) {
                    sb.append(s.charAt(firstLevelElem));
                }
                n += 2 * (nRows - 1);
            }
        }
        return sb.toString();
    }

    /**
     * Remove Nth Node From End of List
     *
     * Given a linked list, remove the nth node from the end of list and return
     * its head.
     *
     * For example,
     *
     * Given linked list: 1->2->3->4->5, and n = 2.
     *
     * After removing the second node from the end, the linked list becomes
     * 1->2->3->5. Note: Given n will always be valid. Try to do this in one
     * pass.
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if (head == null) {
            return head;
        }

        ListNode end = head;
        ListNode front = head;
        int count = 0;
        // move front to keep distance to end as n-1
        while (count < n - 1 && front != null) {
            front = front.next;
            count++;
        }
        if (front == null) {
            // don't have n elements
            return head;
        }
        if (front.next == null) {
            // end is the element to be removed
            return end.next;
        }
        while (front.next.next != null) {
            front = front.next;
            end = end.next;
        }
        // now end stops at to-be-deleted element's parent
        end.next = end.next.next;
        return head;
    }

    /**
     * 3Sum - ThreeSum three sum
     *
     * Given an array S of n integers, are there elements a, b, c in S such that
     * a + b + c = 0? Find all unique triplets in the array which gives the sum
     * of zero.
     *
     * Note: Elements in a triplet (a,b,c) must be in non-descending order. (ie,
     * a °‹ b °‹ c) The solution set must not contain duplicate triplets. For
     * example, given array S = {-1 0 1 2 -1 -4},
     *
     * A solution set is: (-1, 0, 1) (-1, -1, 2)
     *
     */
    private boolean containsTriplet(List<List<Integer>> table, List<Integer> target) {
        for (List<Integer> list : table) {
            if (list.get(0) == target.get(0) && list.get(1) == target.get(1) && list.get(2) == target.get(2)) {
                return true;
            }
        }
        return false;
    }

    // use hashMap : TLE = Time Limit exception
    public List<List<Integer>> threeSum_hashmap_tle(int[] num) {
        // idea: use hashmap to store pair sum a,b, then compare to -c
        // sort the array first to make sure assending order
        if (num == null || num.length == 0) {
            return new ArrayList<List<Integer>>();
        }

        // Arrays.sort(num);
        Map<Integer, List<Pair>> twoSum = new HashMap<Integer, List<Pair>>();
        List<List<Integer>> result = new ArrayList<List<Integer>>();

        // build sum table
        for (int i = 0; i < num.length - 1; i++) {
            for (int j = i + 1; j < num.length; j++) {
                int sum = num[i] + num[j];
                List list = twoSum.get(sum);
                if (list == null) {
                    list = new ArrayList<Pair>();
                }
                list.add(new Pair(i, j));
                twoSum.put(sum, list);
            }
        }

        // compare
        for (int i = 0; i < num.length; i++) {
            // -num[i] in table
            if (twoSum.containsKey(-num[i])) {
                for (Pair p : twoSum.get(-num[i])) {
                    // avoid counting against itself
                    if (p.x == i || p.y == i) {
                        continue;
                    }

                    List<Integer> list = new ArrayList<Integer>();
                    // p.x < p.y
                    int[] arr = { num[i], num[p.x], num[p.y] };
                    Arrays.sort(arr);
                    list.add(arr[0]);
                    list.add(arr[1]);
                    list.add(arr[2]);
                    // avoid duplicate triplet
                    if (!containsTriplet(result, list)) {
                        result.add(list);
                    }
                }
                // now get rid of key
                twoSum.remove(-num[i]);
            }
        }
        return result;
    }

    // sort then squeeze like two sum
    public List<List<Integer>> threeSum(int[] num) {
        // idea: use hashmap to store pair sum a,b, then compare to -c
        // sort the array first to make sure assending order
        if (num == null || num.length == 0) {
            return new ArrayList<List<Integer>>();
        }

        Arrays.sort(num);
        List<List<Integer>> result = new ArrayList<List<Integer>>();

        // compare
        for (int i = 0; i < num.length - 2; i++) {
            for (int j = i + 1, k = num.length - 1; j < k;) {
                if (num[j] + num[k] == -num[i]) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(num[i]);
                    list.add(num[j]);
                    list.add(num[k]);
                    j++;
                    k--;
                    // avoid duplicate triplet
                    if (!containsTriplet(result, list)) {
                        result.add(list);
                    }
                } else if (num[j] + num[k] < -num[i]) {
                    j++;
                } else {
                    k--;
                }
            }
        }
        return result;
    }

    // sum three numbers to a target
    public List<List<Integer>> threeSum_target(int[] num, int target) {
        if (num == null || num.length == 0) {
            return new ArrayList<List<Integer>>();
        }
        // algo - sort the array, then fix one element, and check if rest pairs
        // sum to (target-num[i])
        Arrays.sort(num);
        List<List<Integer>> result = new ArrayList<List<Integer>>();

        // compare target-num[i] against num[j]+num[k] , j starts from i+1, k
        // starts from end of array
        for (int i = 0; i < num.length - 2; i++) {
            for (int j = i + 1, k = num.length - 1; j < k;) {
                if (num[j] + num[k] == target - num[i]) {
                    List<Integer> list = new ArrayList<Integer>();
                    list.add(num[i]);
                    list.add(num[j]);
                    list.add(num[k]);
                    j++;
                    k--;
                    // avoid duplicate triplet
                    if (!containsTriplet(result, list)) {
                        result.add(list);
                    }
                } else if (num[j] + num[k] < target - num[i]) {
                    j++;
                } else {
                    k--;
                }
            }
        }
        return result;
    }

    /**
     * 3Sum Closest
     *
     * Given an array S of n integers, find three integers in S such that the
     * sum is closest to a given number, target. Return the sum of the three
     * integers. You may assume that each input would have exactly one solution.
     *
     * For example, given array S = {-1 2 1 -4}, and target = 1.
     *
     * The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
     */
    public int threeSumClosest(int[] num, int target) {
        // equivalent to find min(abs(d-(a+b+c))) or min(abs( (d-a)-(b+c)))
        int close = Integer.MAX_VALUE;
        int ret = 0;
        Arrays.sort(num);
        for (int i = 0; i < num.length - 2; i++) {
            int a = num[i];
            for (int j = i + 1, k = num.length - 1; j < k;) {
                int b = num[j];
                int c = num[k];
                if (b + c == target - a) {
                    return target;
                } else if (b + c < target - a) {
                    j++;
                } else {
                    k--;
                }
                if (Math.abs(target - a - b - c) < close) {
                    close = Math.abs(target - a - b - c);
                    ret = a + b + c;
                }
            }
        }
        return ret;
    }

    /**
     * Binary Tree Zigzag Level Order Traversal
     *
     * Given a binary tree, return the zigzag level order traversal of its
     * nodes' values. (ie, from left to right, then right to left for the next
     * level and alternate between).
     *
     * For example: Given binary tree {3,9,20,#,#,15,7},
     */
    //@formatter:off
    /*
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
     */
    //@formatter:on
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        // algo - level order traversal with queue - odd leve and stack - even
        // level
        if (root == null) {
            return new ArrayList<List<Integer>>();
        }

        // node queue
        Queue<TreeNode> worker = new LinkedList<TreeNode>();
        // even level
        Stack<TreeNode> st = new Stack<TreeNode>();
        // odd level
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        boolean odd = true;
        int curr = 1;
        int next = 0;
        worker.add(root);
        while (!worker.isEmpty()) {
            TreeNode node = worker.poll();
            if (odd) {
                q.offer(node);
            } else {
                st.push(node);
            }
            curr--;
            if (node.left != null) {
                worker.add(node.left);
                next++;
            }
            if (node.right != null) {
                worker.add(node.right);
                next++;
            }
            if (curr == 0) {
                curr = next;
                next = 0;
                List<Integer> list = new ArrayList<Integer>();
                if (odd) {
                    while (!q.isEmpty()) {
                        list.add(q.poll().val);
                    }
                    odd = false;
                } else {
                    while (!st.isEmpty()) {
                        list.add(st.pop().val);
                    }
                    odd = true;
                }
                result.add(list);
            }
        }
        return result;
    }

    /**
     * Combinations
     *
     * Given two integers n and k, return all possible combinations of k numbers
     * out of 1 ... n.
     *
     * For example, If n = 4 and k = 2, a solution is:
     *
     * [ [2,4], [3,4], [2,3], [1,2], [1,3], [1,4], ]
     */
    public List<List<Integer>> combine(int n, int k) {
        // algo - recursive f(k) = f(k-1)U (n,1)
        if (k == 0) {
            List<Integer> list = new ArrayList<Integer>();
            List<List<Integer>> ret = new ArrayList<List<Integer>>();
            ret.add(list);
            return ret;
        }
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        // add one more element to get (n,k)

        for (List<Integer> list : combine(n, k - 1)) {
            for (int i = 1; i <= n; i++) {
                if (!list.contains(i)) {
                    // must create a new list
                    List<Integer> copy = new ArrayList<Integer>();
                    copy.addAll(list);
                    copy.add(i);
                    result.add(copy);
                }
            }
        }
        return result;
    }

    /**
     * Compare Version Numbers
     *
     * Compare two version numbers version1 and version1. If version1 > version2
     * return 1, if version1 < version2 return -1, otherwise return 0.
     *
     * You may assume that the version strings are non-empty and contain only
     * digits and the . character. The . character does not represent a decimal
     * point and is used to separate number sequences. For instance, 2.5 is not
     * "two and a half" or "half way to version three", it is the fifth
     * second-level revision of the second first-level revision.
     *
     * Here is an example of version numbers ordering:
     *
     * 0.1 < 1.1 < 1.2 < 13.37
     */
    public int compareVersion(String version1, String version2) {
        // split by '.' then compare
        String[] A = version1.split("\\.");
        String[] B = version2.split("\\.");

        int len = Math.max(A.length, B.length);
        int i = 0;
        while (i < len) {
            // pad 0 if out of range
            int m = i < A.length ? Integer.parseInt(A[i]) : 0;
            int n = i < B.length ? Integer.parseInt(B[i]) : 0;
            if (m < n) {
                return -1;
            } else if (m > n) {
                return 1;
            }
            i++;
        }
        return 0;
    }

    /**
     * Excel Column title conversion
     *
     * Excel Sheet Column Title Total Accepted: 1028 Total Submissions: 5888 My
     * Submissions Question Solution Given a positive integer, return its
     * corresponding column title as appear in an Excel sheet.
     *
     * For example:
     */
    //@formatter:off
    /*
     *     1 -> A
           2 -> B
           3 -> C
           ...
           26 -> Z
           27 -> AA
           28 -> AB
     */
    //@formatter:on
    public String convertToTitle(int n) {
        if (n < 1) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        // think about reversed, ABCD = A*26^3 + B*26^2 + C*26 + D
        // so we compute D first, then C, till A
        while (n > 0) {
            char ch = (char) ('A' + (n - 1) % 26);
            sb.insert(0, ch);
            n -= ch - 'A' + 1;
            n /= 26; // get next least significant digit
        }
        return sb.toString();
    }

    /**
     * Excel Sheet Column Number
     *
     * this is the reversed conversion of excel colum
     *
     * Related to question Excel Sheet Column Title
     *
     * Given a column title as appear in an Excel sheet, return its
     * corresponding column number.
     *
     * For example:
     *
     * A -> 1 B -> 2 C -> 3 ... Z -> 26 AA -> 27 AB -> 28
     */
    public int titleToNumber(String s) {
        // ABC = A*26^2+B*26^1+C
        if (s == null) {
            return 0;
        }

        int sum = 0;
        for (int i = 0; i < s.length(); i++) {
            sum = sum * 26 + s.charAt(i) - 'A' + 1;
        }
        return sum;
    }

    /**
     * n-Parentheses - print n valid parenthese combo
     */
    private class ParenElem {
        // host partial parenthese string
        int left;
        int right;
        String paren;

        public ParenElem(String p, int l, int r) {
            paren = String.valueOf(p);
            left = l;
            right = r;
        }
    }

    public List<String> generateParenthesis(int n) {
        if (n <= 0) {
            return new ArrayList<String>(); // empty
        }
        List<String> result = new ArrayList<String>();
        Queue<ParenElem> q = new LinkedList<ParenElem>();
        q.offer(new ParenElem("", 0, 0)); // start from empty element
        while (!q.isEmpty()) {
            ParenElem e = q.poll();
            if (e.left == n && e.right == n) {
                result.add(e.paren);
                continue;
            }
            if (e.left < n) {
                q.add(new ParenElem(e.paren + "(", e.left + 1, e.right));
            }
            if (e.left > e.right) {
                q.add(new ParenElem(e.paren + ")", e.left, e.right + 1));
            }
        }
        return result;
    }

    /**
     * Divide two number with /
     *
     * Divide two integers without using multiplication, division and mod
     * operator.
     *
     * If it is overflow, return MAX_INT.
     */
    public int divide(int dividend, int divisor) {
        // algo: double divisor every time to achive O(logN)
        if (divisor == 0) {
            throw new IllegalArgumentException();
        }
        if (dividend == 0) {
            return 0;
        }

        // overflow
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE; // absolute value of max is less than min
            // by 1
        }

        boolean neg = false;
        if (dividend < 0 && divisor > 0) {
            neg = true;
        } else if (dividend > 0 && divisor < 0) {
            neg = true;
        }
        // convert to long to avoid overflow
        long u = dividend;
        long div = divisor;
        long up = Math.abs(u);
        long down = Math.abs(div); // min (-2147483648) take abs overflow, it
        // remains unchanged -2147483648
        int count = 0;
        while (down <= up) {
            int m = 1; // power 2 count
            long d = down;
            // double divisor if less than dividend
            while ((d << 1) < up) {
                d <<= 1;
                m <<= 1;
            }
            count += m;
            up -= d;
        }
        return neg ? -count : count;
    }

    public List<List<Integer>> subsetsWithDup(int[] num) {
        return subsetsWithDup(num, false); // avoid multiple sort
    }

    private List<List<Integer>> subsetsWithDup(int[] num, boolean sorted) {
        // algo: sort then count adding 1,2,3.. dups with other
        if (num == null || num.length == 0) {
            List<Integer> l = new ArrayList<Integer>();
            List<List<Integer>> list = new ArrayList<List<Integer>>();
            list.add(l);
            return list; // empty
        }
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        // to avoid sort every time, move recursive to seperate function
        if (!sorted) {
            Arrays.sort(num);
        }

        List<List<Integer>> dupList = new ArrayList<List<Integer>>();
        for (int i = 0; i < num.length; i++) {
            if (num[i] != num[0]) {
                break;
            }

            List<Integer> dup = new ArrayList<Integer>();
            for (int j = 0; j <= i; j++) {
                dup.add(num[0]);
            }
            dupList.add(dup);
        }

        // build rest array (minus dup), dupList.size() records how many dups
        int dupCount = dupList.size();
        int count = num.length - dupCount;
        int[] restArr = new int[count];
        for (int i = 0; i < count; i++) {
            restArr[i] = num[dupCount + i];
        }

        // get F(i-1) recursively on the last half
        List<List<Integer>> secondCombo = subsetsWithDup(restArr, true);
        // Step1 - add F(i-1) partial result first
        result.addAll(secondCombo);

        // Step2 - combine duplist with secondCombo
        for (List<Integer> first : dupList) {
            for (List<Integer> second : secondCombo) {
                List<Integer> ll = new ArrayList<Integer>();
                ll.addAll(first);
                ll.addAll(second);
                result.add(ll);
            }
        }

        return result;
    }

    /**
     * Rotate NxN matrix by 90 degree clockwise
     *
     * Note - This question is same as - visit 2D matrix clockwise and from out
     * to inside
     */
    // best solution
    public void rotate(int[][] matrix) {
        // generally, (i,j) -> (j, n-i-1)
        // so, it forms a transform circle:
        // (i,j) -> (j,n-1-i) -> (n-1-i, n-1-j) -> (n-1-j, i) -> (i, j)

        int n = matrix.length;
        for (int i = 0; i < n / 2; i++) {
            for (int j = i; j < n - i - 1; j++) {
                int a = matrix[i][j];
                int b = matrix[j][n - 1 - i];
                int c = matrix[n - 1 - i][n - 1 - j];
                int d = matrix[n - 1 - j][i];

                matrix[j][n - 1 - i] = a;
                matrix[n - 1 - i][n - 1 - j] = b;
                matrix[n - 1 - j][i] = c;
                matrix[i][j] = d;
            }
        }
    }

    // sub-optimal
    public void rotate_dummy_use_visited_table(int[][] matrix) {
        // generally, (i,j) -> (j, n-i-1)
        // so, it forms a transform circle:
        // (i,j) -> (j,n-1-i) -> (n-1-i, n-1-j) -> (n-1-j, i) -> (i, j)

        // mark rotation on (i,j)
        // think how to remove mark table r[][]
        int[][] r = new int[matrix.length][matrix.length];
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (r[i][j] == 1) {
                    continue; // alrady rotated
                }

                int a = matrix[i][j];
                int b = matrix[j][n - 1 - i];
                int c = matrix[n - 1 - i][n - 1 - j];
                int d = matrix[n - 1 - j][i];

                matrix[j][n - 1 - i] = a;
                matrix[n - 1 - i][n - 1 - j] = b;
                matrix[n - 1 - j][i] = c;
                matrix[i][j] = d;

                r[i][j] = 1;
                r[j][n - 1 - i] = 1;
                r[n - 1 - i][n - 1 - j] = 1;
                r[n - 1 - j][i] = 1;
            }
        }
    }

    /**
     * Search Insert Position - binary search variance
     *
     * Given a sorted array and a target value, return the index if the target
     * is found. If not, return the index where it would be if it were inserted
     * in order.
     *
     * You may assume no duplicates in the array.
     *
     * Here are few examples. [1,3,5,6], 5 °˙ 2 [1,3,5,6], 2 °˙ 1 [1,3,5,6], 7 °˙ 4
     * [1,3,5,6], 0 °˙ 0
     */
    public int searchInsert(int[] A, int target) {
        if (A == null || A.length == 0) {
            return 0;
        }

        int low = 0;
        int high = A.length - 1;

        while (low <= high) {
            int mid = (low + high) >>> 1; // handle overflow
            if (A[mid] == target) {
                return mid;
            } else if (A[mid] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        // not found, check trending
        if (low >= A.length) {
            return low;
        } else {
            return A[low] > target ? low : low + 1;
        }
    }

    /**
     * Majority Element
     *
     * Given an array of size n, find the majority element. The majority element
     * is the element that appears more than n/2 times.
     *
     * You may assume that the array is non-empty and the majority element
     * always exist in the array.
     */
    public int majorityElement(int[] num) {
        // if majority element exists, then it must be the only larget one
        // otherwise use hashmap count
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int element : num) {
            if (map.containsKey(element)) {
                int k = map.get(element);
                map.put(element, ++k);
                if (k > (num.length / 2)) {
                    return element;
                }
            } else {
                map.put(element, 1);
            }
        }
        // if so far we cannot decide highest frequent, then it must be the last
        // element
        return num[num.length - 1];
    }

    /**
     * Search a 2D Matrix
     *
     * Write an efficient algorithm that searches for a value in an m x n
     * matrix. This matrix has the following properties:
     *
     * Integers in each row are sorted from left to right. The first integer of
     * each row is greater than the last integer of the previous row. For
     * example,
     *
     * Consider the following matrix:
     */
    //@formatter:off
    /*
    [
     [1,   3,  5,  7],
     [10, 11, 16, 20],
     [23, 30, 34, 50]
   ]
     */
    //@formatter:on
    public boolean searchMatrix(int[][] matrix, int target) {
        // distance between (p, q) and (m, n) is d = (m-p-1)*N + n +(N-q),
        // where N is array size. so mid is (p, q) + d/2, converted to 2d matrix
        // it is: (p,q) to 1-D array is p*N + q. so mid = p*N+q + d/2, convert
        // to 2d , it is (mid/N, mid%N);
        int rows = matrix.length;
        int N = matrix[0].length;
        // simulate 1d array search
        int low = 0;
        int high = rows * N - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (matrix[mid / N][mid % N] == target) {
                return true;
            } else if (matrix[mid / N][mid % N] < target) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return false;
    }

    /**
     * Binary tree delete branch under threshhold sum
     *
     * this is not in leetcode, but in an interview
     *
     * Given a binary tree and a threshhold, remove all branches doesn't sum to
     * the threashhild from root to leaf.
     *
     */
    // sample:
    // @formatter:off
    /*
     *  Given tree and threshhold 20 below
     *          5
     *         / \
     *        5   15
     *       / \
     *      1   10
     *     /
     *    1
     * the tree should look like this after trim under threshhold 20 sum, all root to leaf sum pass threshhold
     *          5
     *         / \
     *        5   15
     *        \
     *         10
     */
    // @formatter:on
    // approach 1 - use post order recursive, not good due to recursion
    public TreeNode delBinaryTreeThreshHold(TreeNode root, int threshHold) {
        // algo - since we can only make decision after all leaves are visited
        // this is intrinsically post order traversal, combined with each
        // subtracted threshhold down to leaf
        if (root == null || isLeaf(root) && root.val < threshHold) {
            return null;
        } else if (root.val >= threshHold) { // all positive data
            return root;
        }
        // post order
        TreeNode left = delBinaryTreeThreshHold(root.left, threshHold - root.val);
        TreeNode right = delBinaryTreeThreshHold(root.right, threshHold - root.val);

        // if any child cannot satisfy the threshhold, then remove it
        if (left == null) {
            root.left = null;
        }
        if (right == null) {
            root.right = null;
        }
        // root.val < threshhold or it has returned in begining
        return (left == null && right == null) ? null : root;
    }

    // approach 2 - post order iterative, much better solution
    public TreeNode delBinaryTreeThreshHold_PostOrder_Iterative(TreeNode root, int threshHold) {
        if (root == null || isLeaf(root) && root.val < threshHold) {
            return null;
        } else if (root.val >= threshHold) { // all positive data
            return root;
        }
        Stack<TreeNode> st = new Stack<TreeNode>();
        // build the track map for each node with its corresponding threshhold
        // or use <TreeNode, Long>, so we could use special mark as
        // Integer.MIN_VALUE-1L to mark invalid node
        Map<TreeNode, Integer> track = new HashMap<TreeNode, Integer>();
        TreeNode node = root;
        TreeNode visited = null;
        while (!st.isEmpty() || node != null) {
            if (node != null) {
                st.push(node);
                // update threshHold value for children
                track.put(node, threshHold);
                threshHold -= node.val;
                node = node.left;
            } else { // either puth right or visit current
                TreeNode peek = st.peek();
                // push right side
                if (peek.right != null && peek.right != visited) {
                    node = peek.right; // visit right child
                } else { // visit current
                    peek = st.pop();
                    visited = peek;
                    // reset threshhold
                    threshHold = track.get(peek);
                    // now check children and set current node
                    if (peek.left != null && (track.get(peek.left) == Integer.MIN_VALUE)) {
                        peek.left = null;
                    }
                    if (peek.right != null && (track.get(peek.right) == Integer.MIN_VALUE)) {
                        peek.right = null;
                    }
                    if (isLeaf(peek) && peek.val < track.get(peek)) {
                        track.put(peek, Integer.MIN_VALUE);
                    }
                }
            }
        }
        return root;
    }

    private boolean isLeaf(TreeNode n) {
        return n.left == null && n.right == null;
    }

    /**
     * Distinct Subsequences - ie, count how many ways to convert S to T by
     * deleting only
     *
     *
     *
     * Given a string S and a string T, count the number of distinct
     * subsequences of T in S.
     *
     * A subsequence of a string is a new string which is formed from the
     * original string by deleting some (can be none) of the characters without
     * disturbing the relative positions of the remaining characters. (ie, "ACE"
     * is a subsequence of "ABCDE" while "AEC" is not).
     *
     * Here is an example: S = "rabbbit", T = "rabbit"
     *
     * Return 3.
     */
    public int numDistinct(String S, String T) {
        // remove char from S to get T, and count how many position deletion
        // combos (of S)
        // Algo - use dp, let t[i][j] be the ways to convert S[0..i] to T[0...j]
        // then t[i][j] = t[i-1][j] + S[i]==T[j] : t[i-1][j-1] : 0;
        if (S == null || T == null) {
            return 0;
        } else if (T.length() == 0) {
            return 1;
        } else if (S.length() == 0) {
            return T.length() == 0 ? 1 : 0;
        }

        int[][] t = new int[S.length()][T.length()];
        // init t[i][0]
        int count = 0;
        for (int i = 0; i < S.length(); i++) {
            count = (S.charAt(i) == T.charAt(0)) ? count + 1 : count;
            t[i][0] = count; // how many repeated chars = how may ways to delete
            // char
        }
        for (int i = 1; i < S.length(); i++) {
            for (int j = 1; j < T.length(); j++) {
                t[i][j] = t[i - 1][j] + ((S.charAt(i) == T.charAt(j)) ? t[i - 1][j - 1] : 0);
            }
        }
        return t[S.length() - 1][T.length() - 1];
    }

    /**
     * Rotate Linked List by n steps
     *
     */
    public ListNode rotateRight(ListNode head, int n) {
        // algo - count rounds = (n % list length), then roate list by rounds
        // times
        if (head == null) {
            return head;
        }

        // count length of list
        int len = 0;
        ListNode front = head;
        while (front != null) {
            len++;
            front = front.next;
        }
        // rotate count, rotates < len
        int rotates = n % len;
        if (rotates == 0) {
            return head;
        }
        ListNode end = head;
        front = head;
        // move front ahead by rotates steps
        int dist = 0;
        while (dist++ < rotates) {
            front = front.next;
        }
        // move front and end together
        while (front.next != null) {
            front = front.next;
            end = end.next;
        }
        // now end.next is new head
        ListNode endNext = end.next;
        end.next = null;
        front.next = head;

        return endNext;
    }

    /**
     * Reverse Nodes in k-Group
     *
     * Given a linked list, reverse the nodes of a linked list k at a time and
     * return its modified list.
     *
     * If the number of nodes is not a multiple of k then left-out nodes in the
     * end should remain as it is.
     *
     * You may not alter the values in the nodes, only nodes itself may be
     * changed.
     *
     * Only constant memory is allowed.
     *
     * For example, Given this linked list: 1->2->3->4->5
     *
     * For k = 2, you should return: 2->1->4->3->5
     *
     * For k = 3, you should return: 3->2->1->4->5
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null || k < 2) {
            return head;
        }

        // we need three variables, previous end node, current (start, end)
        // after current reverse, linke prevEnd to current head
        ListNode newHead = head;
        ListNode start = head;
        ListNode end = head;
        ListNode prevEnd = null;
        boolean firstRun = true;

        // scan list and reverse every k group
        while (end != null) {
            int count = 0;
            while (++count < k && end != null) {
                end = end.next;
            }
            // if contains less than k nodes, stop
            if (count < k || end == null) {
                break;
            }
            // save current end->next, then reverse
            ListNode next = end.next;
            // now reverse (start, end)
            ListNode currHead = reverseNodes(start, end);
            if (firstRun) {
                newHead = currHead;
                firstRun = false;
            } else {
                // connect previous list with current reversed head
                prevEnd.next = currHead;
            }
            prevEnd = start;
            prevEnd.next = next;

            start = next;
            end = next;
        }
        return newHead;
    }

    // reverse the node from start to end as in a linked list
    // return :
    // end - the head of new list
    // start - points to null after processing
    private ListNode reverseNodes(ListNode start, ListNode end) {
        ListNode prev = null;
        ListNode node = start;
        while (prev != end) {
            ListNode next = node.next;
            node.next = prev;
            prev = node;
            node = next;
        }
        // when it breaks prev is end
        return prev;
    }

    /**
     *
     */
    public int trailingZeroes(int n) {
        // 10 is formed by 2x5, so only need to count number of 2 and 5's in
        // pair.
        // since 5's is less than 2's, so only need to count 5's
        // question becomes - count how many 5's as factor <= n
        // dvivide n into intervals as [5, 5^2, 5^3...5^m) where m=log(5,n),
        // then count numbers in each interval [5^k, 5^(k+1)) = 4*k for n only
        if (n < 0) {
            return 0;
        }
        int k = (int) (Math.log(n) / Math.log(5)); // 5^k=floor(n)
        int sum = 0;
        for (int i = 1; i <= k; i++) {
            sum += 4 * i;
        }
        // add left over [5^k, n]
        sum += n / Math.pow(5, k);
        return sum;
    }

    /**
     * Binary Search Tree Iterator
     *
     * Implement an iterator over a binary search tree (BST). Your iterator will
     * be initialized with the root node of a BST.
     *
     * Calling next() will return the next smallest number in the BST.
     *
     * Note: next() and hasNext() should run in average O(1) time and uses O(h)
     * memory, where h is the height of the tree.
     */
    public class BSTIterator {
        // algo - since this is BST, we use in order traversal
        TreeNode current;
        Stack<TreeNode> stack;

        public BSTIterator(TreeNode root) {
            current = root;
            stack = new Stack<TreeNode>();
        }

        /** @return whether we have a next smallest number */
        public boolean hasNext() {
            return !stack.isEmpty() || current != null;
        }

        /** @return the next smallest number */
        public int next() {
            while (current != null) {
                stack.push(current);
                current = current.left;
            }
            // visit current
            TreeNode returnNode = stack.pop();
            current = returnNode.right;

            return returnNode.val;
        }
    }

    /**
     * Next Permutation
     *
     * Implement next permutation, which rearranges numbers into the
     * lexicographically next greater permutation of numbers.
     *
     * If such arrangement is not possible, it must rearrange it as the lowest
     * possible order (ie, sorted in ascending order).
     *
     * The replacement must be in-place, do not allocate extra memory.
     *
     * Here are some examples. Inputs are in the left-hand column and its
     * corresponding outputs are in the right-hand column. 1,2,3 °˙ 1,3,2 3,2,1 °˙
     * 1,2,3 1,1,5 °˙ 1,5,1
     */
    public void nextPermutation(int[] num) {
        if (num == null || num.length < 2) {
            return;
        }

        // algo - scan from end to start, swap if [i-1] < [i]
        // if hit [0], then reverse whole string (sorting)
        for (int i = num.length - 2; i >= 0; i--) {
            if (num[i] < num[i + 1]) {
                // sort i+1...n-1, then insert i into i+1..n-1
                reverse(num, i + 1, num.length - 1);
                int index = getIndexOfFirstLargeElem(num, i + 1, i);
                swap(num, index, i);
                return;
            }
        }
        // if we reached here then the array is either all equal or in
        // descending order
        if (num[0] != num[num.length - 1]) {
            reverse(num, 0, num.length - 1);
        }
    }

    // find first element index larget than target
    private int getIndexOfFirstLargeElem(int[] num, int start, int target) {
        // binary search
        int l = start;
        int r = num.length - 1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            if (num[mid] <= num[target]) {
                if (mid + 1 >= num.length) {
                    return -1; // fake as no bigger element found
                } else if (num[mid + 1] > num[target]) {
                    return mid + 1;
                } else {
                    l = mid + 1;
                }
            } else { // if (num[mid] > num[target]) {
                if (mid - 1 < start || num[mid - 1] <= num[target]) {
                    return mid;
                } else {
                    r = mid - 1;
                }
            }
        }
        return -1;
    }

    private void swap(int[] num, int i, int j) {
        int tmp = num[i];
        num[i] = num[j];
        num[j] = tmp;
    }

    private void reverse(int[] num, int start, int end) {
        for (int i = start; i < (start + end + 1) / 2; i++) {
            int tmp = num[i];
            num[i] = num[start + end - i];
            num[start + end - i] = tmp;
        }
    }

    /**
     * Permutation Sequence
     *
     * The set [1,2,3,°≠,n] contains a total of n! unique permutations.
     *
     * By listing and labeling all of the permutations in order, We get the
     * following sequence (ie, for n = 3):
     *
     * "123" "132" "213" "231" "312" "321" Given n and k, return the kth
     * permutation sequence.
     *
     * Note: Given n will be between 1 and 9 inclusive.
     */
    List<Integer> nums = new ArrayList<Integer>();
    StringBuilder ret = new StringBuilder();

    public String getPermutation(int n, int k) {
        // validate n, k
        if (n <= 0) {
            return "";
        }
        // validate k
        if (k > fact(n)) {
            return "";
        }
        for (int i = 1; i <= n; i++) {
            nums.add(i);
        }

        getPermutation_Recursive(n, k);
        return ret.toString();
    }

    private void getPermutation_Recursive(int n, int k) {
        // the perm is distributed into n groups, start with 1,2...n, each has
        // (n-1)!elements
        // we calculate the position of k mapped to group, then get the
        // correspoinding number out of nums list
        if (n == 1) {
            if (k != 1) {
                // sth wrong, clear ret
                ret.setLength(0);
            } else {
                ret.append(nums.get(0));
            }
            return;
        }
        // each group count (n-1)!
        int groupCount = fact(n - 1);

        int index = (k - 1) / groupCount; // index to nums list
        int lead = nums.remove(index);
        ret.append(lead);
        int remainderK = k - index * groupCount;
        getPermutation_Recursive(n - 1, remainderK);
    }

    private int fact(int n) {
        int fact = 1;
        for (int i = 1; i <= n; i++) {
            fact *= i;
        }
        return fact;
    }

    // slow impl - iterate through next permutation
    public String getPermutation_slow_iteration_impl(int n, int k) {
        // validate n, k
        // get factorial of n
        int fact = 1;
        int[] num = new int[n];
        for (int i = 1; i <= n; i++) {
            fact *= i;
            num[i - 1] = i;
        }

        for (int i = 0; i < fact; i++) {
            if (i == k - 1) {
                return arrayToString(num);
            }
            getNextPermutation(num);
        }
        return ""; // k if out of range
    }

    private String arrayToString(int[] num) {
        StringBuilder sb = new StringBuilder();
        for (int element : num) {
            sb.append(element);
        }
        return sb.toString();
    }

    private void getNextPermutation(int[] num) {
        // 1. scan from right to left,
        // 2. stop at first decreasing number, say d, or index 0
        // 3. sort i+1 till length
        // 4. swap d with first larger than d element in paritail sort
        // eg. 23541, scan and stop on "3", sort 541-> 23145, swap 3 with
        // 4->24135
        // "542"->13245
        if (num == null || num.length == 1) {
            return;
        }

        int i = num.length - 2;
        for (; i >= 0; i--) {
            if (num[i] < num[i + 1]) {
                break;
            }
        }
        if (i == -1 && num[0] != num[1]) {
            // non increasing series
            reverse(num, 0, num.length - 1);
            return;
        }

        // sort i+1 till length-1
        reverse(num, i + 1, num.length - 1);
        int d = getIndexOfFirstLargeElem(num, i + 1, i);
        swap(num, i, d);
    }

    /**
     * Search for a Range
     *
     * Given a sorted array of integers, find the starting and ending position
     * of a given target value.
     *
     * Your algorithm's runtime complexity must be in the order of O(log n).
     *
     * If the target is not found in the array, return [-1, -1].
     *
     * For example, Given [5, 7, 7, 8, 8, 10] and target value 8, return [3, 4].
     */
    public int[] searchRange(int[] A, int target) {
        // algo - use binary search, find the first index of target, then
        // find low boundary on the left, and high boundary on the right
        int[] ret = new int[2];
        ret[0] = -1;
        ret[1] = -1;

        if (A == null || A.length == 0) {
            return ret;
        }
        int low = 0;
        int high = A.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (A[mid] < target) {
                low = mid + 1;
            } else if (A[mid] > target) {
                high = mid - 1;
            } else { // found
                // result was found
                ret[0] = findFirstMatch(A, 0, mid, target);
                ret[1] = findLastMatch(A, mid, A.length - 1, target);
                return ret;
            }
        }
        return ret;
    }

    // find the first index equals to target
    private int findFirstMatch(int[] A, int start, int end, int target) {
        int low = start;
        int high = end;
        // since A is sorted and A[end] == target, so all element should be less
        // than target
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (A[mid] < target) {
                if (mid + 1 <= end && A[mid + 1] == target) {
                    return mid + 1;
                } else {
                    low = mid + 1;
                }
            } else { // found
                if (mid == start) {
                    return mid;
                } else {
                    if (A[mid - 1] < target) {
                        return mid;
                    } else {
                        high = mid - 1;
                    }
                }
            }
        }
        return -1;
    }

    // find the last index that equals to target
    private int findLastMatch(int[] A, int start, int end, int target) {
        int low = start;
        int high = end;
        // since A is sorted, and A[start] == target, all element should be >=
        // target
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (A[mid] > target) {
                if (mid - 1 >= start && A[mid - 1] == target) {
                    return mid - 1;
                } else {
                    high = mid - 1;
                }
            } else { // found
                if (mid == end) {
                    return mid;
                } else {
                    if (A[mid + 1] > target) {
                        return mid;
                    } else {
                        low = mid + 1;
                    }
                }
            }
        }
        return -1;
    }

    /**
     * Search in Rotated Sorted Array
     *
     * Suppose a sorted array is rotated at some pivot unknown to you
     * beforehand.
     *
     * (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     *
     * You are given a target value to search. If found in the array return its
     * index, otherwise return -1.
     *
     * You may assume no duplicate exists in the array.
     */
    public int search(int[] A, int target) {
        if (A == null || A.length == 0) {
            return -1;
        }

        int low = 0;
        int high = A.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (A[mid] == target) {
                return mid;
            } else if (A[mid] < target) {
                if (A[low] <= A[mid]) {
                    // due to rotation, [start, mid] is asending interval
                    // eg. {2,3,4,5,1} search 5
                    low = mid + 1;
                } else {
                    // since A[start] > A[mid], there exists lasgest element
                    // between[start,mid] or [mid,end]
                    // eg. {4,5,1,2,3} search 5
                    if (A[high] >= target) {
                        low = mid + 1;
                    } else {
                        high = mid - 1;
                    }
                }
            } else { // A[mid] > target
                if (A[high] <= A[mid]) { // mid must be rotated
                    // {3,4,5,1,2} search 3
                    if (A[low] <= target) {
                        high = mid - 1;
                    } else { // {3,4,5,1,2} search 2
                        low = mid + 1;
                    }
                } else { // [mid, high] is asending
                    high = mid - 1;
                }
            }
        }
        return -1;
    }

    /**
     * Find Minimum in Rotated Sorted Array
     *
     * Suppose a sorted array is rotated at some pivot unknown to you
     * beforehand.
     *
     * (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     *
     * Find the minimum element.
     *
     * You may assume no duplicate exists in the array.
     */
    public int findMin(int[] num) {
        // find i, st,
        // 1. num[i-1] > num[i] && num[i] < num[i+1] or
        // 2. i-1 is out of scope, then num[i] < num[i+1] && num[i] < num[len]
        // 3. if i+1 if out of scope, then num[i-1] > num[i] && num[i] < num[0]
        // unitify 3 cases together, we can define (treat it as circular array)
        // prev = (i-1)+len%len, next = (i+1) % len, i=0,1...len-1
        // so we look for i, st A[prev] > A[i] && A[next] > A[i]
        if (num == null || num.length == 0) {
            return -1;
        }
        if (num.length == 1) {
            return num[0];
        }

        int len = num.length;
        int low = 0;
        int high = len - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            int prev = (mid - 1 + len) % len;
            int next = (mid + 1) % len;
            if (num[mid] < num[prev] && num[mid] < num[next]) {
                return num[mid];
            } else if (num[mid] > num[next]) {
                // when A[mid] > A[next], the mid is the rotation boundary,
                // hence largest element
                return num[next];
            } else { // A[mid] > A[prev] && A[mid] < A[next]
                // check A[mid] is rotated
                if (num[mid] > num[len - 1]) {
                    // rotated, search right
                    low = mid + 1;
                } else {
                    high = mid - 1; // search left
                }
            }
        }
        return -1; // error
    }

    /**
     * Find Minimum in Rotated Sorted Array II
     *
     * Follow up for "Find Minimum in Rotated Sorted Array": What if duplicates
     * are allowed?
     *
     * Would this affect the run-time complexity? How and why? Suppose a sorted
     * array is rotated at some pivot unknown to you beforehand.
     *
     * (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
     *
     * Find the minimum element.
     *
     * The array may contain duplicates.
     */
    public int findMin2(int[] num) {
        // idea - there is no lg(n) algo, since we cannot clearly eliminate half
        // of the array
        // so the idea is : reduce high and low boudary
        int low = 0;
        int high = num.length - 1;
        while (low < high) {
            int mid = low + (high - low) / 2;
            if (num[mid] > num[high]) {
                low = mid + 1;
            } else if (num[mid] < num[high]) {
                high = mid;
            } else {
                high--;
            }
        }
        return num[low];
    }

    /**
     * Combination Sum - sum k elements to target
     *
     * Given a set of candidate numbers (C) and a target number (T), find all
     * unique combinations in C where the candidate numbers sums to T.
     *
     * The same repeated number may be chosen from C unlimited number of times.
     *
     * Note: All numbers (including target) will be positive integers. Elements
     * in a combination (a1, a2, °≠ , ak) must be in non-descending order. (ie,
     * a1 °‹ a2 °‹ °≠ °‹ ak). The solution set must not contain duplicate
     * combinations. For example, given candidate set 2,3,6,7 and target 7, A
     * solution set is: [7] [2, 2, 3]
     */
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // algo - sort the arr first, then scan arr, increase first element, and
        // recursively call rest sum
        // if (candidates!=null && candidates.length)
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (candidates == null || candidates.length == 0) {
            return result;
        }

        Arrays.sort(candidates);
        result = combinationSumHelper(candidates, target, 0);
        return result;
    }

    private List<List<Integer>> combinationSumHelper(int[] arr, int target, int startIndex) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (startIndex >= arr.length || arr[startIndex] > target) {
            return result; // return empty list
        }

        // try combo as 0 a1, 1 a1, 2 a1...
        for (int i = startIndex; i < arr.length && arr[i] <= target; i++) {
            // to keep sorted order, loop starts with as many arr[i] multipls as
            // possible
            int firstSum = arr[i];
            while (firstSum < target) {
                firstSum += arr[i];
            }
            while (firstSum > 0) {
                // first element sums to target
                if (firstSum == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    for (int j = 0; j < firstSum / arr[i]; j++) {
                        list.add(arr[i]);
                    }
                    result.add(list);
                } else if (firstSum < target) {
                    List<List<Integer>> lastHalfSum = combinationSumHelper(arr, target - firstSum, i + 1);
                    for (List<Integer> half : lastHalfSum) {
                        List<Integer> list = new ArrayList<Integer>();
                        for (int j = 0; j < firstSum / arr[i]; j++) {
                            list.add(arr[i]);
                        }
                        list.addAll(half);
                        result.add(list);
                    }
                }
                firstSum -= arr[i];
            }
        }
        return result;
    }

    /**
     * Combination Sum II
     *
     * Given a collection of candidate numbers (C) and a target number (T), find
     * all unique combinations in C where the candidate numbers sums to T.
     *
     * Each number in C may only be used once in the combination.
     *
     * Note: All numbers (including target) will be positive integers. Elements
     * in a combination (a1, a2, °≠ , ak) must be in non-descending order. (ie,
     * a1 °‹ a2 °‹ °≠ °‹ ak). The solution set must not contain duplicate
     * combinations. For example, given candidate set 10,1,2,7,6,1,5 and target
     * 8, A solution set is: [1, 7] [1, 2, 5] [2, 6] [1, 1, 6]
     */
    public List<List<Integer>> combinationSum2(int[] num, int target) {
        // algo - sort the arr first, then scan arr, increase first element, and
        // recursively call rest sum
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (num == null || num.length == 0) {
            return result;
        }

        Arrays.sort(num);

        result = combinationSumRecursive(num, target, 0);
        return result;
    }

    private List<List<Integer>> combinationSumRecursive(int[] arr, int target, int startIndex) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        if (startIndex >= arr.length || arr[startIndex] > target) {
            return result; // return empty list
        }
        // sum up all duplicates, then subtract one by one and make combo with
        // rest
        int firstSum = 0;
        for (int i = startIndex; i < arr.length && arr[i] <= target; i++) {
            // duplicate count then continue
            if (i < arr.length - 1 && arr[i] == arr[i + 1]) {
                firstSum += arr[i];
                continue;
            }
            // now make combo of previous duplicats
            firstSum += arr[i];
            while (firstSum > 0) {
                // first element sums to target
                if (firstSum == target) {
                    List<Integer> list = new ArrayList<Integer>();
                    for (int j = 0; j < firstSum / arr[i]; j++) {
                        list.add(arr[i]);
                    }
                    result.add(list);
                } else if (firstSum < target) {
                    List<List<Integer>> lastHalfSum = combinationSumRecursive(arr, target - firstSum, i + 1);
                    for (List<Integer> half : lastHalfSum) {
                        List<Integer> list = new ArrayList<Integer>();
                        for (int j = 0; j < firstSum / arr[i]; j++) {
                            list.add(arr[i]);
                        }
                        list.addAll(half);
                        result.add(list);
                    }
                }
                firstSum -= arr[i];
            }
            firstSum = 0;
        }
        return result;
    }

    /**
     *
     */
    public boolean canJump(int[] A) {
        if (A == null || A.length == 0) {
            return true;
        }

        //@formatter:off
        // use dp, t[i] : i'th element is reachable
        // t[i] = true if
        //          t[i-1]=true && A[i-1] > 0 OR
        //          t[i-2]=true && A[i-2] > 1 OR
        //          ...
        //          t[0] = true && A[0] > i-1;
        //        false otherwise
        // init: t[0] = true;
        //@formatter:on
        boolean[] t = new boolean[A.length];
        t[0] = true;
        for (int i = 1; i < A.length; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (t[j] == true && A[j] > i - 1 - j) {
                    t[i] = true;
                    break;
                }
            }
        }
        return t[A.length - 1];
    }

    /**
     * Jump Game II
     *
     * Given an array of non-negative integers, you are initially positioned at
     * the first index of the array.
     *
     * Each element in the array represents your maximum jump length at that
     * position.
     *
     * Your goal is to reach the last index in the minimum number of jumps.
     *
     * For example: Given array A = [2,3,1,1,4]
     *
     * The minimum number of jumps to reach the last index is 2. (Jump 1 step
     * from index 0 to 1, then 3 steps to the last index.)
     */
    // Time Exception with DP
    public int jump(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }

        // there is O(n) algo, dp is not efficient and causes TLE
        // dp, t[i] = minimal steps to reach i
        // t[i] = t[j] + 1 if A[j] > i-1-j; where j=0, i-1
        int[] t = new int[A.length];
        // Arrays.fill(t, Integer.MAX_VALUE);
        t[0] = 0; // no jump needed
        for (int i = 1; i < A.length; i++) {
            for (int j = 0; j < i; j++) {
                if (A[j] > i - 1 - j) {
                    t[i] = t[j] + 1; // A[j]+1 must minimal because we scan from
                    // left to right
                    break;
                }
            }
        }
        return t[A.length - 1];
    }

    /**
     * Permutations
     *
     * Given a collection of numbers, return all possible permutations.
     *
     * For example, [1,2,3] have the following permutations: [1,2,3], [1,3,2],
     * [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
     */
    public List<List<Integer>> permute(int[] num) {
        // use next permutation
        List<List<Integer>> ret = new ArrayList<List<Integer>>();
        if (num == null || num.length == 0) {
            return ret;
        }

        // run each permutation
        int permuCount = 1;
        for (int i = 1; i <= num.length; i++) {
            permuCount *= i;
        }
        Arrays.sort(num);
        for (int i = 0; i < permuCount - 1; i++) {
            List<Integer> list = new ArrayList<Integer>();
            for (int k : num) {
                list.add(k);
            }
            ret.add(list);
            nextPermutation(num);
        }
        // add last permu
        List<Integer> list = new ArrayList<Integer>();
        for (int k : num) {
            list.add(k);
        }
        ret.add(list);

        return ret;
    }

    /**
     * Dungeon Game
     *
     *
     * The demons had captured the princess (P) and imprisoned her in the
     * bottom-right corner of a dungeon. The dungeon consists of M x N rooms
     * laid out in a 2D grid. Our valiant knight (K) was initially positioned in
     * the top-left room and must fight his way through the dungeon to rescue
     * the princess.
     *
     * The knight has an initial health point represented by a positive integer.
     * If at any point his health point drops to 0 or below, he dies
     * immediately.
     *
     * Some of the rooms are guarded by demons, so the knight loses health
     * (negative integers) upon entering these rooms; other rooms are either
     * empty (0's) or contain magic orbs that increase the knight's health
     * (positive integers).
     *
     * In order to reach the princess as quickly as possible, the knight decides
     * to move only rightward or downward in each step.
     *
     *
     * Write a function to determine the knight's minimum initial health so that
     * he is able to rescue the princess.
     *
     * For example, given the dungeon below, the initial health of the knight
     * must be at least 7 if he follows the optimal path RIGHT-> RIGHT -> DOWN
     * -> DOWN.
     *
     */
    // @formatter:off
    /*
        -2 (K)  -3   3
        -5      -10  1
        10      30  -5 (P)

    Notes:
        The knight's health has no upper bound.
        Any room can contain threats or power-ups, even the first room the knight enters and the bottom-right room where the princess is imprisoned.
     */
    // @formatter:on
    public int calculateMinimumHP(int[][] dungeon) {
        // idea - use dp, traverse from princess (last cell), since we know this
        // cell's minimal is 1, so we subtract each cell till (0,0)
        // if any subtractio at cell (i,j) is negative, then it means this cell
        // has large positive number, and we need to reset it to 1 (sum has to
        // be positve)

        // dp - traverse reverse from [m][n] back to [0][0]
        // sum[i][j] : expected sum to make total sum from dungeon(0,0) to
        // dungeon(i,j) be positive
        // note that sum[i][j] must >=1

        int rows = dungeon.length;
        int cols = dungeon[0].length;
        int[][] sum = new int[rows][cols];

        // if data is positive, then we need 0, otherwise abs()+1
        sum[rows - 1][cols - 1] = Math.max(1, 1 - dungeon[rows - 1][cols - 1]);
        // init last column
        for (int i = rows - 2; i >= 0; i--) {
            // if the sum till [i][j] <= 0, we set it be 1, otherwise take the
            // diff
            sum[i][cols - 1] = Math.max(1, sum[i + 1][cols - 1] - dungeon[i][cols - 1]);
        }
        // init last row
        for (int j = cols - 2; j >= 0; j--) {
            sum[rows - 1][j] = Math.max(sum[rows - 1][j + 1] - dungeon[rows - 1][j], 1);
        }
        for (int i = rows - 2; i >= 0; i--) {
            for (int j = cols - 2; j >= 0; j--) {
                // minimal has to be positive
                int down = Math.max(sum[i + 1][j] - dungeon[i][j], 1);
                int right = Math.max(sum[i][j + 1] - dungeon[i][j], 1);
                sum[i][j] = Math.min(down, right);
            }
        }

        return Math.max(sum[0][0], 1);
    }

    /**
     * Minimal adjustment cost - this is not leetcode but from lintcode
     *
     * Given an integer array, adjust each integers so that the difference of
     * every adjcent integers are not greater than a given number target.
     *
     * If the array before adjustment is A, the array after adjustment is B, you
     * should minimize the sum of |A[i]-B[i]|
     *
     * Note: You can assume each number in the array is a positive integer and
     * not greater than 100
     *
     * example:
     *
     * Given [1,4,2,3] and target=1, one of the solutions is [2,3,2,3], the
     * adjustment cost is 2 and it's minimal. Return 2.
     */
    public int minimalAdjust(int[] A, int target) {
        // use DP
        // cost[i][v] - minimal cost to change A[i] to value v (and satify above
        // condition)
        // cost[i][j] = Min (cost[i-1][v] + abs(A[i]-j) ), where v=1,...100,
        // j=1...100

        // skip single element
        if (A == null || A.length == 0) {
            return 0;
        }

        int max = 100; // all numbers are in [1,100]
        int[][] cost = new int[A.length][max + 1]; // 1,2...100

        for (int i = 0; i < A.length; i++) {
            for (int j = 1; j <= max; j++) {
                cost[i][j] = Integer.MAX_VALUE;
                if (i == 0) {
                    // for the first element, the change to j cost is
                    // automatically their distance, ignore if it satisify <=
                    // target
                    cost[i][j] = Math.abs(A[i] - j);
                } else { // update and populate value
                    for (int v = 1; v <= max; v++) {
                        if (Math.abs(j - v) > target) {
                            continue;
                        }
                        int dist = cost[i - 1][v] + Math.abs(A[i] - j);
                        cost[i][j] = Math.min(cost[i][j], dist);
                    }
                }
            }
        }
        int ret = Integer.MAX_VALUE;
        for (int i = 1; i <= max; i++) {
            ret = Math.min(cost[A.length - 1][i], ret);
        }
        return ret;
    }

    /**
     * Largest Number
     *
     * Given a list of non negative integers, arrange them such that they form
     * the largest number.
     *
     * For example, given [3, 30, 34, 5, 9], the largest formed number is
     * 9534330.
     *
     * Note: The result may be very large, so you need to return a string
     * instead of an integer.
     */
    public String largestNumber(int[] num) {
        // algo - sort based on defined comparable:
        // a > b in lexical sort, eg, 4>3, 4>40, 4>41, 4>42,4>43, 4==44, 4<45...
        // ie, compare prefix and suffix
        // based on this, implement quick sort, then construct string reversed;
        if (num == null || num.length == 0) {
            return "";
        }

        quicksort(num, 0, num.length - 1);
        StringBuilder sb = new StringBuilder();
        for (int i = num.length - 1; i >= 0; i--) {
            sb.append(num[i]);
        }
        // trim leading 0
        while (sb.length() > 1 && sb.charAt(0) == '0') {
            sb.deleteCharAt(0);
        }
        return sb.toString();
    }

    public void quicksort(int[] num, int left, int right) {
        if (left < right) {
            int q = partition(num, left, right);
            quicksort(num, left, q - 1);
            quicksort(num, q + 1, right);
        }
    }

    public int partition(int[] num, int left, int right) {
        int pivot = num[right];
        // i tracks small value; j tracks large value
        int i = left - 1;
        for (int j = left; j < right; j++) {
            // num[j] is smaller
            if (compare(num[j], pivot) <= 0) {
                i++;
                swap(num, i, j);
            }
        }
        swap(num, i + 1, right);
        return i + 1;
    }

    // lexical sort compare, but data may overflow
    private int compare(int a, int b) {
        String s1 = String.format("%s%s", a, b);
        String s2 = String.format("%s%s", b, a);
        return s1.compareTo(s2); // if s1 <s2, return neg; 0 if equal; or
        // positive if s1>s2
    }

    /**
     * Gray code grey code
     *
     * The gray code is a binary numeral system where two successive values
     * differ in only one bit.
     *
     * Given a non-negative integer n representing the total number of bits in
     * the code, print the sequence of gray code. A gray code sequence must
     * begin with 0.
     *
     * For example, given n = 2, return [0,1,3,2]. Its gray code sequence is:
     */
    //@formatter:off
    /*  00 - 0
        01 - 1
        11 - 3
        10 - 2
     */
    //@formatter:on
    /**
     * Note: For a given n, a gray code sequence is not uniquely defined.
     *
     * For example, [0,2,3,1] is also a valid gray code sequence according to
     * the above definition.
     *
     * For now, the judge is able to judge based on one instance of gray code
     * sequence. Sorry about that.
     */
    public List<Integer> grayCode(int n) {
        // the result contains 2^n numbers, which cannot be done by loops,
        // must be recursive -
        // get n-1 list, then attach 0 in front of sequence; attach 1 in front
        // of reversed sequence
        // eg, n=2, get n=1 as {0,1}, then => {00,01, 11, 10}
        if (n == 0) {
            List<Integer> list = new ArrayList<Integer>();
            list.add(0);
            return list;
        }
        List<Integer> half = grayCode(n - 1);
        List<Integer> full = new ArrayList<Integer>();
        // append 0 to head of list and 1 to the tail
        int i = 0;
        int k = 1;
        k <<= n - 1; // left shit n-1 bits to get 10000..
        for (int val : half) {
            full.add(i++, val); // add "0xx"
            full.add(i, k | val); // add"1xx to the end
        }
        return full;
    }

    /**
     * Implement pow(x, n).
     */
    public double pow(double x, int n) {
        if (n == 0) {
            return 1;
        }

        if (n < 0) {
            // note abs(Integer.MIN_VALUE) = abs(Integer.MAX_VALUE) + 1
            // so -Integer.Min_VALUE will overflow instead of positive int
            // handle this corner case
            if (n == Integer.MIN_VALUE) {
                return 1 / x * pow(1 / x, Integer.MAX_VALUE);
            } else {
                return pow(1 / x, -n);
            }
        }

        return n % 2 == 0 ? pow(x * x, n / 2) : x * pow(x * x, (n - 1) / 2);
    }

    /**
     * Sort color
     *
     * use O(N) and O(1) space
     */
    public void sortColors(int[] A) {
        // maintain low and high, squeeze from both end to middle
        // stop when i = high
        if (A == null || A.length == 0) {
            return;
        }
        int low = 0;
        int high = A.length - 1;
        int i = low;
        while (i <= high) {
            switch (A[i]) {
            case 0:
                if (i > low) {
                    swap(A, low, i);
                } else {
                    i++;
                }
                low++;
                break;
            case 1:
                i++;
                break;
            case 2:
                swap(A, i, high);
                high--;
                break;
            }
        }
    }

    /**
     * Construct Binary Tree from Inorder and Postorder Traversal
     *
     * Given inorder and postorder traversal of a tree, construct the binary
     * tree.
     *
     * Note: You may assume that duplicates do not exist in the tree.
     */

    // consideration - recusrive call divided by root,
    // improvement: 1. use map to reduce root lookup from o(n) to o(1)
    // improvement: 2. use array index but not array copy to reduce huge array
    // copy cost

    // map value to its index
    Map<Integer, Integer> indexLookup = new HashMap<Integer, Integer>();

    public TreeNode buildTree(int[] inorder, int[] postorder) {
        // algo - last elem in postorder is root, use it in inorder to divide
        // array to left child and right
        // then call recursively,
        // eg, in={1,2,3,4,5,6,7,8,9}, post={1,2,4,5,3,7,9,8,6}
        // 6 is root, then (1,2,3,4,5) is left child's inorder and matching post
        // order is (1,2,4,5,3)/
        // the rest is right child, then call recursively
        // with the help of map, we can reduce the recusive call from nlg(n) to
        // o(n)
        for (int i = 0; i < inorder.length; i++) {
            indexLookup.put(inorder[i], i);
        }

        return buildTree_recursive(inorder, postorder);
    }

    // copy array is much slower
    private TreeNode buildTree_recursive(int[] inorder, int[] postorder) {
        if (inorder == null || postorder == null || inorder.length == 0 || postorder.length == 0) {
            return null;
        }
        int rootIndex = indexLookup.get(postorder[postorder.length - 1]);

        // left child [0, rootIndex)
        int[] leftInorder = Arrays.copyOfRange(inorder, 0, rootIndex); // [)
        int[] leftPostorder = Arrays.copyOfRange(postorder, 0, rootIndex);

        // right child[rootIndex+1, len)
        int[] rightInorder = Arrays.copyOfRange(inorder, rootIndex + 1, inorder.length); // [)
        int[] rigthPostorder = Arrays.copyOfRange(postorder, rootIndex, postorder.length - 1);

        TreeNode rootNode = new TreeNode(postorder[postorder.length - 1]);
        rootNode.left = buildTree(leftInorder, leftPostorder);
        rootNode.right = buildTree(rightInorder, rigthPostorder);

        return rootNode;
    }

    public TreeNode buildTree2(int[] inorder, int[] postorder) {
        // algo - last elem in postorder is root, use it in inorder to divide
        // array to left child and right
        // then call recursively,
        // eg, in={1,2,3,4,5,6,7,8,9}, post={1,2,4,5,3,7,9,8,6}
        // 6 is root, then (1,2,3,4,5) is left child's inorder and matching post
        // order is (1,2,4,5,3)/
        // the rest is right child, then call recursively
        // with the help of map, we can reduce the recusive call from nlg(n) to
        // o(n)
        for (int i = 0; i < inorder.length; i++) {
            indexLookup.put(inorder[i], i);
        }
        int len = inorder.length;
        return buildTree_recursive(inorder, 0, len - 1, postorder, 0, len - 1);
    }

    // without copy array, this is much faster
    private TreeNode buildTree_recursive(int[] inorder, int s1, int e1, int[] postorder, int s2, int e2) {
        if (s1 > e1 || s2 > e2) {
            return null;
        }
        int rootIndex = indexLookup.get(postorder[e2]);

        TreeNode rootNode = new TreeNode(postorder[e2]);
        rootNode.left = buildTree_recursive(inorder, s1, rootIndex - 1, postorder, s2, s2 + rootIndex - s1 - 1);
        rootNode.right = buildTree_recursive(inorder, rootIndex + 1, e1, postorder, s2 + rootIndex - s1, e2 - 1);
        // rootNode.right = buildTree_recursive(inorder, rootIndex + 1, e1,
        // postorder, e2-e1 + rootIndex, e2 - 1);

        return rootNode;
    }

    /**
     * Construct Binary Tree from Preorder and Inorder Traversal
     *
     * Given preorder and inorder traversal of a tree, construct the binary
     * tree.
     *
     * Note: You may assume that duplicates do not exist in the tree.
     */
    // map value to its index
    Map<Integer, Integer> indexLookup2 = new HashMap<Integer, Integer>();

    public TreeNode buildTree_pre_inorder(int[] preorder, int[] inorder) {
        // same idea as inorder + postorder, just use preorder to define the
        // root
        for (int i = 0; i < inorder.length; i++) {
            indexLookup2.put(inorder[i], i);
        }
        int len = inorder.length;
        return buildTree_recursive2(inorder, 0, len - 1, preorder, 0, len - 1);
    }

    // without copy array, this is much faster
    private TreeNode buildTree_recursive2(int[] inorder, int s1, int e1, int[] preorder, int s2, int e2) {
        if (s1 > e1 || s2 > e2) {
            return null;
        }
        int rootIndex = indexLookup2.get(preorder[s2]);

        TreeNode rootNode = new TreeNode(preorder[s2]);
        rootNode.left = buildTree_recursive2(inorder, s1, rootIndex - 1, preorder, s2 + 1, s2 - s1 + rootIndex);
        rootNode.right = buildTree_recursive2(inorder, rootIndex + 1, e1, preorder, e2 - e1 + rootIndex + 1, e2);
        // rootNode.right = buildTree_recursive2(inorder, rootIndex + 1, e1,
        // postorder, e2-e1 + rootIndex, e2 - 1);

        return rootNode;
    }

    /**
     * Partition List
     *
     * Given a linked list and a value x, partition it such that all nodes less
     * than x come before nodes greater than or equal to x.
     *
     * You should preserve the original relative order of the nodes in each of
     * the two partitions.
     *
     * For example, Given 1->4->3->2->5->2 and x = 3, return 1->2->2->4->3->5.
     */
    public ListNode partition(ListNode head, int x) {
        // use two pointer, one track last smallest, the other walk through and
        // swap small to end of small pointer
        if (head == null) {
            return head;
        }

        ListNode dummy = new ListNode(Integer.MIN_VALUE);
        dummy.next = head;
        ListNode small = dummy;
        ListNode large = dummy;
        while (large.next != null) {
            if (large.next.val >= x) {
                large = large.next;
                continue;
            }
            // large.next.val < x
            // initial state, move together when smaller
            if (large == small) {
                large = large.next;
                small = small.next;
                continue;
            }
            ListNode smallNext = small.next;
            ListNode largeNext = large.next;
            large.next = large.next.next;
            small.next = largeNext;
            largeNext.next = smallNext;

            small = small.next;
        }
        return dummy.next;

    }

    /**
     * Mirror binary tree
     *
     * check if an binary tree is mirrow of another
     */
    public boolean isMirror(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null) {
            return root1 == root2;
        }
        // preorder traversal
        if (root1.val != root2.val) {
            return false;
        }

        return isMirror(root1.left, root2.right) && isMirror(root1.right, root2.left);
    }

    /**
     * Find duplicate start and end in sorted array
     *
     * A sorted array may contain duplicate, find a search key's start and end
     * index
     *
     * eg, {1,1,1,1,2,2,2,3,3,3} find 2, returns {4,6}
     */
    public int[] findDupInSortedArray(int[] arr, int target) {
        if (arr == null) {
            return null;
        }

        int[] ret = new int[2];
        int mid = -1;
        // use binary search, found target first, then search left for start
        // index and right for end index
        // use mid to store the first time found target index
        int l = 0;
        int r = arr.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (arr[m] == target) {
                mid = m;
                break;
            } else if (arr[m] < target) {
                l = m + 1;
            } else {
                r = m - 1;
            }
        }
        if (mid == -1) { // not found
            return null;
        }
        // from [0, mid] find starting index of target where arr[mid] = target
        l = 0;
        r = mid;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (arr[m] == target) {
                if (m - 1 < 0 || arr[m - 1] < arr[m]) { // found
                    ret[0] = m;
                    break;
                } else {
                    // search left part
                    r = m - 1;
                }
            } else { // arr[m] < target
                l = m + 1;
            }
        }
        // now found end index from [mid, len-1]
        l = mid;
        r = arr.length - 1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (arr[m] == target) {
                if (m + 1 == arr.length || arr[m] < arr[m + 1]) {
                    ret[1] = m;
                    break;
                } else {
                    l = m + 1; // searach right side
                }
            } else { // arr[m] > target
                r = m - 1;
            }
        }
        // now we got both start and end
        return ret;
    }

    /**
     * Given a sorted integer array without duplicates, return the summary of
     * its ranges.
     *
     * For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
     */
    public List<String> summaryRanges(int[] nums) {
        List<String> list = new ArrayList<String>();
        if (nums == null || nums.length == 0) {
            return list;
        }
        int start = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] == nums[i - 1] + 1) {
                continue;
            }
            append(list, start, nums[i - 1]);
            start = nums[i];
        }
        append(list, start, nums[nums.length - 1]);
        return list;
    }

    private void append(List<String> list, int start, int end) {
        StringBuilder sb = new StringBuilder();
        sb.append(start);
        if (end != start) {
            sb.append("->").append(end);
        }
        list.add(sb.toString());
    }

    /**
     * Implement a basic calculator to evaluate a simple expression string.
     *
     * The expression string contains only non-negative integers, +, -, *, /
     * operators and empty spaces . The integer division should truncate toward
     * zero.
     *
     * You may assume that the given expression is always valid.
     *
     * Some examples: "3+2*2" = 7 " 3/2 " = 1 " 3+5 / 2 " = 5
     */
    public int calculate(String s) {
        if (s == null || s.length() == 0) {
            throw new IllegalArgumentException();
        }
        // algo: build a dequeue to hold data and operator
        // scan the string,
        // if space, ignore and continue;
        // if number, accumulate and keep scanning till operator or end of
        // string;
        // if operator,
        // if op='*/', compute immediatley then euqueue result;
        // if op='+-', enqueu current number and op, reset number
        // compute queue
        Deque<String> deq = new LinkedList<String>();
        int num = 0; // accumulator
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch == ' ') {
                continue;
            }
            if (isNumber(ch)) {
                num = num * 10 + ch - '0';
                continue;
            }
            // now only operator left
            addToDeque(deq, num);
            deq.addLast(String.valueOf(ch));
            num = 0;
        }
        // handle last number
        addToDeque(deq, num);
        return processDequeu(deq);
    }

    private boolean isNumber(char ch) {
        return ch >= '0' && ch <= '9';
    }

    private boolean isPriorityOp(String s) {
        return s.equals("*") || s.equals("/");
    }

    private boolean isAddOrSub(String s) {
        return s.equals("+") || s.equals("-");
    }

    private void addToDeque(Deque<String> deq, int num) {
        if (!deq.isEmpty() && isPriorityOp(deq.peekLast())) {
            String op = deq.removeLast(); // '*/' pops out
            int op1 = Integer.valueOf(deq.removeLast());
            num = op.equals("*") ? (op1 * num) : (op1 / num);
        }
        deq.addLast(String.valueOf(num));
    }

    private int processDequeu(Deque<String> deq) {
        int num = 0;
        String op = "";
        while (!deq.isEmpty()) {
            String s = deq.pollFirst();
            if (isAddOrSub(s)) {
                op = s;
                continue;
            }
            // now s is number
            if (op.equals("+")) {
                num += Integer.valueOf(s);
            } else if (op.equals("-")) {
                num -= Integer.valueOf(s);
            } else { // op is not set
                num = Integer.valueOf(s);
            }
        }
        return num;
    }

    /**
     *
     */
    public TreeNode invertTree(TreeNode root) {
        invert(root);
        return root;
    }

    private void invert(TreeNode root) {
        if (root == null) {
            return;
        }

        TreeNode left = root.left;
        root.left = root.right;
        root.right = left;

        invertTree(root.left);
        invertTree(root.right);
    }
}
