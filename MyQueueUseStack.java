import java.util.Stack;

/*
 *  Implement the following operations of a queue using stacks.

    push(x) -- Push element x to the back of queue.
    pop() -- Removes the element from in front of queue.
    peek() -- Get the front element.
    empty() -- Return whether the queue is empty.
 */

public class MyQueueUseStack {
	
	Stack<Integer> s1 = new Stack<Integer>();
	Stack<Integer> s2 = new Stack<Integer>();
	
	// Push element x to the back of queue.
    public void push(int x) {
        s1.push(x);
    }

    // Removes the element from in front of queue.
    public void pop() {
        if (s1.isEmpty()) return;
        while (!s1.isEmpty()){
        	s2.push(s1.pop());
        }
        s2.pop();
        while(!s2.isEmpty()){
        	s1.push(s2.pop());
        }
    }

    // Get the front element.
    public int peek() {
        while (!s1.isEmpty()){
        	s2.push(s1.pop());
        }
        int ret = s2.peek();
        while(!s2.isEmpty()){
        	s1.push(s2.pop());
        }
        return ret;
    }

    // Return whether the queue is empty.
    public boolean empty() {
        return s1.empty();
    }
}
