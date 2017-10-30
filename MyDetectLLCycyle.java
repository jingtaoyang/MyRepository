//Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
//Note: Do not modify the linked list.

public class MyDetectLLCycyle {
	public MyListNode detectCycle(MyListNode head) {
        if (head==null) return null;
        if (head.next==null) return null;
        MyListNode fast = head;
        MyListNode slow = head;
        
        while(true){
            if (fast.next==null) return null;
            fast = fast.next;
            if (fast.next==null) return null;
            fast = fast.next;
            slow = slow.next;
            if (slow==fast) { //cyclye
                if (slow==head) return head;
                slow = head;
                while(true){
                    slow=slow.next;
                    fast=fast.next;
                    if (slow==fast) return slow;
                }
            }
        }
        
    }

}
