/*
Merge two sorted linked lists and return it as a new list. The new list 
should be made by splicing together the nodes of the first two lists.		
*/

public class MyMergeTwoSortList {
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




}
