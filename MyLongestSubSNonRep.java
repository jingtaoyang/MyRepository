import java.util.HashMap;

public class MyLongestSubSNonRep {
	
	public int lengthOfLongestSubstring(String s) {
        if ((s==null)||(s.length()==0))
            return 0;
        if (s.length()==1)
        	return 1;
        int fast = 1;
        int slow = 0;
        int max = 1;
        String subS = s.charAt(0)+"";
        while(fast<s.length()){
        	String c = s.charAt(fast)+"";
        	if(!subS.contains(c)){
        		subS=subS+c;
        		if (subS.length()>max)
        			max = subS.length();
        	}else{
        		for(int i=slow;i<fast;i++){
        			if (s.charAt(i)==s.charAt(fast)){
        				slow = i+1;
        				int index = subS.indexOf(s.charAt(i));
        				if (index!=fast-1){
        					subS = subS.substring(index+1);
        					subS = subS+s.charAt(i);
        				}else{
        					subS = ""+s.charAt(i);
        				}
        				break;
        			}
        		}
        	}
        	fast++;
        }
        if(subS.length()>max)
    	    max = subS.length();
        return max;
        
    }

	
	public static void main(String[] args){
		String input = "bpfbhmipx";
		MyLongestSubSNonRep ml = new MyLongestSubSNonRep();
		System.out.println(ml.lengthOfLongestSubstring(input));
		
	}
}
