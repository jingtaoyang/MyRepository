//Count the number of prime numbers less than a non-negative number, n.
import java.util.ArrayList;
public class MyCntPrimeNum {
	 public int countPrimes(int n) {
	        if (n<=2) return 0;
	        //if (n==2) return 1;
	        if (n==3) return 1;
	        if (n==4) return 2;
	        ArrayList<Integer> pList = new ArrayList<Integer>();
	        //pList.add(2);
	        pList.add(3);
	        //int cnt = 2;
	        for (int i=5;i<n;i=i+2) {
	        	if (i%3==0) continue;
	        	if (i%5==0) continue;
	        	if (i%7==0) continue;
	        	if (i%11==0) continue;
	        	if (i%13==0) continue;
	        	if (i%17==0) continue;
	            for (int j=0;j<pList.size();j++){
	                if (i%pList.get(j)==0) break;
	            }
	            pList.add(i);
	            //cnt++;
	        }
	        return pList.size()+1;
	    }
	 
	 public static void main(String[] args){
		 MyCntPrimeNum mcp = new MyCntPrimeNum();
		 System.out.println(mcp.countPrimes(1500000));
	 }
}
