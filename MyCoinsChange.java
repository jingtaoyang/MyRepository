import java.util.Arrays;

public class MyCoinsChange {

	 public int coinChange(int[] coins, int amount) {
	        if ((amount<0) || (coins==null) ||(coins.length<1))
				return -1;
			Arrays.sort(coins);
			int max = coins[coins.length-1];
			int index = coins.length-1;
			int cnt = 0;
			int div = 0;
			int rem = 0;
			while(index>=0){
			    max = coins[index];
				div = amount/max;
				rem = amount%max;
				if (div >0){
					cnt = cnt + div;
				}
				amount = rem;
				index = index-1;
			}
			
			if (rem>0) 
			    return -1;
			return cnt;
	    }
	 
	 public static void main(String[] args){
		 
		 int[] coins = {186,419,83,408};
		 int amount = 6249;
		 
		 MyCoinsChange mcc = new MyCoinsChange();
		 System.out.println(mcc.coinChange(coins, amount));
	 }
}