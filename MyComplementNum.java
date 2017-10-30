
public class MyComplementNum {

	public int findComplement(int num) {

		if (num==0) return 0;
		int div=num/2;
		int rem=num%2;
		int newNum=0;
		int pow=0;
		while (div !=0){
			newNum += (rem==0)? 1*Math.pow(2,pow):0;
			pow++;
			rem = div%2;
			div = div/2;
		}
		newNum += (rem==0)? 1*Math.pow(2,pow):0;
		return newNum;
	}

	public static void main(String[] args){
		
		MyComplementNum mcn = new MyComplementNum();
		mcn.findComplement(5);

	}

}
