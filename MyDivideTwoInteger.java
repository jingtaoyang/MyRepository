/*Divide two integers without using multiplication, division and mod operator.
If it is overflow, return MAX_INT. 
*/
public class MyDivideTwoInteger {
	
	public int divide(int dividend, int divisor) {
        
        if (divisor==0) return Integer.MAX_VALUE;
        if ((dividend>Integer.MAX_VALUE)||(dividend<Integer.MIN_VALUE)) return Integer.MAX_VALUE;
        if ((divisor>Integer.MAX_VALUE)||(divisor<Integer.MIN_VALUE)) return Integer.MAX_VALUE;
        
        long divisorL = (long) divisor;
        long dividendL = (long) dividend;
        
        boolean isNegative = false;
        if (dividendL<0) {
            isNegative = true;
            dividendL = 0 - dividendL;
        }
        if (divisorL<0){
            isNegative = isNegative? false:true;
            divisorL = 0 - divisorL;
        }
        long ret = 0;
        long inc = divisorL;
        while((dividendL >= divisorL)&&(divisorL>=0)){
            ret = ret + 1;
            divisorL = divisorL +inc;
        }
        if (isNegative) ret = 0-ret;
        if ((ret>Integer.MAX_VALUE)||(ret<Integer.MIN_VALUE)) ret=Integer.MAX_VALUE;
        return (int)ret;
    }
	
	public static void main(String[] args){
		
		MyDivideTwoInteger mdt = new MyDivideTwoInteger();
		System.out.println(mdt.divide(-2147483648,-1));
		
	}

}
