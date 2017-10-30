// Determine whether an integer is a palindrome. Do this without extra space.

public class MyPalindrome {

    public boolean isPalindrome(int x) {
        if (x<0) return false;
        if (x==0) return true;
        
        int y = x;
        int z = 0;
        
        while ((y/10!=0)||(y%10!=0)){
            z=z*10+y%10;
            y=y/10;
        }
        if (z==x) return true;
        return false;
    }
	
	public static void main(String[] args){
		MyPalindrome my = new MyPalindrome();
		System.out.println(my.isPalindrome(11));
		
	}
}
