/* The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: "PAHNAPLSIIGYIR"
Write the code that will take a string and make this conversion given a number of rows:
string convert(string text, int nRows);
convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR". 
*/

public class MyZigZag {
	
   public String convert(String s, int numRows) {
        
        if ((s==null) || (numRows <= 0))
            return "";
        if ((numRows==1) || (numRows >=s.length()))
            return s;
        String newS = "";
        for (int i=1; i<=numRows; i++){
            int j = i-1;
            while(j< s.length()){
                if (j==s.length()-1)
                    newS = newS + s.substring(j);
                else
                    newS = newS + s.substring(j,j+1);
                if (i<=numRows/2)
                   j = j + i*2;
                else
                   j = j + 2*(numRows-i+1);
            }
        }
        return newS;
    }
   
    public static void main(String[] args){
    	
    	MyZigZag mzz = new MyZigZag();
    	System.out.println(mzz.convert("ABCD",3));
    	
    }

}


/*
0   4     8
1 3   5  7
2      6

numrow = 3
0 4 8 1 3 5 7 2 6
+4+4 +2+2 +4

(numrows-i+1)*2 
0: 4  2*3-2
1: 2  2*3-4
2: 4  2*3-2

nowrow = 2
024681357

02468
1357

0: 2*2-2
1: 2*2-2



numrows = 4
06
157
248
3

06 15 7 24 8 3
+6+4+2+2+4

1: 2*4 -2*(1)
2: 2*4 -2*(2)
3: 2*4 -2*(2)
4: 2*4 -2*(1)
*/