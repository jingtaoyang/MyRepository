/*Given numRows, generate the first numRows of Pascal's triangle.
**For example, given numRows = 5,
**Return 
**[
** [1],
**[1,1],
**[1,2,1],
**[1,3,3,1],
**[1,4,6,4,1]
**]
*/
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MyPascalTriangle {
	
	public List<List<Integer>> generate(int numRows) {
        
		List<List<Integer>> tL = new ArrayList<List<Integer>>();
		if (numRows<1) return tL;
		
		List<Integer> nL = Arrays.asList(1);
		tL.add(nL);
				
		for (int i=1; i<numRows; i++){
			List<Integer> mL = new ArrayList<Integer>();
			List<Integer> pL = tL.get(i-1);
			for(int j=0;j<=i;j++){
				if ((j==0)||(j==i)) 
					mL.add(1);
				else
				    mL.add(pL.get(j)+pL.get(j-1));
			}
			tL.add(mL);
		}   
		return tL;
    }
}
