import java.util.ArrayList;
import java.util.List;

public class MyCompareVersionNumber {
	
public int compareVersion(String version1, String version2) {
        
        if ((version1==null) || (version2==null))
            return 0;
        
        List<String> v1s = new ArrayList<String>();
        List<String> v2s = new ArrayList<String>();
        int dotInd1 = -1;
        int dotNext1 = 0;
        int dotInd2 = -1;
        int dotNext2 = 0;
        
        while(true){
        	dotNext1 = version1.indexOf(".",dotInd1+1);
        	if (dotNext1<0){
        		v1s.add(version1.substring(dotInd1+1));//end
        		break;
        	}else{
        		v1s.add(version1.substring(dotInd1+1,dotNext1));
        		dotInd1 = dotNext1;
        		if (dotInd1>version1.length()-1) break;
        	}
        }
        
        while(true){
        	dotNext2 = version2.indexOf(".",dotInd2+1);
        	if (dotNext2<0){
        		v2s.add(version2.substring(dotInd2+1));//end
        		break;
        	}else{
        		v2s.add(version2.substring(dotInd2+1,dotNext2));
        		dotInd2 = dotNext2;
        		if (dotInd2>version2.length()-1) break;
        	}
        }
        
        
        for(int i=0; i<v1s.size(); i++){
        	
        	if (i>=v2s.size())
        		return 1;
        	
        	if (Integer.parseInt(v1s.get(i)) > Integer.parseInt(v2s.get(i))){
                return 1;
            }
            if (Integer.parseInt(v1s.get(i)) < Integer.parseInt(v2s.get(i))){
                return -1;
            }
        	
        }
        
        if (v2s.size()>v1s.size())
        	return -1;
     
        return 0;
    }

	public static void main(String[] args){
		
		MyCompareVersionNumber mcvn = new MyCompareVersionNumber();
		
		int ret = mcvn.compareVersion("1.1.1", "1.1");
		System.out.println(ret);
		
	}
}
