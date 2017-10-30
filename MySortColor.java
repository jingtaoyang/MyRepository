
public class MySortColor {

    public void sortColors(int[] nums) {
        if (nums==null) return;
        if (nums.length==1) return;
        int[] cnts = new int[3];
        
        for (int num: nums){
            cnts[num]=cnts[num]+1;
        }
        for (int i=0;i<cnts[0];i++){
            nums[i]=0;
        }
        for (int j=0;j<cnts[1];j++){
            if ((cnts[0]+j)<nums.length)
               nums[cnts[0]+j]=1;
        }
        for (int k=cnts[0]+cnts[1];k<nums.length;k++){
            nums[k]=2;
        }
    }
    
    
    public void sortColors1(int[] nums) {
    	if (nums==null) return;
        if (nums.length==1) return;
        int index0 = 0;
        int index2 = nums.length -1;
        int i=0;
        while(i<=index2){
        	switch(nums[i]){
        	case 0:
        		if (i!=index0){
        			nums[i]=nums[index0];
        			nums[index0]=0;
        		}else{
        		    i++;
        		}
        		index0++;
        		break;
        	case 1:
        	    i++;
        		break;
        	case 2:
        		nums[i] = nums[index2];
        		nums[index2]=2;
        		index2--;
        	    break;
        	}
        }
    }
    
    
    public static void main(String[] args){    	
    	MySortColor msc = new MySortColor();
    	int[] ints = {0,0};
    	msc.sortColors(ints);
    	
    }
    
}
