import java.util.ArrayList;

public class MyKeyboardRow {
	public String[] findWords(String[] words) {

		if ((words==null)||(words.length==0)) return words;
		ArrayList<String> al = new ArrayList<String>();
		String line1="QWERTYUIOPqwertyuiop";
		String line2="ASDFGHJKLasdfghjkl";
		String line3="ZXCVBNMzxcvbnm";

		for (int i=0; i<words.length; i++){
			boolean outOfline = false;
			String line ="";
			if (line1.indexOf(words[i].charAt(0))>=0)
				line = line1;
			if (line2.indexOf(words[i].charAt(0))>=0)
				line = line2;
			if (line3.indexOf(words[i].charAt(0))>=0)
				line = line3;

			for(int j=1;j<words[i].length();j++){
				if (line.indexOf(words[i].charAt(j))<0){
					outOfline = true;
					break;
				}
			}
			if (outOfline==false)
				al.add(words[i]);
		}
		return al.toArray(new String[al.size()]);
	}

	public static void main(String[] args){
		MyKeyboardRow mkr=new MyKeyboardRow();
		String [] arr = {"Hello","Alaska","Dad","Peace"};
		mkr.findWords(arr);
	}

}
