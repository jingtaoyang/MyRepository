import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;

public class MyStringUtils {


	/*(Given a pattern and a string str, find if str follows the same pattern.
	Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty word in str.
	Examples:
	pattern = "abba", str = "dog cat cat dog" should return true.
	pattern = "abba", str = "dog cat cat fish" should return false.
	pattern = "aaaa", str = "dog cat cat dog" should return false.
	pattern = "abba", str = "dog dog dog dog" should return false.
	 */
	public boolean wordPattern(String pattern, String str) {
		String[] arr= str.split(" ");
		HashMap<Character, String> map = new HashMap<Character, String>();
		if(arr.length!= pattern.length())
			return false;
		for(int i=0; i<arr.length; i++){
			char c = pattern.charAt(i);
			if(map.containsKey(c)){
				if(!map.get(c).equals(arr[i]))
					return false;
			}else{
				if(map.containsValue(arr[i]))
					return false;
				map.put(c, arr[i]);
			}    
		}
		return true;
	}


	public String reverseString(String s) {

		if (s==null) return null;
		char[] cs = s.toCharArray();
		char tmp=' ';
		for(int i=0,j=cs.length-1;i<j;i++,j--){
			tmp = cs[i];
			cs[i] = cs[j];
			cs[j] = tmp;
		}
		return new String(cs);
	}

	//Implement strStr().
	//Returns the index of the first occurrence of needle in haystack, 
	//or -1 if needle is not part of haystack.

	public int strStr(String haystack, String needle) {

		if ((haystack==null)||(needle==null)) return -1;
		int lenH = haystack.length();
		int lenN = needle.length();
		if ((lenH==0)&&(lenN==0)) return 0;
		if (lenH==0) return -1;
		if (lenN>lenH) return -1;

		char cN = needle.charAt(0);
		for(int i=0; i<= lenH-lenN; i++){
			char cH = haystack.charAt(i);
			if (cH==cN){
				if (haystack.substring(i,i+lenN).equals(needle)){
					return i;}
			}
		}
		return -1;
	}


	// Write a function to find the longest common prefix string amongst an array of strings. 

	public String longestCommonPrefix(String[] strs) {
		if ((strs==null) || (strs.length<1) || (strs[0]==null) ) return "";
		String prefix = "";
		String prePrefix = "";
		for(int j=0;j<strs[0].length();j++){
			prefix=prefix+strs[0].charAt(j);
			for (int i=0;i<strs.length;i++){
				if (strs[i]==null) return null;
				if (!strs[i].startsWith(prefix)){
					return prePrefix;        
				}
			}
			prePrefix=prefix;
		}
		return strs[0];
	}


	//MyOneCharLess

	public String getNewString(String st, int ind){

		if (st==null) return st;
		int len = st.length();
		if (ind > len) return st;

		if (ind == len){
			return st.substring(0,len-1);
		}
		return st.substring(0,ind-1) + st.substring(ind);
	}


	/*
	 * Given two strings s and t, write a function to determine if t is an anagram of s.
    	For example,
    	s = "anagram", t = "nagaram", return true.
    	s = "rat", t = "car", return false.
    	Note:
    	You may assume the string contains only lowercase alphabets.
	 */
	public boolean isAnagram(String s, String t) {
		if((s==null)&&(t==null)) return true;
		if((s==null)||(t==null)) return false;
		if(s.length()!=t.length()) return false;
		HashMap<Character,Integer> hm = new HashMap<Character,Integer>();
		for(int i=0;i<s.length();i++){
			Character c = s.charAt(i);
			if (hm.get(c)==null){
				hm.put(c, 1);
			}else{
				hm.put(c,hm.get(c)+1);
			}
		}
		for(int j=0;j<t.length();j++){
			Character c = t.charAt(j);
			if (hm.get(c)==null){
				return false;
			}else{
				int cnt = hm.get(c);
				if (cnt==0) return false;
				hm.put(c,cnt-1);
			}
		}
		return true;
	}

	/*
	Given a string and a string dictionary, find the longest string in the dictionary 
	that can be formed by deleting some characters of the given string. If there are 
	more than one possible results, return the longest word with the smallest 
	lexicographical order. If there is no possible result, return the empty string.
    Example 1: 
			Input: s = "abpcplea", d = ["ale","apple","monkey","plea"]
			Output: "apple"		
	Example 2:
			Input: s = "abpcplea", d = ["a","b","c"]
			Output: "a" 
	 */
	public String findLongestWord(String s, List<String> d) {

		if ((d==null)||(d.size()==0)) return "";
		if ((s==null)||(s.length()==0)) return "";
		String retS = "";
		HashMap<Character, Integer> hm = new HashMap<Character, Integer>();
		for(int i=0; i<s.length(); i++){
			Character c = s.charAt(i);
			if (hm.get(c)==null){
				hm.put(c,1);
			}else{
				hm.put(c, hm.get(c)+1);
			}
		}

		String[] sa = d.toArray(new String[d.size()]);
		Arrays.sort(sa);
		for(String st:sa){
			HashMap<Character, Integer> hm1 = new HashMap<Character, Integer>();
			for (int j=0;j<st.length();j++){
				if (hm.get(st.charAt(j))==null) break;
				if (hm1.get(st.charAt(j))==null) hm1.put(st.charAt(j),1);
				else{
					if (hm1.get(st.charAt(j))>=hm.get(st.charAt(j))) break;
					else hm1.put(st.charAt(j),hm1.get(st.charAt(j))+1);
				}
				if (j==st.length()-1){
					if (st.length()>retS.length()) retS=st;
				}
			}
		}

		return retS;
	}

	/*Given a string, sort it in decreasing order based on the frequency of characters.
	 Example 1: Input: "tree"
	 Output: "eert"
	 Explanation:
	 'e' appears twice while 'r' and 't' both appear once.
	 So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.
	 */
	public String frequencySort(String s) {

		if (s==null) return s;
		HashMap<Character,Integer> hm = new HashMap<Character, Integer>();
		for (int i=0; i<s.length(); i++){
			Character c = s.charAt(i);
			if (hm.get(c) == null)
				hm.put(c, 1);
			else
				hm.put(c, hm.get(c)+1);
		}
		Character ll[] = new Character[hm.size()];
		int index = 0;
		for(Character c:hm.keySet()){
			ll[index] = c;
			index++;
		}
		Arrays.sort(ll, new Comparator<Character>(){
			public int compare(Character a, Character b) {
				// TODO Auto-generated method stub
				return hm.get(b)-hm.get(a);
			}
		});	
		StringBuffer sb = new StringBuffer();
		for (int i=0;i<ll.length;i++){
			for(int j=0;j<hm.get(ll[i]);j++)
				sb.append(ll[i]);
		}
		return sb.toString();
	}

	/*Given an input string, reverse the string word by word.
	 * For example, Given s = "the sky is blue",
	 * return "blue is sky the".*/
	public String reverseWords(String s) {

		if ((s==null)||(s.length()==0)) return s;
		String ret = "";
		char[] cs = s.toCharArray();
		Stack<Character> cStack = new Stack<Character>();

		for (int i=0;i<cs.length;i++){
			if (cs[i]!=' '){
				cStack.push(cs[i]);
			}
			if ((cs[i]==' ')&&(cStack.empty()==false)){
				String s1="";
				while(!cStack.empty()){
					s1 = cStack.pop() + s1;
				}
				if (ret=="")
					ret = s1;
				else
					ret = s1+" "+ret;
			}
		}

		String s2="";
		while(!cStack.empty()){
			s2 = cStack.pop() + s2;
		}
		if ((ret=="")&&(s2!=""))
			ret = s2;
		else if ((s2!=""))
			ret = s2+" "+ret;

		return ret;
	}


	/*
	 * Given a string, you need to reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.
Example 1:
Input: "Let's take LeetCode contest"
Output: "s'teL ekat edoCteeL tsetnoc"
Note: In the string, each word is separated by single space and there will not be any extra space in the string.
	 */
	public String reverseWordsI(String s) {

		if ((s==null)||(s.length()==0))
			return s;
		char[] cs = s.toCharArray();
		int b = 0;
		int e = 0;
		int len = s.length();
		while(b<len){
			if (cs[b]==' '){
				b++;
				e=b;
			}else{
				if ((e<len)&&(cs[e]!=' ')){
					e++;
				}else{
					e--;
					reverse(cs, b, e);
					e++;
					b=e;
				}
			}    
		}
		return new String(cs);
	}

	public void reverse(char[] cs, int b, int e){
		if ((cs==null)||(cs.length==0)) return;
		while(b<e){
			char tmp = cs[b];
			cs[b] = cs[e];
			cs[e] = tmp;
			b++;
			e--;
		}
		return;
	}



	/*
	 * Given a word, you need to judge whether the usage of capitals in it is right or not.
		We define the usage of capitals in a word to be right when one of the following cases holds.
		All letters in this word are capitals, like "USA".
		All letters in this word are not capitals, like "leetcode".
		Only the first letter in this word is capital if it has more than one letter, like "Google".
		Otherwise, we define that this word doesn't use capitals in a right way.
		Example 1: Input: "USA" Output: True
		Example 2: Input: "FlaG" Output: False
	 */
	public boolean detectCapitalUse(String word) {

		if ((word==null)||(word.length()<2)) return true;  
		boolean is1stUp = ((word.charAt(0)>='A')&&(word.charAt(0)<='Z'))? true:false;
		int upCnt = is1stUp ? 1:0;
		int lowCnt = is1stUp? 0:1;
		int len = word.length();  
		for (int i=1; i<len; i++){
			char c = word.charAt(i);
			if ((c>='A')&&(c<='Z')){
				upCnt++;
				if (!is1stUp)
					return false;
			}else if ((c>='a')&&(c<='z')){
				lowCnt++;
			}
		}
		if ((upCnt==len)||(lowCnt==len))
			return true;
		if ((is1stUp)&&(lowCnt==len-1))
			return true;
		return false; 
	}


	/*
Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.
Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.
The order of output does not matter.
Example 1:
Input: s: "cbaebabacd" p: "abc"
Output:
[0, 6]
Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".
Example 2:
Input:s: "abab" p: "ab"
Output:
[0, 1, 2]
Explanation:
The substring with start index = 0 is "ab", which is an anagram of "ab".
The substring with start index = 1 is "ba", which is an anagram of "ab".
The substring with start index = 2 is "ab", which is an anagram of "ab".
	 */
	public List<Integer> findAnagrams(String s, String p) {
	        
	        List<Integer> ret = new ArrayList<Integer>();
	        if ((s==null)||(p==null)) return ret;
	        int lenP = p.length();
	        int lenS = s.length();
	        if (lenS<lenP) return ret;
	        int i=0;
	        int j=0;
	        int[] intP = new int[26];
	        
	        for (j=0;j<lenP;j++){
	            intP[p.charAt(j)-'a']=intP[p.charAt(j)-'a']+1;
	        }
	        j = 0;
	        int cnt = lenP;
	        while(j<lenS){
	            if (intP[s.charAt(j)-'a'] >= 1){
	                cnt--;
	            }
	            intP[s.charAt(j)-'a']--;
	            j++;
	            if (cnt==0) {
	                ret.add(i);
	            }
	            if ((j-i)==lenP){
	                if(intP[s.charAt(i)-'a']>=0){
	                    cnt++;
	                }
	                intP[s.charAt(i)-'a']++;
	                i++;
	            }
	        }
	        return ret;
	}


	/*
	 * Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
	 * For example,
	 * "A man, a plan, a canal: Panama" is a palindrome.
	 * "race a car" is not a palindrome.
	 */
	public boolean isPalindrome(String s) {

		if ((s==null)||(s.length()==1)) return true;
		int b = 0;
		int e = s.length()-1;
		while(b<e){
			char bc = s.charAt(b);
			char ec = s.charAt(e);
			if ((bc == ec )||(isSameChar(bc,ec))){
				b++;
				e--;
			}else if ((!isCharacter(bc))&&(!isNumber(bc))){
				b++;
			}else if ((!isCharacter(ec))&&(!isNumber(ec))){
				e--;
			}else{
				return false;
			}
		}
		return true;

	}

	public boolean isNumber(char c){
		if (('0'<=c)&&(c<='9'))
			return true;

		return false;
	}

	public boolean isCharacter(char c){
		if (('a'<=c)&&(c<='z'))
			return true;
		if (('A'<=c)&&(c<='Z'))
			return true;
		return false;
	}

	public boolean isSameChar(char a, char b){
		if (!isCharacter(a) || !isCharacter(b))
			return false;
		if (((a-'a')==(b-'a')) || ((a-'A')==(b-'A')) || ((a-'a')==(b-'A')) || ((a-'A')==(b-'a')))   
			return true;
		return false;
	}




	public static void main(String[] args){

		MyStringUtils msu=new MyStringUtils();
		//String pattern = "abba";
		//String st = "dog cat cat dog";
		//System.out.println(msu.wordPattern(pattern, st));

		msu.isPalindrome("0P");

		msu.findAnagrams("eklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbg", "yqrbgjdwtcaxzsnifvhmou");

		msu.reverseWords("   a   b ");

		//["ale","apple","monkey","plea"]
		//"aewfafwafjlwajflwajflwafj"
		//["apple","ewaf","awefawfwaf","awef","awefe","ewafeffewafewf"]
		List<String> d = new ArrayList<String>();
		d.add("apple");
		d.add("ewaf");
		d.add("awefawfwaf");
		d.add("awef");
		d.add("awefe");
		d.add("ewafeffewafewf");

		msu.findLongestWord("aewfafwafjlwajflwajflwafj", d);

		String hay = "a";
		String need = "a";
		int ret = msu.strStr(hay, need);
		System.out.println(ret);
	}


}
