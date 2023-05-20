import java.util.*;
import java.io.*;
import java.lang.*;

public class clafflac
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("calfflac.in"));
      String str = "";
      while(in.hasNext())
         str += in.nextLine() + '\n';
      
      in.close();
      
      int max = 0;
      int start = 0, end = 0;
      String palinString = "";
      
      for(int i = 0; i < str.length(); i++)
      {
         int[] result;
         if(Character.isLetter(str.charAt(i)))
         {  
            int nextLetter = nextLetterRight(i, str);
            result = palindrome(i, nextLetter, str);
         }
         else
         {
            i = nextLetterRight(i, str);
            int nextLetter = nextLetterRight(i, str);
            result = palindrome(i, nextLetter, str);
         }
         
         if(result[2] > max)
         {
            start = result[0];
            end = result[1];
            max = result[2];
         }
      }
      
      System.out.println(max);
      
      palinString = str.substring(start, end + 1);
      palinString = palinString.replaceAll("^[^a-zA-Z0-9\\s]+|[^a-zA-Z0-9\\s]+$", "");
      System.out.println(palinString);
   }
   
   public static int[] palindrome(int pos1, int pos2, String str)
   {
      int[] result = new int[3];
      
      boolean even = false;
      
      int count = 0, center = pos1;
      
      if(Character.toLowerCase(str.charAt(pos1)) != Character.toLowerCase(str.charAt(pos2)))
      {
         pos2 = pos1;
         count = 1;
      }
      else
      {
         count = 2;
         even = true;
      }
      
      while(pos1 > 0 && pos2 < str.length())
      {
         pos1 = nextLetterLeft(pos1, str);
         pos2 = nextLetterRight(pos2, str);
         
         if(pos1 >= 0 && pos2 < str.length() && Character.toLowerCase(str.charAt(pos1)) == Character.toLowerCase(str.charAt(pos2)))
         {
            count += 2;
            continue;
         }
         else
         {
            if(center != 0)
            {
               pos1 = nextLetterRight(pos1, str);
               pos2 = nextLetterLeft(pos2, str);
            }
            
            break;
         }
      }
      
      if(even == true)
      {
         if(Math.min(pos1, pos2) > 0 && str.charAt(nextLetterLeft(Math.min(pos1, pos2), str)) == str.charAt(Math.max(pos1, pos2)))
         {
            count++;
            if(pos1 > pos2)
               pos2 = nextLetterLeft(pos2, str);
            else
               pos1 = nextLetterLeft(pos1, str);
         }
         
         else if(Math.max(pos1, pos2) < str.length() - 1 && str.charAt(nextLetterRight(Math.max(pos1, pos2), str)) == str.charAt(Math.min(pos1, pos2)))
         {
            count++;
            if(pos1 > pos2)
               pos1 = nextLetterRight(pos1, str);
            else
               pos2 = nextLetterRight(pos2, str);
         }
      }
      
      result[0] = Math.min(pos1, pos2);
      result[1] = Math.max(pos1, pos2);
      result[2] = count;
      return result;  
   }
   
   public static int nextLetterRight(int i, String str)
   {
      i++;
      while(i < str.length() && !Character.isLetter(str.charAt(i)))
         i++;
         
      if(i >= str.length())
         return str.length() - 1;
      return i; 
   }
   
   public static int nextLetterLeft(int i, String str)
   {
      i--;
      while(i >= 0 && !Character.isLetter(str.charAt(i)))
         i--;
      
      if(i < 0)
         return 0; 
      return i; 
   }
}