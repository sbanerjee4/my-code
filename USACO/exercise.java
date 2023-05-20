import java.util.*;
import java.io.*;
import java.lang.*;
public class exercise
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("exercise.in"));
      int n = in.nextInt();
      long b = in.nextLong();
      String cows = "";
      for(int i = 0; i < n; i++) cows += in.next();
   
      ArrayList<String> states = new ArrayList<>();
      states.add(cows);
      
      for(int i = 1; i <= b; i++)
      {
         cows = toggle(cows);
         if(states.contains(cows))
         {
            states.add(cows);
            int x = states.indexOf(cows);
            int y = states.lastIndexOf(cows);
            int k = x + (int)((b - x) % (y - x));
            print(states.get(k));
            System.exit(0);
         }
         else
         {
            states.add(cows);
         }
      }
      
      print(cows);
   }
   
   static void print(String s)
   {
      for(char c : s.toCharArray()) System.out.println(c);
   }
   
   static String toggle(String s)
   {
      int[] temp = new int[s.length()];
      for(int i = 0; i < s.length(); i++)
         temp[i] = Integer.parseInt(s.substring(i, i + 1));
      
      int[] temp2 = new int[temp.length];
      for(int i = 0; i < temp.length; i++)
         temp2[i] = temp[i];
      
      for(int i = 1; i < s.length(); i++)
         if(temp[i - 1] == 1) temp2[i] = 1 - temp2[i];
      if(temp[s.length() - 1] == 1) temp2[0] = 1 - temp2[0];
      
      s = "";
      for(int a : temp2)
         s += a + "";
      return s;
   }
}