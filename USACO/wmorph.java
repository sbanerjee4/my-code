import java.io.*;
import java.util.*;
import java.lang.*;
public class wmorph
{
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("wmorph.in"));
      String s1 = in.next(), s2 = in.next();
      String[] dictionary = new String[25000];
      int ind = 0;
      while(in.hasNext())
      {
         String temp = in.next();
         if(temp.length() == s1.length())
         {
            dictionary[ind] = temp;
            ind++;
         }
      }
      in.close();
      
      Queue<String> q1 = new ArrayDeque<>();
      Queue<Integer> q2 = new ArrayDeque<>();
      boolean[] visited = new boolean[25000];
      for(int i = 0; i < dictionary.length; i++)
         if(dictionary[i] != null && dictionary[i].equals(s1)) visited[i] = true;
      q1.add(s1);
      q2.add(0);
      while(!q1.isEmpty())
      {
         String cw = q1.remove();
         int cm = q2.remove();
         if(cw.equals(s2))
         {
            System.out.println(cm);
            System.exit(0);
         }
         for(int i = 0; i < dictionary.length; i++)
         {
            if(dictionary[i] != null && oneAway(dictionary[i], cw) && !visited[i])
            {
               q1.add(dictionary[i]);
               q2.add(cm + 1);
               visited[i] = true;
            }
         }
      }
   }
   
   public static boolean oneAway(String s1, String s2)
   {
      int count = 0;
      for(int i = 0; i < s1.length(); i++)
         if(s1.charAt(i) != s2.charAt(i)) count++;
      if(count == 1) 
         return true;
      return false;
   }
}